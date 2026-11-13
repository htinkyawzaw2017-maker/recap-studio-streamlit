"""Audio-led retiming: the voice stays natural, the picture yields.

The old engine forced every narration line into its original window by
speeding the voice up (edge-tts rate boosts + ``atempo``), which is exactly
what made the narration race on one line and crawl on the next. The rule
now (owner decisions, latest v1.30):

* the voice is synthesised at 1.0x and used in full — text is NEVER cut;
* when a line needs more time than its window has, FIRST the picture
  slows, to at most ``MIN_VIDEO_SPEED`` (0.60x since v1.30 — the owner
  accepts slow-motion down to 0.60x);
* if even 0.60x is not enough, the VOICE may speed up to
  ``MAX_VOICE_TEMPO`` (1.2x — still natural sounding, no words dropped);
* FREEZE IS BANNED (owner, v1.30): the last frame is never held. If the
  voice still needs more room, the picture simply keeps playing into the
  following footage at 1.0x — the next line's slot shifts later, the
  story never stops moving;
* spans without speech keep their original 1.0x length, so pauses stay
  natural.

``compute_spans`` is pure arithmetic (unit-testable); ``retime_video``
renders the retimed, video-only MP4 that ``render_master`` then mixes
with the retimed narration master.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .media import get_video_info, run_ffmpeg
from .util import get_logger

log = get_logger("recap.retime")

#: Slowest the picture may go to meet a long line of speech (owner rule,
#: v1.30: 0.60x — slow-motion is acceptable, a frozen frame is not;
#: override with RECAP_RETIME_MIN_SPEED between 0.5 and 0.95).
def _min_speed() -> float:
    import os
    try:
        value = float(os.getenv("RECAP_RETIME_MIN_SPEED", "0.60"))
    except (TypeError, ValueError):
        value = 0.60
    return max(0.5, min(0.95, value))


MIN_VIDEO_SPEED = _min_speed()


#: Fastest the VOICE may be compressed to fit (owner rule, v1.30: 1.2x —
#: a gentle read, no words cut). Override with RECAP_MAX_VOICE_TEMPO.
def _max_tempo() -> float:
    import os
    try:
        value = float(os.getenv("RECAP_MAX_VOICE_TEMPO", "1.2"))
    except (TypeError, ValueError):
        value = 1.2
    return max(1.0, min(1.5, value))


MAX_VOICE_TEMPO = _max_tempo()


@dataclass
class Span:
    """One contiguous piece of the source video on the output timeline."""
    src_start: float
    src_end: float
    #: natural speech length inside this span (0.0 = silent gap span)
    speech: float = 0.0
    line_index: int = -1
    speed: float = 1.0
    #: v1.30: how much the VOICE is compressed for this line (1.0 = natural).
    #: The clip is played through atempo=tempo; text is never cut.
    tempo: float = 1.0
    #: frozen-tail seconds — BANNED since v1.30, always 0.0 (kept for
    #: backwards compatibility of saved plans/tests)
    freeze: float = 0.0
    slot_start: float = 0.0
    #: source seconds dropped after this span (Recap "tighten": the dead
    #: tail of a scene whose narration already finished is not emitted)
    dropped: float = 0.0

    @property
    def src_len(self) -> float:
        return max(0.0, self.src_end - self.src_start)

    @property
    def slot_len(self) -> float:
        """Output length of this span's picture (freeze no longer exists)."""
        return self.src_len / self.speed


def compute_spans(windows: list[tuple[float, float, float, int]],
                  duration: float, *, tighten_pause: float | None = None,
                  tighten_threshold: float = 1.2,
                  fps: float = 30.0) -> list[Span]:
    """Build spans covering ``[0, duration]`` of the source.

    ``windows`` are ``(src_start, src_end, speech_seconds, line_index)``
    sorted by start; the spaces between them become 1.0x gap spans.

    Recap tightening (``tighten_pause`` set): when a line's speech ends
    well before its window does, the rest of that window used to play in
    silence — the listener heard the narrator "speak when he feels like
    it". Now the dead tail of the picture is dropped instead: the slot
    ends ``tighten_pause`` seconds after the speech, and the next line's
    scene starts right away. The voice then runs continuously while every
    line still sits on its own scene. (Dubbing passes ``None``: its
    timing must stay faithful to the source.)

    ``fps`` (v1.27): every slot is snapped to a whole number of video
    frames, so the audio timeline and the rendered video land on identical
    boundaries instead of drifting apart by half a frame per span.

    v1.30 budgeting (owner rules — no freeze, ever, and no extra footage):

    1. voice fits the window  → picture 1.0x
    2. voice is longer        → the PICTURE SLOWS, down to
       ``MIN_VIDEO_SPEED`` (0.60x) — the only lever, and it is only speed,
       never a frozen frame
    3. still not enough       → the VOICE is compressed up to
       ``MAX_VOICE_TEMPO`` (1.2x — no words are cut, only the pace)
    4. still not enough       → the line simply takes the time it needs:
       the next line's slot starts after it (its own window stays on the
       source footage). No freeze, no duplicated/extended footage — only
       the original picture, moving at 0.60x–1.0x.
    """
    spans: list[Span] = []
    cursor = 0.0
    drop_next_gap = False
    for start, end, speech, line_index in sorted(windows):
        start = min(float(start), duration)
        end = max(start, min(float(end), duration))
        if drop_next_gap:
            # previous span was tightened: skip the footage in between too
            cursor = max(cursor, start)
            drop_next_gap = False
        start = max(cursor, start) if spans or cursor > 0 else start
        if start > cursor + 0.01:
            spans.append(Span(src_start=cursor, src_end=start))
            cursor = start
        src_len = end - start
        speech = max(0.0, float(speech))
        speed, tempo, dropped = 1.0, 1.0, 0.0
        span_end = end
        tightened = False
        if speech > src_len and src_len > 0.05:
            needed = src_len / speech
            if needed >= MIN_VIDEO_SPEED:
                speed = needed            # picture slows, voice stays natural
            else:
                speed = MIN_VIDEO_SPEED
                pic_time = src_len / MIN_VIDEO_SPEED
                # v1.30: picture at the 0.60x floor is still not enough →
                # gently speed the VOICE up to 1.2x. NO freeze, NO extra
                # footage: the line simply takes as long as it takes, and
                # the next line's slot starts right after it.
                tempo = min(MAX_VOICE_TEMPO, max(1.0, speech / pic_time))
                # effective spoken length after the gentle speed-up — the
                # slot is sized for this so the clip lands inside its slot
                # instead of overrunning it by the tempo ratio.
                speech = speech / tempo
        elif (tighten_pause is not None and speech > 0.05
              and src_len - speech > tighten_threshold):
            keep = speech + tighten_pause
            if keep < src_len:
                span_end = start + keep
                dropped = end - span_end
                tightened = True

        if fps > 0 and span_end > start:
            import math as _m
            real_len = span_end - start
            # frame-grid snapping (see docstring): the source cut and the
            # slot must be whole frames or plan and render disagree
            if speed < 0.999:
                pic_frames = max(1, _m.floor(real_len / speed * fps))
                pic_len = pic_frames / fps
                speed = real_len / pic_len if pic_len > 0 else 1.0
                if speed >= 1.0:      # snapping pushed it back to 1.0x
                    speed = 1.0
                    pic_frames = _m.floor(real_len * fps)
                    pic_len = pic_frames / fps
                    span_end = start + pic_len
                # v1.30: when the snapped picture lands a hair short of the
                # voice (under one frame), take ONE more frame of the
                # ORIGINAL footage — no freeze, the slot then holds the
                # whole spoken line.
                if (pic_frames > 0 and speech > pic_len
                        and speech - pic_len <= 1.0 / fps
                        and start + pic_len + 1.0 / fps <= span_end + 1e-6):
                    pic_frames += 1
                    pic_len = pic_frames / fps
                    speed = real_len / pic_len if pic_len > 0 else 1.0
            else:
                speed = 1.0
                pic_frames = _m.floor(real_len * fps)
                # v1.30: when frame snapping leaves the picture a hair
                # shorter than the voice (under one frame), take ONE more
                # frame of the ORIGINAL footage so the voice sits inside its
                # slot — no freeze, no extra material, just the next frame
                # of the same scene.
                if (pic_frames > 0 and speech > pic_frames / fps
                        and speech - pic_frames / fps <= 1.0 / fps
                        and start + (pic_frames + 1) / fps <= duration + 1e-6):
                    pic_frames += 1
                pic_len = pic_frames / fps
                span_end = start + pic_len
                if not tightened:
                    dropped = 0.0     # sub-frame trim is not a "drop"
        spans.append(Span(src_start=start, src_end=span_end, speech=speech,
                          line_index=line_index, speed=speed, tempo=tempo,
                          dropped=dropped))
        cursor = max(end, span_end)
        drop_next_gap = dropped > 0.0
    if cursor < duration - 0.01 and not drop_next_gap:
        spans.append(Span(src_start=cursor, src_end=duration))
    slot = 0.0
    for span in spans:
        span.slot_start = slot
        slot += span.slot_len
    return spans


def total_duration(spans: list[Span]) -> float:
    return sum(s.slot_len for s in spans)


def retimed_dialogues(dialogues: list[dict], spans: list[Span]) -> list[dict]:
    """Dialogue timings remapped onto the output (retimed) timeline."""
    by_index = {s.line_index: s for s in spans if s.line_index >= 0}
    out: list[dict] = []
    for idx, item in enumerate(dialogues):
        span = by_index.get(idx)
        if span is None:
            continue
        new_item = dict(item)
        new_item["start"] = round(span.slot_start, 3)
        new_item["end"] = round(span.slot_start + max(span.speech, 0.6), 3)
        if span.speed < 0.999:
            new_item["video_speed"] = round(span.speed, 3)
        out.append(new_item)
    return out


def retime_video(input_video: str, spans: list[Span], output_video: str | Path, *,
                 progress: Optional[Callable[[float, str], None]] = None,
                 cancel: Optional[Callable[[], bool]] = None,
                 owner: str = "") -> dict:
    """Render a video-only MP4 whose spans play at the planned speeds.

    Spans that need retiming are few in practice (only over-long lines),
    but the whole timeline is rebuilt so every span boundary is exact.
    """
    info = get_video_info(input_video)
    width = int(info.get("width") or 720)
    height = int(info.get("height") or 1280)
    parts: list[str] = []
    labels: list[str] = []
    for i, span in enumerate(spans):
        chain = (f"[0:v]trim=start={span.src_start:.3f}:end={span.src_end:.3f},"
                 f"setpts=PTS-STARTPTS")
        if abs(span.speed - 1.0) > 0.001:
            chain += f",setpts=PTS/{span.speed:.5f}"
        # v1.30: NO freeze — tpad is gone. The picture only ever changes
        # speed (0.60x–1.0x); it never stops moving.
        chain += f",fps=30,scale={width}:{height},format=yuv420p,setsar=1[v{i}]"
        parts.append(chain)
        labels.append(f"[v{i}]")
    total = total_duration(spans)
    if not parts:
        raise ValueError("retime: no spans to render")
    filter_complex = ";".join(parts) + ";" + "".join(labels) + \
        f"concat=n={len(labels)}:v=1:a=0[outv]"
    cmd = [
        "-y", "-i", input_video,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", "30",
        "-t", f"{total:.3f}",
        "-movflags", "+faststart", "-max_muxing_queue_size", "1024",
        str(output_video),
    ]
    run_ffmpeg(cmd, total_duration=total, label="🎞️ Retime",
               progress=progress, timeout=None, check=True,
               cancel=cancel, owner=owner)
    return {"path": str(output_video), "duration": total,
            "spans": len(spans),
            "slowed": sum(1 for s in spans if s.speed < 0.999),
            "sped_voice": sum(1 for s in spans if s.tempo and s.tempo > 1.005),
            "frozen_seconds": 0.0}
