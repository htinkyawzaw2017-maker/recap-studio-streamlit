"""Audio-led retiming: the voice stays natural, the picture yields.

The old engine forced every narration line into its original window by
speeding the voice up (edge-tts rate boosts + ``atempo``), which is exactly
what made the narration race on one line and crawl on the next. The rule
now (owner's decision, 2026-10-22):

* the voice is ALWAYS synthesised and used at 1.0x — never sped up,
  never slowed down, never trimmed to fit;
* a line's slot on the output timeline is as long as the speech needs;
* the source video inside the slot is slowed down to match, but never
  below ``MIN_VIDEO_SPEED`` (0.9x);
* if even 0.9x is not enough, the last frame is frozen (held) for the
  remaining time — sync is exact no matter how long the video is;
* spans without speech (gaps between lines) keep their original 1.0x
  length, so pauses stay natural.

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

#: Slowest the picture may go to meet a long line of speech (owner rule).
MIN_VIDEO_SPEED = 0.9


@dataclass
class Span:
    """One contiguous piece of the source video on the output timeline."""
    src_start: float
    src_end: float
    #: natural speech length inside this span (0.0 = silent gap span)
    speech: float = 0.0
    line_index: int = -1
    speed: float = 1.0
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
        """Output length: slowed picture + optional frozen tail."""
        return self.src_len / self.speed + self.freeze


def compute_spans(windows: list[tuple[float, float, float, int]],
                  duration: float, *, tighten_pause: float | None = None,
                  tighten_threshold: float = 1.2) -> list[Span]:
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
        speed, freeze, dropped = 1.0, 0.0, 0.0
        span_end = end
        if speech > src_len and src_len > 0.05:
            needed = src_len / speech
            if needed >= MIN_VIDEO_SPEED:
                speed = needed
            else:
                speed = MIN_VIDEO_SPEED
                freeze = max(0.0, speech - src_len / MIN_VIDEO_SPEED)
        elif (tighten_pause is not None and speech > 0.05
              and src_len - speech > tighten_threshold):
            keep = speech + tighten_pause
            if keep < src_len:
                span_end = start + keep
                dropped = end - span_end
        spans.append(Span(src_start=start, src_end=span_end, speech=speech,
                          line_index=line_index, speed=speed, freeze=freeze,
                          dropped=dropped))
        cursor = end
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
        if span.freeze > 0.05:
            chain += f",tpad=stop_mode=clone:stop_duration={span.freeze:.3f}"
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
            "frozen_seconds": round(sum(s.freeze for s in spans), 2)}
