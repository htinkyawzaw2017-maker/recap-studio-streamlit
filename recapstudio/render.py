"""Final master render: reframe + subtitles + logo + audio mix.

Performance notes
-----------------
* A **stream copy fast path** is used whenever the requested output needs no
  pixel work (original aspect ratio, no logo, no burned-in subtitles). On a
  1 GB input that turns a multi-minute re-encode into a few seconds.
* The old code always re-encoded at ``-threads 0`` *and* re-encoded audio
  twice (concat mp3 → render mp4). We now do a single encode pass, expose a
  quality/speed preset and report ffmpeg's own progress + ETA to the UI.
* If the subtitle filter fails (a libass/font problem is the classic cause)
  the render is retried **without** burned-in subtitles instead of failing
  the whole job - the user still gets a video plus a .srt/.ass file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from . import config
from . import media
from .fonts import FONTS_DIR
from .media import get_video_info, run_ffmpeg
from .subtitles import ass_filter_arg
from .util import CancelledError, get_logger

log = get_logger("recap.render")

ProgressFn = Optional[Callable[[float, str], None]]

QUALITY_PRESETS = {
    "fast": {"preset": "veryfast", "crf": 23, "audio": "160k", "max_height": 1280},
    "balanced": {"preset": "faster", "crf": 21, "audio": "192k", "max_height": 1280},
    "quality": {"preset": "medium", "crf": 19, "audio": "224k", "max_height": 1920},
}

REFRAME_MODES = {
    "Smart Blur Background": "blur",
    "Center Crop": "crop",
    "Original (no reframe)": "none",
}


def _video_filter_chain(reframe: str, width: int, height: int,
                        ass_path: str | None, logo_path: str | None,
                        logo_pos: tuple[float, float] | None,
                        logo_input_index: int = 2) -> tuple[str, str]:
    """Return (filter_chain, final_video_label).

    ``logo_input_index`` must match the position of the logo ``-i`` argument
    (input 0 = source video, 1 = narration audio, 2 = logo). Getting this
    wrong is what produced "Invalid file index 2 in filtergraph".
    """
    parts: list[str] = []
    mode = REFRAME_MODES.get(reframe, "blur")
    if mode == "blur":
        parts.append(
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=24:8,eq=brightness=-0.18[bg];"
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]"
        )
    elif mode == "crop":
        parts.append(
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}[v0]"
        )
    else:
        parts.append("[0:v]null[v0]")

    current = "v0"
    label_index = 1
    if ass_path:
        nxt = f"v{label_index}"
        parts.append(f"[{current}]{ass_filter_arg(ass_path, FONTS_DIR)}[{nxt}]")
        current, label_index = nxt, label_index + 1
    if logo_path:
        nxt = f"v{label_index}"
        x_expr, y_expr = "main_w-overlay_w-24", "24"
        if logo_pos is not None:
            px = max(0.0, min(100.0, float(logo_pos[0]))) / 100.0
            py = max(0.0, min(100.0, float(logo_pos[1]))) / 100.0
            x_expr = f"(main_w-overlay_w)*{px:.4f}"
            y_expr = f"(main_h-overlay_h)*{py:.4f}"
        logo_w = max(48, int(width * 0.16))
        parts.append(f"[{logo_input_index}:v]scale={logo_w}:-1:flags=bicubic,format=rgba[logo];"
                     f"[{current}][logo]overlay={x_expr}:{y_expr}:format=auto[{nxt}]")
        current, label_index = nxt, label_index + 1
    return ";".join(parts), current


def _mix_two(label_a: str, label_b: str, out_label: str, post_gain: float = 1.0) -> str:
    """``amix`` that keeps the levels intact on every ffmpeg version.

    ``normalize=0`` only exists from ffmpeg 5.1; older builds always divide
    by the number of inputs, so the same gain is applied afterwards instead.
    """
    opts = "inputs=2:duration=first:dropout_transition=0"
    gain = post_gain
    if media.ffmpeg_filter_supports("amix", "normalize"):
        opts += ":normalize=0"
    else:
        gain *= 2.0
    chain = f"[{label_a}][{label_b}]amix={opts}"
    if abs(gain - 1.0) > 0.001:
        chain += f",volume={gain:.2f}"
    return f"{chain}[{out_label}]"


def _audio_filter(input_has_audio: bool, mute_original: bool, keep_original_level: float,
                  narration_gain: float, use_ducking: bool) -> tuple[str, str]:
    if not input_has_audio or mute_original or keep_original_level <= 0.001:
        return f"[1:a]volume={narration_gain:.2f}[aout]", "aout"
    if use_ducking:
        return (
            "[1:a]asplit[ai_main][ai_key];"
            "[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            f"volume={keep_original_level:.2f}[sfx];"
            "[sfx][ai_key]sidechaincompress=threshold=0.02:ratio=12:attack=8:release=600[sfx_ducked];"
            + _mix_two("sfx_ducked", "ai_main", "aout", post_gain=narration_gain)
        ), "aout"
    return (
        "[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
        f"volume={keep_original_level:.2f}[sfx];"
        "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
        f"volume={narration_gain:.2f}[ai];"
        + _mix_two("sfx", "ai", "aout")
    ), "aout"


def render_master(input_video: str, narration_wav: str, output_video: str, *,
                  ass_path: str | None = None,
                  logo_path: str | None = None,
                  logo_pos: tuple[float, float] | None = None,
                  reframe: str = "Smart Blur Background",
                  mute_original: bool = True,
                  keep_original_level: float = 0.12,
                  narration_gain: float = 1.25,
                  quality: str = "balanced",
                  target_size: tuple[int, int] | None = None,
                  duration: float | None = None,
                  progress: ProgressFn = None,
                  cancel: Optional[Callable[[], bool]] = None,
                  owner: str = "",
                  force_reencode: bool = False) -> dict:
    """Render the final MP4. Returns metadata incl. whether subs were burned."""
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["balanced"])
    if quality == "custom":
        preset = dict(preset)
        preset["preset"] = config.settings.video_preset
        preset["crf"] = config.settings.video_crf
        preset["audio"] = config.settings.audio_bitrate

    info = get_video_info(input_video)
    duration = duration or info["duration"] or 30.0
    out_w, out_h = target_size or (720, 1280)
    mode = REFRAME_MODES.get(reframe, "blur")

    needs_video_work = bool(ass_path) or bool(logo_path) or mode != "none" or force_reencode
    warnings: list[str] = []
    burned_subs = bool(ass_path)

    if not needs_video_work:
        # Nothing to draw: copy both streams (or add the new audio track).
        audio_filter, audio_label = _audio_filter(
            info["has_audio"], mute_original, keep_original_level, narration_gain, True)
        cmd = [
            "-y", "-i", input_video, "-i", narration_wav,
            "-filter_complex", audio_filter,
            "-map", "0:v:0", "-map", f"[{audio_label}]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", preset["audio"],
            "-t", f"{duration:.3f}", "-movflags", "+faststart",
            "-max_muxing_queue_size", "1024", output_video,
        ]
        log.info("render: stream-copy fast path (no visual changes requested)")
        run_ffmpeg(cmd, total_duration=duration, label="🎬 Video copy", progress=progress,
                   timeout=3600, check=True, cancel=cancel, owner=owner)
        return {"path": output_video, "duration": duration, "burned_subtitles": False,
                "fast_path": True, "warnings": warnings}

    valid_logo = logo_path if (logo_path and Path(logo_path).exists()) else None

    def build_cmd(with_subs: bool) -> list[str]:
        chain, vlabel = _video_filter_chain(
            reframe, out_w, out_h,
            ass_path if with_subs else None,
            valid_logo,
            logo_pos,
            logo_input_index=2,
        )
        audio_filter, audio_label = _audio_filter(
            info["has_audio"], mute_original, keep_original_level, narration_gain, True)
        inputs = ["-i", input_video, "-i", narration_wav]
        if valid_logo:
            inputs += ["-i", valid_logo]
        return [
            "-y", *inputs,
            "-filter_complex", f"{chain};{audio_filter}",
            "-map", f"[{vlabel}]", "-map", f"[{audio_label}]",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", preset["preset"], "-crf", str(preset["crf"]),
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
            "-r", "30",
            "-c:a", "aac", "-b:a", preset["audio"], "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart", "-max_muxing_queue_size", "1024",
            output_video,
        ]

    try:
        run_ffmpeg(build_cmd(True), total_duration=duration, label="🎬 Render",
                   progress=progress, timeout=None, check=True, cancel=cancel, owner=owner)
    except CancelledError:
        # A user cancel (⏹ ရပ်မည်) must never trigger the "render again without
        # subtitles" fallback — that would start a second ffmpeg run for a job
        # the user just stopped, and the UI would look like cancel did nothing.
        raise
    except Exception as exc:
        if not ass_path:
            raise
        log.error("render with subtitles failed, retrying without burn-in: %s", exc)
        warnings.append(
            "စာတန်းထိုး Burn-in လုပ်ရာတွင် ပြဿနာတက်သဖြင့် စာတန်းမပါဘဲ Render လုပ်လိုက်ပါသည် "
            "(SRT/ASS ဖိုင် သီးသန့် ရရှိပါမည်)။"
        )
        burned_subs = False
        run_ffmpeg(build_cmd(False), total_duration=duration, label="🎬 Render (no subs)",
                   progress=progress, timeout=None, check=True, cancel=cancel, owner=owner)

    return {
        "path": output_video,
        "duration": duration,
        "burned_subtitles": burned_subs,
        "fast_path": False,
        "warnings": warnings,
        "size": os.path.getsize(output_video) if os.path.exists(output_video) else 0,
    }


def export_srt(dialogues: list[dict], srt_path: str | Path, hook: bool = False) -> Path:
    """Plain SRT next to the master file (editable, platform friendly)."""
    srt_path = Path(srt_path)

    def ts(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        ms_total = int(round(seconds * 1000))
        h, rem = divmod(ms_total, 3600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    out: list[str] = []
    for idx, item in enumerate(dialogues, start=1):
        try:
            start = float(item.get("start", 0))
            end = float(item.get("end", start + 2))
        except (TypeError, ValueError):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        out.append(f"{idx}\n{ts(start)} --> {ts(max(end, start + 0.6))}\n{text}\n")
    srt_path.write_text("\n".join(out), encoding="utf-8")
    return srt_path
