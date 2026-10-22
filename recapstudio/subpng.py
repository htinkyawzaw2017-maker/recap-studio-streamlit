"""Shaped-text subtitle burn-in via PIL + Raqm PNG overlays (v1.4).

Why this exists
---------------
The burned subtitles rendered through libass (ffmpeg ``subtitles=`` filter)
come out UNSHAPED for Myanmar in the deployment environments we ship to:
vowel signs and medials are drawn as separate glyphs instead of combining
with their base consonant, which is exactly the owner's repeated report
"မြန်မာစာ မမှန် / font မမှန်". An A/B render of the same line proved it:
libass output is unshaped both locally and on Streamlit Cloud, while
Pillow's Raqm layout engine (libraqm/HarfBuzz) shapes the identical
string perfectly.

So subtitles are no longer burned with libass. Each line is rendered to a
transparent PNG with Pillow+Raqm (correct Myanmar shaping, stroke + soft
box), the PNGs become a frame-accurate alpha overlay track (concat of
stills, qtrle), and ffmpeg simply overlays that track on the finished
video. The ASS/SRT files are still exported as sidecars; libass remains
only as a fallback when Raqm is unavailable.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from . import fonts as _fonts
from .media import run_ffmpeg
from .util import get_logger

log = get_logger("recap.subpng")


def raqm_available() -> bool:
    """True when Pillow can shape complex text (libraqm present)."""
    try:
        from PIL import features
        return bool(features.check("raqm"))
    except Exception:
        return False


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    path = _fonts.myanmar_font_path(bold=bold) or _fonts.myanmar_font_path()
    if path:
        return ImageFont.truetype(path, size,
                                  layout_engine=ImageFont.Layout.RAQM)
    return ImageFont.load_default()


def _wrap(text: str, font, max_width: int, draw) -> list[str]:
    """Greedy word wrap using shaped text measurement (max 3 lines)."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not cur:
            # a single over-wide word is hard-split by characters
            while draw.textlength(word, font=font) > max_width and len(word) > 1:
                cut = len(word)
                while cut > 1 and draw.textlength(word[:cut], font=font) > max_width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            cur = f"{cur} {word}".strip() if cur else word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines[:3]


def render_text_png(text: str, out_path: str | Path, *, frame_width: int,
                    font_size: int, color: str = "#00F2FE", bold: bool = False,
                    box: bool = True, stroke: int = 0) -> Optional[Path]:
    """One transparent PNG (full frame width) holding the shaped text."""
    from PIL import Image, ImageDraw
    probe = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(probe)
    font = _font(font_size, bold=bold)
    max_text_w = int(frame_width * 0.90)
    lines = _wrap(text, font, max_text_w, draw)
    if not lines:
        return None
    line_h = int(font_size * 1.45)
    pad_x, pad_y = int(font_size * 0.45), int(font_size * 0.28)
    text_w = max(int(draw.textlength(ln, font=font)) for ln in lines)
    img_w = frame_width
    img_h = line_h * len(lines) + pad_y * 2
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if box:
        box_w = text_w + pad_x * 2
        x0 = (img_w - box_w) // 2
        draw.rounded_rectangle([x0, 2, x0 + box_w, img_h - 2], radius=10,
                               fill=(0, 0, 0, 168))
    y = pad_y
    for ln in lines:
        w = int(draw.textlength(ln, font=font))
        x = (img_w - w) // 2
        if stroke > 0:
            draw.text((x, y), ln, font=font, fill=color,
                      stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        else:
            draw.text((x, y), ln, font=font, fill=color)
        y += line_h
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def build_subtitle_track(dialogues: list[dict[str, Any]], *, width: int,
                         height: int, duration: float, work_dir: str | Path,
                         font_size: int = 0, color: str = "#00F2FE",
                         bottom_margin_ratio: float = 0.08) -> Optional[Path]:
    """Alpha overlay track (qtrle .mov) with one still per subtitle state.

    Every dialogue becomes a PNG shown exactly for [start, end]; gaps use
    a fully transparent frame. Returns None when there is nothing to show.
    The track's own height covers only the subtitle band; it is positioned
    by :func:`burn_subtitles` at the bottom margin.
    """
    from PIL import Image
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not font_size:
        font_size = max(26, min(72, int(round(width * 0.052))))
    items = []
    for item in dialogues:
        try:
            start = max(0.0, float(item.get("start", 0.0)))
            end = float(item.get("end", start + 2.0))
        except (TypeError, ValueError):
            continue
        text = str(item.get("text", "")).strip()
        if text and end > start:
            items.append((start, min(end, duration), text))
    items.sort()
    if not items:
        return None
    blank_h = 8
    blank = work_dir / "blank.png"
    Image.new("RGBA", (width, blank_h), (0, 0, 0, 0)).save(blank)
    states: list[tuple[str, float]] = []  # (png, seconds to show)
    cursor = 0.0
    png_h = blank_h
    for idx, (start, end, text) in enumerate(items):
        png = work_dir / f"sub_{idx:04d}.png"
        if render_text_png(text, png, frame_width=width,
                           font_size=font_size, color=color) is None:
            continue
        from PIL import Image as _Im
        png_h = max(png_h, _Im.open(png).size[1])
        if start > cursor + 0.01:
            states.append((str(blank), start - cursor))
        states.append((str(png), end - start))
        cursor = end
    if cursor < duration - 0.01:
        states.append((str(blank), duration - cursor))
    if not states:
        return None
    # re-render every state PNG at a uniform height so the concat track is
    # a well-formed constant-size stream (pad at the top: text sits low).
    from PIL import Image as _Im
    uniform: list[tuple[str, float]] = []
    for path, secs in states:
        im = _Im.open(path)
        if im.size[1] != png_h:
            canvas = _Im.new("RGBA", (width, png_h), (0, 0, 0, 0))
            canvas.paste(im, (0, png_h - im.size[1]))
            fixed = path.replace(".png", "_u.png")
            canvas.save(fixed)
            path = fixed
        uniform.append((path, max(0.04, secs)))
    list_file = work_dir / "subs_concat.txt"
    with open(list_file, "w", encoding="utf-8") as fh:
        for path, secs in uniform:
            fh.write(f"file '{Path(path).resolve()}'\n")
            fh.write(f"duration {secs:.3f}\n")
        fh.write(f"file '{Path(uniform[-1][0]).resolve()}'\n")
    track = work_dir / "subs_track.mov"
    run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-vsync", "vfr", "-pix_fmt", "rgba", "-c:v", "qtrle",
                "-video_track_timescale", "1000", str(track)],
               check=True, label="📝 Subtitle track")
    # the overlay band height is simply the track's own video height,
    # readable by callers via media.get_video_info(track)["height"]
    log.info("subtitle PNG track: %d lines, band height %dpx", len(items), png_h)
    return track


def render_hook_png(hook1: str, hook2: str, out_path: str | Path, *,
                    width: int) -> Optional[Path]:
    """Top banner PNG (yellow, bold, stroked, no box) — shaped like subs."""
    text = " ".join(t for t in (hook1.strip(), hook2.strip()) if t)
    if not text:
        return None
    size = max(16, int(width * 0.058))
    return render_text_png(text, out_path, frame_width=width, font_size=size,
                           color="#FFFF00", bold=True, box=False, stroke=3)


def burn_subtitles(video_in: str, video_out: str, *, track: Optional[Path],
                   track_height: int, height: int,
                   hook_png: Optional[Path] = None, hook_seconds: float = 0.0,
                   duration: float = 0.0,
                   progress=None) -> dict:
    """Overlay the PNG subtitle track (+ optional hook) onto a video."""
    if track is None and hook_png is None:
        raise ValueError("burn_subtitles: nothing to burn")
    y = max(0, height - track_height - int(height * 0.08)) if track else 0
    inputs = ["-i", str(video_in)]
    filters = []
    cur = "0:v"
    idx = 1
    if track is not None:
        inputs += ["-i", str(track)]
        filters.append(f"[{cur}][{idx}:v]overlay=0:{y}:eof_action=pass:format=auto[v{idx}]")
        cur = f"v{idx}"
        idx += 1
    if hook_png is not None:
        inputs += ["-loop", "1", "-i", str(hook_png)]
        hy = int(height * 0.05)
        enable = f":enable='between(t,0,{hook_seconds:.3f})'" if hook_seconds > 0 else ""
        filters.append(f"[{cur}][{idx}:v]overlay=0:{hy}{enable}:format=auto[v{idx}]")
        cur = f"v{idx}"
        idx += 1
    cmd = ["-y", *inputs, "-filter_complex", ";".join(filters),
           "-map", f"[{cur}]", "-map", "0:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "copy",
           "-movflags", "+faststart", str(video_out)]
    run_ffmpeg(cmd, total_duration=duration or None, label="📝 Burn subtitles",
               progress=progress, timeout=None, check=True)
    return {"path": str(video_out), "method": "png-raqm"}
