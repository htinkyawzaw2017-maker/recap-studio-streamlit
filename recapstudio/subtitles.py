"""ASS subtitle / hook builder.

Fixes vs. the previous implementation
-------------------------------------
* ASS text fields were written straight from model output: a ``{`` or ``}``
  (or a stray ``\\N``) corrupted the dialogue line and could make libass
  render nothing at all for the rest of the file. Text is now escaped.
* The font family name was hard-coded even when the font was missing, which
  is what produced tofu boxes. The resolved font family is used now.
* PlayRes is taken from the real output frame so subtitles scale identically
  on 9:16, 1:1 and 16:9 renders.
* The hook banner used to cover the whole video; its duration is now
  configurable and the banner is optional.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .fonts import MM_FONT_FAMILY
from .util import get_logger

log = get_logger("recap.subtitles")

_FORBIDDEN = re.compile(r"[{}\\]")
_TAG_LIKE = re.compile(r"\\[Nnh]|\\t\([^)]*\)")


def escape_ass_text(text: str) -> str:
    """Make arbitrary user/model text safe for an ASS Dialogue field."""
    if not text:
        return ""
    cleaned = str(text).replace("\r", " ").replace("\n", " ")
    cleaned = _TAG_LIKE.sub(" ", cleaned)
    cleaned = _FORBIDDEN.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def hex_to_ass(hex_code: str) -> str:
    """#RRGGBB -> ASS &H00BBGGRR& (ASS stores colours as BGR)."""
    raw = (hex_code or "#00F2FE").lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        raw = "00F2FE"
    try:
        r, g, b = raw[0:2], raw[2:4], raw[4:6]
        int(raw, 16)
    except ValueError:
        r, g, b = "00", "F2", "FE"
    return f"&H00{b}{g}{r}&".upper()


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:d}:{minutes:02d}:{secs:05.2f}"


BG_STYLES = {
    "Solid Box": ("3", "1.6", "0", "&H80000000"),
    "Semi Transparent Box": ("3", "1.2", "0", "&H50000000"),
    "Outline Only": ("1", "4.0", "2", "&H00000000"),
}


def build_ass(dialogues: list[dict[str, Any]], duration: float, ass_path: str | Path,
              *, hook_line1: str = "", hook_line2: str = "",
              font_size: int = 42, font_family: str = MM_FONT_FAMILY,
              v_margin: int = 280, hex_color: str = "#00F2FE",
              bg_style: str = "Solid Box", play_res: tuple[int, int] = (720, 1280),
              hook_seconds: float = 0.0, subtitle_alpha: int = 0,
              stroke_width: float = 0.0, uppercase_hook: bool = False,
              width_percent: int = 90) -> Path:
    """Write an ASS file. ``hook_seconds=0`` means "show for the whole video"."""
    ass_path = Path(ass_path)
    width, height = play_res
    width_percent = max(60, min(96, int(width_percent)))
    side_margin = int(round(width * (100 - width_percent) / 200.0))
    primary = hex_to_ass(hex_color)
    border, outline, shadow, back = BG_STYLES.get(bg_style, BG_STYLES["Solid Box"])
    if stroke_width > 0:
        border, outline, shadow = "1", f"{stroke_width:.1f}", "1"

    primary_with_alpha = f"&H{max(0, min(255, int(subtitle_alpha))):02X}{primary[4:]}"

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "Collisions: Normal",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: SubtitleStyle,{font_family},{max(12, int(font_size))},{primary_with_alpha},"
        f"&H000000FF,&H00101010,{back},-1,0,0,0,100,100,0,0,{border},{outline},{shadow},"
        f"2,{side_margin},{side_margin},{int(v_margin)},1",
        f"Style: HookStyle,{font_family},{max(16, int(width * 0.058))},&H0000FFFF,&H000000FF,"
        f"&H00202020,&HA0000000,-1,0,0,0,100,100,0,0,1,4.0,2,8,{int(width * 0.04)},"
        f"{int(width * 0.04)},{int(height * 0.06)},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    hook1 = escape_ass_text(hook_line1)
    hook2 = escape_ass_text(hook_line2)
    if uppercase_hook:
        hook1, hook2 = hook1.upper(), hook2.upper()
    if hook1 or hook2:
        text = f"{hook1}\\N{hook2}" if hook1 and hook2 else (hook1 or hook2)
        end = duration if hook_seconds <= 0 else min(duration, hook_seconds)
        lines.append(
            f"Dialogue: 1,{_fmt_time(0)},{_fmt_time(end)},HookStyle,,0,0,0,,"
            f"{{\\fad(220,220)\\t(0,180,\\fscx106\\fscy106)\\t(180,340,\\fscx100\\fscy100)}}{text}"
        )

    written = 0
    for item in dialogues:
        try:
            start = max(0.0, float(item.get("start", 0.0)))
            end = float(item.get("end", start + 2.5))
        except (TypeError, ValueError):
            continue
        text = escape_ass_text(item.get("text", ""))
        if not text:
            continue
        if end <= start:
            end = start + 2.0
        end = min(max(end, start + 0.6), max(duration, start + 0.6))
        lines.append(
            f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},SubtitleStyle,,0,0,0,,"
            f"{{\\fad(90,90)\\t(0,110,\\fscx108\\fscy108)\\t(110,230,\\fscx100\\fscy100)}}{text}"
        )
        written += 1

    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("ASS written: %s (%d subtitle lines, hook=%s)", ass_path.name, written,
             bool(hook1 or hook2))
    return ass_path


def ass_filter_arg(ass_path: str | Path, fonts_dir: str | None = None) -> str:
    """Build a correctly escaped ``subtitles=...`` filter argument.

    Escaping rules for ffmpeg filtergraph values are notoriously fiddly
    (``:`` separates options, ``\\`` and ``'`` escape). The previous
    implementation produced an invalid escape sequence for paths containing
    quotes or colons, which aborted the render. This version escapes the
    full set properly and also points libass at the bundled font folder.
    """
    def esc(value: str) -> str:
        value = str(value).replace("\\", "/")
        value = value.replace("'", "\\'")
        value = value.replace(":", "\\:")
        value = value.replace("[", "\\[").replace("]", "\\]")
        value = value.replace(",", "\\,")
        return value

    arg = f"subtitles=filename='{esc(ass_path)}'"
    if fonts_dir:
        arg += f":fontsdir='{esc(fonts_dir)}'"
    arg += ":charenc=UTF-8"
    return arg
