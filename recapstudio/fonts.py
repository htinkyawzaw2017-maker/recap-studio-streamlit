"""Myanmar font resolution.

Bug that used to exist
----------------------
* ``Pyidaungsu.ttf`` was committed to the repository as a **1 byte file**, so
  every Myanmar glyph fell back to a font without Myanmar coverage and the
  burned-in subtitles rendered as tofu boxes.
* The "fix" downloaded a font from raw.githubusercontent at import time which
  (a) blocked application start-up for up to a minute when the network was
  slow and (b) silently produced no font at all on locked-down networks /
  blocked egress (very common on AWS private subnets).

Now: two Noto Sans Myanmar weights (Regular + Bold, SIL OFL licensed) ship
inside ``assets/fonts`` and are used directly. Network download is only a
last-ditch fallback when the assets are missing.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

from . import config
from .util import get_logger

log = get_logger("recap.fonts")

#: Font family name that libass/fontconfig must resolve.
MM_FONT_FAMILY = "Noto Sans Myanmar"

_REGULAR = config.BUNDLED_FONT_DIR / "NotoSansMyanmar-Regular.ttf"
_BOLD = config.BUNDLED_FONT_DIR / "NotoSansMyanmar-Bold.ttf"

_REMOTE_MIRRORS = (
    "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansMyanmar/NotoSansMyanmar-Regular.ttf",
    "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSansMyanmar/NotoSansMyanmar-Regular.ttf",
)

# The subtitle filter needs a directory libass can scan; the bundled assets
# folder is stable regardless of the current working directory.
FONTS_DIR = str(config.BUNDLED_FONT_DIR)

_resolved: Optional[str] = None


def _valid_font(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 40_000:
            return False
        from PIL import ImageFont
        ImageFont.truetype(str(path), 40)
        return True
    except Exception:
        return False


def _download_fallback() -> Optional[Path]:
    if os.getenv("RECAP_ALLOW_FONT_DOWNLOAD", "1") != "1":
        return None
    target = config.BUNDLED_FONT_DIR / "NotoSansMyanmar-Regular.ttf"
    target.parent.mkdir(parents=True, exist_ok=True)
    for url in _REMOTE_MIRRORS:
        tmp = target.with_suffix(".tmp")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "recap-studio/4.0"})
            with urllib.request.urlopen(req, timeout=20) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            if _valid_font(tmp):
                os.replace(tmp, target)
                log.info("Downloaded Myanmar font fallback from %s", url)
                return target
            tmp.unlink(missing_ok=True)
        except Exception as exc:
            log.warning("Font mirror failed (%s): %s", url, exc)
            tmp.unlink(missing_ok=True)
    return None


def myanmar_font_path(bold: bool = False) -> Optional[str]:
    """Absolute path of a usable Myanmar font file (or None)."""
    global _resolved
    preferred = _BOLD if bold else _REGULAR
    if _valid_font(preferred):
        return str(preferred)
    other = _REGULAR if bold else _BOLD
    if _valid_font(other):
        return str(other)
    if _resolved and _valid_font(Path(_resolved)):
        return _resolved
    downloaded = _download_fallback()
    if downloaded:
        _resolved = str(downloaded)
        return _resolved
    # System fonts as a very last resort (some base images ship Padauk).
    for candidate in (
        "/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.ttf",
        "/usr/share/fonts/truetype/padauk/Padauk-Regular.ttf",
        "/usr/share/fonts/noto/NotoSansMyanmar-Regular.ttf",
    ):
        if _valid_font(Path(candidate)):
            log.warning("Using system Myanmar font %s", candidate)
            return candidate
    log.error("No Myanmar capable font found - subtitles may show boxes")
    return None


def get_mm_pil_font(size: int, bold: bool = False):
    """PIL font with Myanmar coverage, falling back safely."""
    from PIL import ImageFont
    path = myanmar_font_path(bold=bold)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception as exc:
            log.warning("PIL could not load %s: %s", path, exc)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def font_status() -> dict:
    reg = myanmar_font_path()
    return {
        "family": MM_FONT_FAMILY,
        "regular": reg,
        "bold": myanmar_font_path(bold=True),
        "fonts_dir": FONTS_DIR,
        "ok": bool(reg),
        "bundled_ok": _valid_font(_REGULAR) or _valid_font(_BOLD),
    }
