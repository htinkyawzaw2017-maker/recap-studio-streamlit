"""Burmese lexicon helpers (pronunciation + storytelling vocabulary).

The repository ships two curated dictionaries that the old Whisper based app
never actually used:

* ``pronunciation.txt`` — 700+ mappings such as ``FBI = အက်ဖ်ဘီအိုင်``. Edge
  TTS happily reads Latin acronyms with an English accent in the middle of a
  Burmese sentence, which sounds wrong; mapping them to Burmese phonetic
  spellings makes the dub sound like a native narrator.
* ``dictionary.txt`` — natural storytelling connectors
  (``However = ဒါပေမဲ့``) that are fed to the AI as a style reference.

Both are plain ``key = value`` text files, so they can be edited without
touching code.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from . import config
from .util import get_logger

log = get_logger("recap.lexicon")

PRONUNCIATION_FILE = config.APP_ROOT / "pronunciation.txt"
DICTIONARY_FILE = config.APP_ROOT / "dictionary.txt"

_IGNORED_KEYS = {"[NATURAL STORYTELLING VOCABULARY]"}


def _parse_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    try:
        if not path.exists():
            return pairs
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not key or not value or key in _IGNORED_KEYS:
                continue
            if key.lower() == value.lower():
                continue
            if len(key) > 48 or len(value) > 80:
                continue
            pairs.append((key, value))
    except Exception as exc:
        log.warning("could not read %s: %s", path.name, exc)
    return pairs


def _compile_pairs(pairs: Iterable[tuple[str, str]]) -> list[tuple[re.Pattern, str]]:
    compiled: list[tuple[re.Pattern, str]] = []
    # longest keys first so "New York" wins over "New"
    for key, value in sorted(pairs, key=lambda kv: len(kv[0]), reverse=True):
        # Only replace standalone occurrences of the term.
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(key)}(?![A-Za-z0-9])", re.IGNORECASE)
        compiled.append((pattern, value))
    return compiled


class Lexicon:
    def __init__(self) -> None:
        self._pronunciation = _compile_pairs(_parse_pairs(PRONUNCIATION_FILE))
        self._dictionary_pairs = _parse_pairs(DICTIONARY_FILE)
        log.info("lexicon loaded: %d pronunciation rules, %d vocabulary entries",
                 len(self._pronunciation), len(self._dictionary_pairs))

    # ── TTS text normalisation ─────────────────────────────────────────
    def apply_pronunciation(self, text: str, limit: int = 60) -> str:
        if not text:
            return ""
        applied = 0
        out = text
        for pattern, replacement in self._pronunciation:
            if applied >= limit:
                break
            new_out, count = pattern.subn(replacement, out)
            if count:
                applied += count
                out = new_out
        return out

    # ── prompt style hints ─────────────────────────────────────────────
    def connectors_hint(self, limit: int = 30) -> str:
        """A short Burmese connector list the AI can use for natural flow."""
        picked = [f"{en} → {my}" for en, my in self._dictionary_pairs[:limit]]
        return "\n".join(picked)


lexicon = Lexicon()
