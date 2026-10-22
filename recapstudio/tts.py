"""Voice synthesis with drift-free timeline fitting.

The two user visible bugs this module removes
------------------------------------------------
*"dubbing only speaks part of the video"* — the old mixer computed each
line's available window as ``next_start - cursor`` and then clamped it with
``max(0.3, ...)``. The moment one line ran long, every following line was
cut down to a 0.3 second blip (or skipped entirely) and the tail of the video
was left silent. Here every line is *fitted* instead of dropped:

    * natural synthesis first (best quality),
    * if it overruns its window → re-synthesise slightly faster (edge-tts
      "rate" keeps the pitch natural),
    * still overrunning → ``atempo`` time compression (pitch preserving),
    * only as an absolute last resort a fade-out trim, and that case is
      reported to the UI as a warning.

*"rendering takes forever"* — every line used to be synthesised **one at a
time**, each with its own fresh asyncio event loop and websocket handshake.
Lines are now synthesised in a thread pool (default 8 workers) and cached by
(text, voice, rate) so a re-render with small edits only synthesises what
actually changed.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import config
from .media import FFMPEG, get_media_duration, make_silence_wav
from .util import CancelledError, estimate_speech_seconds, get_logger, human_time

log = get_logger("recap.tts")

ProgressFn = Optional[Callable[[float, str], None]]

VOICE_CATALOG: dict[str, dict[str, dict[str, str]]] = {
    "my": {
        "thiha": {"name": "မင်းသန့် (Action Narrator)", "voice": "my-MM-ThihaNeural",
                  "rate": "+0%", "pitch": "-1Hz", "lang": "my"},
        "nilar": {"name": "မေသူ (Drama & Expressive)", "voice": "my-MM-NilarNeural",
                  "rate": "+0%", "pitch": "+1Hz", "lang": "my"},
    },
    "en": {
        "christopher": {"name": "Christopher (Cinematic Male)", "voice": "en-US-ChristopherNeural",
                        "rate": "+0%", "pitch": "-1Hz", "lang": "en"},
        "jenny": {"name": "Jenny (Energetic Female)", "voice": "en-US-JennyNeural",
                  "rate": "+0%", "pitch": "+1Hz", "lang": "en"},
        "guy": {"name": "Guy (Documentary Male)", "voice": "en-US-GuyNeural",
                "rate": "+0%", "pitch": "+0Hz", "lang": "en"},
        "aria": {"name": "Aria (Calm Narrator)", "voice": "en-US-AriaNeural",
                 "rate": "+0%", "pitch": "+0Hz", "lang": "en"},
    },
}

PHONETICS_MM = {
    r"\bAI\b": "အေအိုင်", r"\bFBI\b": "အက်ဖ်ဘီအိုင်", r"\bCIA\b": "စီအိုင်အေ",
    r"\bVIP\b": "ဗွီအိုင်ပီ", r"\bCEO\b": "စီအီးအို", r"\bOK\b": "အိုကေ",
    r"\bTikTok\b": "တစ်တော့ခ်", r"\bFacebook\b": "ဖေ့စ်ဘွတ်ခ်", r"\bYouTube\b": "ယူကျု့ဘ်",
    r"\bSubscribe\b": "ဆပ်စခရိုက်ဘ်", r"\bLike\b": "လိုက်", r"\bPolice\b": "ရဲ",
    r"\bDoctor\b": "ဒေါက်တာ",
}


# ── speaking-rate budget (v4.3.1) ────────────────────────────────────────
#: Measured edge-tts output at rate "+0%" (natural, unhurried delivery).
#: Burmese neural voices speak ~10.5 characters per second, English ~14.5
#: (≈2.6 words/s). Everything that plans *how much text* may sit inside a
#: window derives from these two numbers, so the AI, the subtitle writer and
#: the voice engine all agree instead of each guessing.
CHARS_PER_SECOND = {"my": 10.5, "en": 14.5}
#: how much faster than natural we are willing to read before it sounds rushed
COMFORT_SPEEDUP = 1.12


def chars_per_second(lang: str = "my") -> float:
    return CHARS_PER_SECOND.get((lang or "my").lower()[:2], CHARS_PER_SECOND["my"])


def budget_chars(seconds: float, lang: str = "my", comfort: bool = True) -> int:
    """How many characters fit in ``seconds`` of natural narration."""
    rate = chars_per_second(lang) * (COMFORT_SPEEDUP if comfort else 1.0)
    return max(8, int(max(0.0, seconds) * rate))


#: sentence / clause boundaries we may cut at, best first
_CUT_POINTS = ("။", ".", "!", "?", "၊", ",", ";", ":", " ")


def condense_line(text: str, seconds: float, lang: str = "my") -> tuple[str, bool]:
    """Shorten ``text`` so it can be *spoken calmly* inside ``seconds``.

    The #2 complaint — "narration is too wordy so the voice races to fit" —
    is caused by the model writing 3 sentences for a 4 second window. Speeding
    the take up (atempo) is the wrong fix: it sounds robotic. Instead the line
    is cut back to its budget at the nearest natural boundary **before** TTS,
    and the subtitle uses the same shortened text, so audio and subtitle stay
    identical.

    Returns ``(text, condensed?)``.
    """
    text = (text or "").strip()
    if not text:
        return text, False
    limit = budget_chars(seconds, lang)
    if len(text) <= limit:
        return text, False
    head = text[:limit]
    cut = -1
    for mark in _CUT_POINTS:
        pos = head.rfind(mark)
        # only accept a boundary that keeps at least 55% of the budget,
        # otherwise we would throw away most of the sentence
        if pos > limit * 0.55:
            cut = max(cut, pos + (0 if mark == " " else len(mark)))
    short = (head[:cut] if cut > 0 else head).strip(" ,;:၊")
    if not short:
        short = head.strip()
    if lang == "my":
        if not short.endswith(("။", "!", "?")):
            short += "။"
    elif not short.endswith((".", "!", "?")):
        short += "."
    return short, True


def fit_lines_to_windows(dialogues: list[dict], duration: float,
                         lang: str = "my", *, respect_end: bool = False) -> tuple[list[dict], int]:
    """Condense every line that cannot be spoken calmly inside its window.

    The window of a line ends where the next line starts (that is also how
    :meth:`VoiceEngine.plan_lines` fits the audio), so this is the single
    place where "too much text for the time available" is solved — for the
    voice *and* the burned-in subtitle at once.
    """
    ordered = sorted(
        [d for d in (dialogues or []) if str(d.get("text", "")).strip()],
        key=lambda d: float(d.get("start", 0.0) or 0.0),
    )
    condensed = 0
    for idx, item in enumerate(ordered):
        start = max(0.0, float(item.get("start", 0.0) or 0.0))
        nxt = float(ordered[idx + 1].get("start", duration)) if idx + 1 < len(ordered) else duration
        declared_end = float(item.get("end", nxt) or nxt)
        window_end = min(nxt, duration, declared_end) if respect_end else min(nxt, duration)
        window = max(0.4, window_end - start - 0.06)
        if respect_end:
            # Dubbing must preserve every source utterance. If translation or
            # TTS cannot fit the actual speech interval, fail safely in the
            # strict synthesis pass and ask for a human edit; never summarize
            # or silently drop words to make the clock appear to fit.
            continue
        text, changed = condense_line(str(item.get("text", "")), window, lang)
        if changed:
            item["text"] = text
            item["condensed"] = True
            condensed += 1
    return ordered, condensed


def clean_script_line(text: str, lang: str = "my") -> str:
    """Strip anything that must never be spoken aloud."""
    if not text:
        return ""
    t = re.sub(r"[၀-၉0-9]+:[၀-၉0-9]+(\s*-\s*[၀-၉0-9]+:[၀-၉0-9]+)?", "", text)
    t = re.sub(r"(?m)^\s*[၀-၉0-9]+[\.\)။\-]\s*", "", t)
    t = re.sub(r"[\(\[（【].*?[\)\]）】]", "", t, flags=re.DOTALL)
    t = re.sub(r"[*#_~>`]", "", t)
    t = re.sub(r"[\r\n]+", " ", t)
    if lang == "my":
        for pattern, replacement in PHONETICS_MM.items():
            t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
        # curated pronunciation.txt (700+ acronyms / foreign words) so the
        # narrator does not switch into an English accent mid-sentence
        try:
            from .lexicon import lexicon
            if t.isascii():
                # pure Latin line: pronounce the whole line with the lexicon
                t = lexicon.apply_pronunciation(t)
            else:
                # mixed line: only translate the Latin tokens inside it
                t = re.sub(
                    r"[A-Za-z][A-Za-z0-9&.\-]{1,23}",
                    lambda m: lexicon.apply_pronunciation(m.group(0)) or m.group(0),
                    t,
                )
        except Exception:
            pass
    t = re.sub(r"\s+", " ", t).strip()
    return t


@dataclass
class PlannedLine:
    index: int
    start: float
    end: float
    text: str
    target_seconds: float
    #: absolute ceiling for the produced audio: the next line's start (so the
    #: timeline can never drift later and cut the end of the video) or the
    #: video length for the final line.
    hard_limit_seconds: float = 0.0
    tempo: float = 1.0
    tts_rate: str = "+0%"
    speech_seconds: float = 0.0
    trimmed: bool = False
    skipped: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "index": self.index, "start": round(self.start, 3), "end": round(self.end, 3),
            "target_seconds": round(self.target_seconds, 3),
            "hard_limit_seconds": round(self.hard_limit_seconds, 3),
            "speech_seconds": round(self.speech_seconds, 3),
            "tempo": round(self.tempo, 3), "trimmed": self.trimmed, "skipped": self.skipped,
            "reason": self.reason, "text": self.text,
        }


@dataclass
class MixResult:
    audio_path: str
    duration: float
    planned: list[PlannedLine] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: windows of the finished audio that contain no speech at all
    gaps: list[dict] = field(default_factory=list)
    spoken_seconds: float = 0.0

    def stats(self) -> dict:
        trimmed = sum(1 for p in self.planned if p.trimmed)
        sped = sum(1 for p in self.planned if p.tempo > 1.02 or p.tts_rate != "+0%")
        skipped = sum(1 for p in self.planned if p.skipped)
        max_gap = max((g["end"] - g["start"] for g in self.gaps), default=0.0)
        return {
            "lines": len(self.planned),
            "trimmed_lines": trimmed,
            "speed_adjusted_lines": sped,
            "skipped_lines": skipped,
            "audio_duration": round(self.duration, 2),
            "spoken_seconds": round(self.spoken_seconds, 2),
            "silent_seconds": round(max(0.0, self.duration - self.spoken_seconds), 2),
            "max_silence_seconds": round(max_gap, 2),
            "silent_windows": [{"start": round(g["start"], 1), "end": round(g["end"], 1)}
                               for g in self.gaps[:20]],
            "warnings": self.warnings[:10],
        }


def _rate_to_percent(rate: str) -> float:
    m = re.match(r"([+-]?\d+(?:\.\d+)?)%", rate or "+0%")
    return float(m.group(1)) if m else 0.0


def _percent_to_rate(value: float) -> str:
    return f"{value:+.0f}%"


def _hash_key(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]


class TTSEngine:
    """edge-tts based synthesis with caching and time fitting."""

    def __init__(self, cache_dir: Path = config.CACHE_DIR, workers: int | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workers = max(1, workers or config.settings.tts_workers)
        self.fake = bool(config.settings.fake_tts)
        self._voice_locks: dict[str, threading.Lock] = {}

    # ── raw synthesis ──────────────────────────────────────────────────
    def _voice_lock(self, name: str) -> threading.Lock:
        with threading.RLock():
            return self._voice_locks.setdefault(name, threading.Lock())

    def synthesize(self, text: str, voice_cfg: dict[str, str], out_mp3: Path,
                   rate_override: str | None = None) -> bool:
        """Produce an mp3 for ``text``. Returns True on success."""
        if self.fake:
            self._fake_speech(text, voice_cfg.get("lang", "my"), out_mp3)
            return out_mp3.exists()

        import edge_tts

        rate = rate_override or voice_cfg.get("rate", "+0%")
        pitch = voice_cfg.get("pitch", "+0Hz")
        voice_name = voice_cfg["voice"]
        attempts = config.settings.tts_retries
        for attempt in range(attempts):
            try:
                async def _run() -> None:
                    communicate = edge_tts.Communicate(text=text, voice=voice_name,
                                                       rate=rate, pitch=pitch)
                    await communicate.save(str(out_mp3))

                asyncio.run(_run())
                if out_mp3.exists() and out_mp3.stat().st_size > 200:
                    return True
            except Exception as exc:
                msg = str(exc)
                log.warning("TTS attempt %d/%d failed for voice %s: %s",
                            attempt + 1, attempts, voice_name, msg[:200])
                # A freshly opened websocket right after a burst of parallel
                # calls occasionally gets throttled - back off a little.
                import time
                time.sleep(0.6 * (attempt + 1))
            finally:
                if out_mp3.exists() and out_mp3.stat().st_size <= 200:
                    out_mp3.unlink(missing_ok=True)
        return False

    def _fake_speech(self, text: str, lang: str, out_mp3: Path) -> None:
        """Deterministic beep track the same length as the estimated speech."""
        seconds = max(0.7, estimate_speech_seconds(text, lang))
        try:
            subprocess.run([
                FFMPEG, "-hide_banner", "-nostdin", "-y",
                "-f", "lavfi", "-i", f"sine=frequency=320:sample_rate=44100:duration={seconds:.2f}",
                "-af", "volume=0.25", "-c:a", "libmp3lame", "-b:a", "96k", str(out_mp3),
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=60)
        except Exception as exc:
            log.error("fake tts failed: %s", exc)

    # ── fitting ────────────────────────────────────────────────────────
    def _convert(self, src: str | Path, dst: Path, tempo: float, max_seconds: float | None) -> bool:
        filters = []
        if abs(tempo - 1.0) > 0.005:
            # atempo must stay inside 0.5-2.0; chain for larger factors
            remaining = tempo
            while remaining > 2.0:
                filters.append("atempo=2.0")
                remaining /= 2.0
            filters.append(f"atempo={remaining:.4f}")
        args = [FFMPEG, "-hide_banner", "-nostdin", "-y", "-i", str(src)]
        if filters:
            args += ["-filter:a", ",".join(filters)]
        args += ["-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le"]
        if max_seconds:
            args += [
                "-af", ((",".join(filters) + ",") if filters else "") +
                       f"afade=t=out:st={max(0.0, max_seconds - 0.12):.3f}:d=0.12",
                "-t", f"{max_seconds:.3f}",
                "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
            ]
        args.append(str(dst))
        try:
            subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           check=True, timeout=180)
            return dst.exists() and dst.stat().st_size > 500
        except Exception as exc:
            log.warning("audio convert failed (%s): %s", dst.name, exc)
            return False

    # ── planning ───────────────────────────────────────────────────────
    def plan_lines(self, dialogues: list[dict], duration: float,
                   lang: str = "my", *, strict_timing: bool = False) -> list[PlannedLine]:
        cleaned: list[tuple[float, float, str]] = []
        for item in sorted(dialogues, key=lambda d: float(d.get("start", 0.0))):
            text = clean_script_line(str(item.get("text", "")), lang)
            try:
                start = max(0.0, float(item.get("start", 0.0)))
            except (TypeError, ValueError):
                continue
            if not text or start >= duration:
                continue
            cleaned.append((start, float(item.get("end", start + 2.5) or start + 2.5), text))

        planned: list[PlannedLine] = []
        guard = 0.06
        for idx, (start, end, text) in enumerate(cleaned):
            # The window a line may occupy = until the next line starts. We do
            # NOT cap it by the model's own (often optimistic) `end` value:
            # speech is never stretched, so a longer window simply means the
            # take is used at its natural speed instead of being compressed.
            next_start = cleaned[idx + 1][0] if idx + 1 < len(cleaned) else None
            if strict_timing:
                # Dialogue dubbing owns only the real speech span from the
                # source, never the silence before a later speaker turn.
                window_end = min(end, next_start if next_start is not None else duration,
                                 duration)
                window_end = max(start + 0.4, window_end)
                target = max(0.4, window_end - start - guard)
                hard = max(0.4, window_end - start - 0.02)
            else:
                # Recap narration can use the gap up to the next planned line.
                window_end = next_start if next_start is not None else duration
                target = max(0.4, min(window_end, duration) - start - guard)
                # Hard ceiling: a take may NEVER run past the next line's start.
                if next_start is not None:
                    hard = max(0.4, next_start - start - 0.02)
                else:
                    hard = max(target, min(max(0.4, duration - 0.05 - start), target + 3.0))
            planned.append(PlannedLine(index=idx, start=start, end=end, text=text,
                                       target_seconds=target, hard_limit_seconds=hard))
        return planned

    def _fit_to_window(self, line: PlannedLine, voice_cfg: dict[str, str],
                       work_dir: Path, tag: str, *, strict_timing: bool = False) -> Optional[Path]:
        """Synthesise ``line`` and make sure it fits ``line.target_seconds``.

        Strategy (best sounding first):
          1. natural take
          2. a faster take at synthesis time (keeps pitch/timbre natural)
          3. pitch preserving ``atempo`` compression
          4. fade-out trim (last resort - reported as a warning)
        """
        target = max(0.4, line.target_seconds)
        raw = work_dir / f"raw_{tag}_{line.index}.mp3"
        if not self.synthesize(line.text, voice_cfg, raw):
            line.skipped = True
            line.reason = "TTS failed (voice/network)"
            return None
        raw_seconds = get_media_duration(raw)
        if raw_seconds <= 0.05:
            raw.unlink(missing_ok=True)
            line.skipped = True
            line.reason = "empty audio"
            return None
        line.speech_seconds = raw_seconds

        # Dialogue dubbing is intentionally conservative: fit each line to
        # its own spoken interval, allow at most a small 12% pace adjustment,
        # and fail the dub rather than silently cutting words or running late.
        natural_tolerance = 1.02 if strict_timing else 1.03
        if raw_seconds <= target * natural_tolerance:
            line.tempo = 1.0
            line.reason = ""
        else:
            ratio = raw_seconds / target
            if strict_timing:
                boost = min(12.0, max(2.0, (ratio - 1.0) * 100.0 + 1.0))
                faster = work_dir / f"raw_{tag}_{line.index}_fast.mp3"
                applied_boost = 0.0
                if self.synthesize(line.text, voice_cfg, faster,
                                   rate_override=_percent_to_rate(boost)):
                    fast_seconds = get_media_duration(faster)
                    if fast_seconds > 0.05:
                        raw.unlink(missing_ok=True)
                        raw, raw_seconds = faster, fast_seconds
                        line.speech_seconds = fast_seconds
                        line.tts_rate = _percent_to_rate(boost)
                        applied_boost = boost
                        line.reason = f"အသံ {boost:.0f}% ဖြင့် အနည်းငယ် ချိန်ထားပါသည်"
                        ratio = raw_seconds / target
                    else:
                        faster.unlink(missing_ok=True)
                        line.tts_rate = "+0%"
                else:
                    faster.unlink(missing_ok=True)
                    line.tts_rate = "+0%"
                # `rate_override` and atempo multiply. Budget their product so
                # combined speed-up never exceeds 12%, even when both stages
                # are needed; otherwise a nominal 12% adjustment could turn
                # into a very rushed ~25% take.
                tempo_cap = 1.12 / (1.0 + applied_boost / 100.0)
                line.tempo = min(tempo_cap, max(1.0, ratio))
                if line.tempo > 1.02:
                    line.reason += f" · pitch-preserving {line.tempo:.2f}x"
                if raw_seconds / line.tempo > target * 1.02:
                    raw.unlink(missing_ok=True)
                    line.skipped = True
                    line.reason = "spoken text cannot fit this dialogue window at a natural pace"
                    return None
            else:
                # Recap lines may use more expressive time fitting; they are
                # not locked to source-speaker lip timing.
                if ratio <= 1.35:
                    boost = min(45.0, (ratio - 1.0) * 100.0 + 5.0)
                    line.tts_rate = _percent_to_rate(boost)
                    faster = work_dir / f"raw_{tag}_{line.index}_fast.mp3"
                    if self.synthesize(line.text, voice_cfg, faster, rate_override=line.tts_rate):
                        fast_seconds = get_media_duration(faster)
                        if fast_seconds > 0.05:
                            raw.unlink(missing_ok=True)
                            raw, raw_seconds = faster, fast_seconds
                            line.speech_seconds = fast_seconds
                            line.reason = f"အသံ {boost:.0f}% မြန်စွာ ဖတ်ထားပါသည်"
                            ratio = raw_seconds / target
                        else:
                            faster.unlink(missing_ok=True)
                if ratio > 1.03:
                    line.tempo = min(config.settings.max_tempo, ratio)
                    if line.tempo > 1.02:
                        extra = (f" + atempo {line.tempo:.2f}x" if line.tts_rate != "+0%"
                                 else f"atempo {line.tempo:.2f}x")
                        line.reason = (line.reason + extra) if line.reason else extra
                if line.tempo >= config.settings.max_tempo and raw_seconds / line.tempo > target * 1.02:
                    line.trimmed = True
                    line.reason += " • အစွန်း အနည်းငယ် ဖြတ်ထားပါသည်"

        out_wav = work_dir / f"clip_{tag}_{line.index}.wav"
        # Hard ceiling is the next speaker start (or source speech end in strict mode).
        hard_limit = max(0.4, line.hard_limit_seconds or target)
        if not self._convert(raw, out_wav, line.tempo, hard_limit):
            line.trimmed = True
            if not self._convert(raw, out_wav, line.tempo, min(target, hard_limit)):
                raw.unlink(missing_ok=True)
                line.skipped = True
                line.reason = "audio conversion failed"
                return None
        raw.unlink(missing_ok=True)
        produced = get_media_duration(out_wav)
        if produced > hard_limit + 0.05:
            # ffmpeg occasionally overshoots by a frame; trim once more so the
            # assembly below is exact.
            if not self._convert(out_wav, out_wav.with_suffix(".trim.wav"), 1.0, hard_limit):
                line.trimmed = True
            else:
                out_wav.with_suffix(".trim.wav").replace(out_wav)
            produced = get_media_duration(out_wav)
        line.speech_seconds = produced
        return out_wav

    # ── main entry ─────────────────────────────────────────────────────
    def build_narration(self, dialogues: list[dict], voice_cfg: dict[str, str],
                        duration: float, work_dir: Path, tag: str = "job",
                        progress: ProgressFn = None,
                        cancel: Optional[Callable[[], bool]] = None,
                        fallback_voice_cfg: dict[str, str] | None = None,
                        strict_timing: bool = False) -> MixResult:
        lang = voice_cfg.get("lang", "my")
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        planned = self.plan_lines(dialogues, duration, lang, strict_timing=strict_timing)
        warnings: list[str] = []

        if not planned:
            if strict_timing:
                raise RuntimeError("Dialogue Dubbing: no source dialogue lines were detected; refusing to render a silent dub.")
            silence = work_dir / f"narration_{tag}.wav"
            make_silence_wav(duration, silence)
            if progress:
                progress(100, "အသံဖိုင် အလွတ် ဖြစ်နေပါသည် (စကားပြောခန်း မတွေ့ပါ)")
            warnings.append("စကားပြောခန်း မတွေ့ပါ - အသံလိုင်း အလွတ် ဖြစ်နေပါသည်။ "
                            "AI Analysis ကို ပြန်လုပ်ကြည့်ပါ။")
            return MixResult(str(silence), duration, [], warnings)

        total = len(planned)
        if progress:
            progress(2, f"🎙️ အသံလိုင်း {total} ခု စတင် သွင်းနေပါသည် ({self.workers} parallel workers)...")

        clips: dict[int, Path] = {}
        done = 0

        def synth_one(line: PlannedLine) -> tuple[int, Optional[Path]]:
            if cancel and cancel():
                raise CancelledError("အလုပ်ကို ရပ်တန့်လိုက်ပါပြီ")
            cache_key = _hash_key(voice_cfg["voice"], voice_cfg.get("rate", "+0%"),
                                  voice_cfg.get("pitch", "+0Hz"), line.text,
                                  f"{line.target_seconds:.2f}",
                                  f"{line.hard_limit_seconds:.2f}",
                                  "strict" if strict_timing else "flexible")
            cached = self.cache_dir / f"{cache_key}.wav"
            if cached.exists() and cached.stat().st_size > 500:
                cached_seconds = get_media_duration(cached)
                if (not strict_timing or
                        (cached_seconds <= line.hard_limit_seconds + 0.05 and
                         cached_seconds <= line.target_seconds * 1.02)):
                    line.speech_seconds = cached_seconds
                    return line.index, cached
                log.warning("discarding cached dubbing line that exceeds its source window: %s", cached.name)
                cached.unlink(missing_ok=True)
            produced = self._fit_to_window(line, voice_cfg, work_dir, tag,
                                           strict_timing=strict_timing)
            if produced is None and fallback_voice_cfg:
                # A line that fails on one voice usually succeeds on the other
                # (edge-tts throttles individual voices) - better than a hole
                # in the narration.
                line.skipped = False
                line.reason = "အသံ အခြားတစ်ခုဖြင့် ပြန်သွင်းထားပါသည်"
                produced = self._fit_to_window(line, fallback_voice_cfg, work_dir, f"{tag}_alt",
                                               strict_timing=strict_timing)
            if produced is None:
                return line.index, None
            tmp_cache = cached.with_suffix(".tmp.wav")
            try:
                shutil.copyfile(produced, tmp_cache)
                os.replace(tmp_cache, cached)
            except Exception as exc:
                log.debug("cache write failed: %s", exc)
                tmp_cache.unlink(missing_ok=True)
            return line.index, produced

        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="tts") as pool:
            futures = [pool.submit(synth_one, line) for line in planned]
            for future in as_completed(futures):
                if cancel and cancel():
                    for f in futures:
                        f.cancel()
                    raise CancelledError("အလုပ်ကို ရပ်တန့်လိုက်ပါပြီ")
                try:
                    idx, path = future.result()
                except Exception as exc:
                    log.warning("tts worker failed: %s", exc)
                    continue
                if path is not None:
                    clips[idx] = path
                done += 1
                if progress:
                    progress(2 + (done / total) * 78,
                             f"🎙️ အသံသွင်းပြီး {done}/{total} လိုင်း "
                             f"({human_time(duration * (done / total))} / {human_time(duration)})")

        skipped = [line for line in planned if line.skipped or line.index not in clips]
        strict_trims = [line for line in planned if strict_timing and line.trimmed]
        if strict_timing and (skipped or strict_trims):
            failed = skipped + [line for line in strict_trims if line not in skipped]
            labels = ", ".join(f"#{line.index + 1} ({line.start:.1f}s)" for line in failed[:6])
            raise RuntimeError(
                f"Dialogue Dubbing အချိန်ကိုက် စစ်ဆေးမှု မအောင်မြင်ပါ — "
                f"လိုင်း {len(failed)} ခုကို သဘာဝကျသောနှုန်းဖြင့် မဖြည့်နိုင်ပါ ({labels})။ "
                "အသံလွတ်/စကားပြတ်/နောက်ကျသော output မထုတ်ရန် render ကို ရပ်လိုက်ပါသည်။ "
                "Timeline Editor တွင် ထိုစာကြောင်းကို တိုတိုပြင်ပါ သို့မဟုတ် speech window ကို တိုးပါ။"
            )
        if skipped and not clips:
            # Every single line failed: telling the user "some lines failed"
            # here would be misleading, the video would be mute.
            raise RuntimeError(
                f"အသံသွင်း၍ မရပါ (လိုင်း {len(skipped)} ခုလုံး မအောင်မြင်ပါ)။ "
                "Server မှ Microsoft Edge TTS (speech.platform.bing.com) သို့ ချိတ်ဆက်နိုင်/မနိုင် စစ်ပါ "
                "သို့မဟုတ် အခြား Narrator အသံကို ရွေးစမ်းပါ။"
            )
        if skipped:
            warnings.append(f"အသံသွင်း၍ မရသော လိုင်း {len(skipped)} ခု (voice/network) - "
                            "ကျန် လိုင်းများ ဆက်လက် ထွက်ရှိပါမည်။")
            log.warning("skipped %d lines during TTS", len(skipped))

        # ── assemble: silence gaps + clips, exactly `duration` long ─────
        if progress:
            progress(82, "🧵 Timeline အသံများ ပေါင်းစပ်နေပါသည်...")
        segments: list[Path] = []
        cursor = 0.0
        silent_windows: list[dict] = []
        placed: list[tuple[float, float]] = []
        if planned and planned[0].start > 0.4:
            silent_windows.append({"start": 0.0, "end": round(planned[0].start, 2)})
        for line in planned:
            clip = clips.get(line.index)
            if clip is None:
                continue
            gap = line.start - cursor
            if gap > 0.02:
                gap_file = work_dir / f"gap_{tag}_{line.index}.wav"
                make_silence_wav(gap, gap_file)
                segments.append(gap_file)
                cursor = line.start
            elif gap < -0.02:
                # The fitting above guarantees this cannot happen; if it ever
                # did we would rather log it than silently drift out of sync.
                log.warning("timeline overlap of %.3fs at line %d", -gap, line.index)
                cursor = line.start
            segments.append(clip)
            length = get_media_duration(clip) or line.target_seconds
            if length > (line.hard_limit_seconds or length) + 0.05:
                # safety net: never let one line push the rest of the timeline
                log.warning("clip %d overruns its window by %.2fs - trimming",
                            line.index, length - line.hard_limit_seconds)
                length = line.hard_limit_seconds
                line.trimmed = True
            placed.append((line.start, line.start + length))
            cursor += length

        if cursor < duration - 0.02:
            tail = work_dir / f"tail_{tag}.wav"
            make_silence_wav(duration - cursor, tail)
            segments.append(tail)

        # Report the windows that really are silent so the caller can fill
        # them (AI gap-fill → re-synthesis) instead of shipping a video where
        # the narrator stops for a minute.
        for idx, (start, end) in enumerate(placed):
            if idx + 1 < len(placed):
                next_start = placed[idx + 1][0]
                if next_start - end > 1.2:
                    silent_windows.append({"start": round(end, 2), "end": round(next_start, 2)})
            elif duration - end > 1.2:
                silent_windows.append({"start": round(end, 2), "end": round(duration, 2)})

        final_wav = work_dir / f"narration_{tag}.wav"
        self._concat(segments, final_wav, duration)
        for seg in segments:
            if seg.parent == work_dir and seg.name.startswith(("gap_", "tail_")):
                seg.unlink(missing_ok=True)

        if progress:
            progress(100, f"🎧 အသံ ပြီးစီးပါပြီ ({human_time(get_media_duration(final_wav))})")

        trimmed = sum(1 for line in planned if line.trimmed)
        if trimmed:
            warnings.append(f"လိုင်း {trimmed} ခုကို အချိန်နှင့် ကိုက်ညီစေရန် အနည်းငယ် ဖြတ်ခဲ့ပါသည်")
        spoken = sum(max(0.0, e - s) for s, e in placed)
        return MixResult(str(final_wav), get_media_duration(final_wav), planned, warnings,
                         gaps=silent_windows, spoken_seconds=spoken)

    def _concat(self, segments: list[Path], out_path: Path, duration: float) -> None:
        if not segments:
            make_silence_wav(duration, out_path)
            return
        list_file = out_path.with_suffix(".txt")
        with open(list_file, "w", encoding="utf-8") as fh:
            for seg in segments:
                safe = str(Path(seg).resolve()).replace("'", "'\\''")
                fh.write(f"file '{safe}'\n")
        try:
            subprocess.run([
                FFMPEG, "-hide_banner", "-nostdin", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-t", f"{duration:.3f}",
                "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(out_path),
            ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=900)
        except subprocess.CalledProcessError as exc:
            tail = (exc.stderr or b"").decode("utf-8", "ignore")[-1200:]
            raise RuntimeError(f"အသံ ပေါင်းစပ်၍ မရပါ: {tail}") from exc
        finally:
            list_file.unlink(missing_ok=True)


def master_audio(src_wav: str | Path, out_mp3: str | Path, duration: float,
                 loudnorm: bool = True, bitrate: str = "192k") -> str:
    """Encode the mixed narration to mp3 (loudness normalised for platforms)."""
    filters = "loudnorm=I=-16:TP=-1.5:LRA=11" if loudnorm else "anull"
    try:
        subprocess.run([
            FFMPEG, "-hide_banner", "-nostdin", "-y", "-i", str(src_wav),
            "-af", filters, "-c:a", "libmp3lame", "-b:a", bitrate,
            "-t", f"{duration:.3f}", str(out_mp3),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=900)
        return str(out_mp3)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"").decode("utf-8", "ignore")[-800:]
        log.warning("mp3 mastering failed (%s) - using plain encode", tail)
        subprocess.run([
            FFMPEG, "-hide_banner", "-nostdin", "-y", "-i", str(src_wav),
            "-c:a", "libmp3lame", "-b:a", bitrate, "-t", f"{duration:.3f}", str(out_mp3),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=900)
        return str(out_mp3)


tts_engine = TTSEngine()
