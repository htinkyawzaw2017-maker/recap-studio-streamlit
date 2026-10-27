"""Gemini powered timeline extraction with guaranteed full-video coverage.

Root causes fixed here
----------------------
1. **Analysis stopped early.** One single ``generate_content`` call over the
   whole video with ``max_output_tokens=8192`` simply ran out of output
   tokens on anything longer than a few minutes, so the timeline ended
   mid-video and the dub never reached the end.
2. **Timestamps drifted.** With no offset handling, lines from the second
   half of a long video came back relative to nothing in particular.
3. **One giant upload.** The full-resolution source video was uploaded to
   the Files API for every attempt: slow, quota hungry and frequently
   rejected for long clips.

Now the video is analysed in **parallel, frame-accurate low-cost proxies**
(1 fps, 480p, mono 16 kHz - exactly the sampling Gemini uses internally), each
chunk is asked for timestamps *relative to its own start* which are then
shifted by the chunk offset, and a coverage sweep fills whatever windows the
model left empty so the narration really does run from 0 s to the end.
"""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from . import config
from .keys import KeyRing, key_ring as default_key_ring
from .media import run_ffmpeg
from .util import CancelledError, estimate_speech_seconds, get_logger

log = get_logger("recap.ai")

#: errors that mean "this key cannot be used right now" → try the next slot
_KEY_ERROR_MARKERS = (
    "api key not valid", "api_key_invalid", "invalid api key", "permission denied",
    "permission_denied", "unauthenticated", "401", "403", "quota", "resource_exhausted",
    "429", "rate limit", "billing",
)

ProgressFn = Optional[Callable[[float, str], None]]

#: Below this size the proxy is sent inline with the prompt (one request, no
#: Files-API wait). Above it the Files API is used — uploads of small files
#: used to be the source of the "file uri and mime_type are required" failure.
INLINE_PART_LIMIT = 12 * 1024 * 1024

#: extension -> mime for the (always mp4) proxies we build
_VIDEO_MIMES = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
    ".webm": "video/webm", ".avi": "video/x-msvideo", ".mkv": "video/mp4",
    ".ts": "video/mp4", ".flv": "video/mp4", ".3gp": "video/3gpp",
}


def video_mime(path: str | Path) -> str:
    """Mime type for a video part.

    The SDK only fills in ``mime_type`` when it can guess the extension, and a
    part without one is rejected with *"file uri and mime_type are required."*
    — so we always decide the mime ourselves.
    """
    suffix = Path(path).suffix.lower()
    if suffix in _VIDEO_MIMES:
        return _VIDEO_MIMES[suffix]
    try:
        import mimetypes
        guess, _ = mimetypes.guess_type(str(path))
        if guess and guess.startswith("video/"):
            return guess
    except Exception:
        pass
    return "video/mp4"

MODEL_ALIASES = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-1.5-flash": "gemini-2.5-flash",   # retired upstream -> remap
    "gemini-1.5-pro": "gemini-2.5-pro",
    "gemini-2.0-flash-exp": "gemini-2.0-flash",
}

DIALOGUE_SCHEMA = {
    "type": "object",
    "properties": {
        "hook_line1": {"type": "string"},
        "hook_line2": {"type": "string"},
        "dialogues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["start", "end", "text"],
            },
        },
    },
    "required": ["dialogues"],
}

GAP_FILL_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["index", "text"],
            },
        }
    },
    "required": ["lines"],
}

THUMBNAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "number"},
        "headline": {"type": "string"},
        "subheadline": {"type": "string"},
        "text_position": {"type": "string", "enum": ["left", "right"]},
        "rationale": {"type": "string"},
    },
    "required": ["timestamp", "headline", "subheadline", "text_position", "rationale"],
}


@dataclass
class CoverageReport:
    total_seconds: float = 0.0
    spoken_seconds: float = 0.0
    lines: int = 0
    gaps_filled: int = 0
    longest_gap: float = 0.0
    chunks: int = 0
    per_chunk: list[dict] = None  # type: ignore[assignment]
    quiet_gaps: list = None  # type: ignore[assignment]

    def as_dict(self) -> dict:
        pct = (self.spoken_seconds / self.total_seconds * 100.0) if self.total_seconds else 0.0
        return {
            "total_seconds": round(self.total_seconds, 2),
            "spoken_seconds": round(self.spoken_seconds, 2),
            "coverage_percent": round(pct, 1),
            "lines": self.lines,
            "gaps_filled": self.gaps_filled,
            "longest_gap_seconds": round(self.longest_gap, 2),
            "chunks": self.chunks,
            "per_chunk": self.per_chunk or [],
            "quiet_gaps": self.quiet_gaps or [],
        }


# ── SDK shim ─────────────────────────────────────────────────────────────
def _load_sdk():
    try:
        from google import genai  # type: ignore
        return genai, True
    except ImportError:
        try:
            import google.generativeai as legacy  # type: ignore
            return legacy, False
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "google-genai package မတွေ့ပါ။ `pip install google-genai` လုပ်ပေးပါ။"
            ) from exc


class GeminiClient:
    """Thin wrapper handling both the new and legacy Google SDKs.

    When a :class:`~recapstudio.keys.KeyRing` is supplied the client can also
    **switch keys at runtime**: a 429/quota/permission error puts the current
    slot on cooldown and the next key takes over, so a long recap survives a
    key that runs out mid-job (previously the job simply failed).
    """

    def __init__(self, api_key: str = "", key_ring: KeyRing | None = None,
                 on_switch: Optional[Callable[[str], None]] = None):
        self.sdk, self.is_new = _load_sdk()
        self.key_ring = key_ring or (default_key_ring if not api_key else None)
        self.on_switch = on_switch
        self.switches: list[str] = []
        key, slot = self._pick_initial_key(api_key)
        if not key:
            raise ValueError("Gemini API Key ထည့်သွင်းပေးရန် လိုအပ်ပါသည်။ "
                             "⚙️ Settings တွင် Key #1-#3 ထည့်နိုင်ပါသည်။")
        self._apply_key(key, slot)

    def _pick_initial_key(self, api_key: str) -> tuple[str, int]:
        if self.key_ring and self.key_ring.has_key():
            key, slot = self.key_ring.active_key()
            return key, slot or 1
        return (api_key or "").strip(), 1

    def _apply_key(self, key: str, slot: int) -> None:
        self.api_key = key
        self.slot = slot
        self._client = self.sdk.Client(api_key=key) if self.is_new else None  # type: ignore[union-attr]
        if not self.is_new:
            self.sdk.configure(api_key=key)  # type: ignore[attr-defined]

    def rotate_key(self, error: Exception) -> bool:
        """Move to the next healthy key. Returns True when one was found."""
        if not self.key_ring:
            return False
        nxt = self.key_ring.rotate_after_failure(self.slot, str(error), message_fn=self._announce)
        if not nxt:
            return False
        self._apply_key(nxt, self.key_ring.active_index + 1)
        self.switches.append(f"slot {self.slot}")
        return True

    def _announce(self, message: str) -> None:
        if self.on_switch:
            try:
                self.on_switch(message)
            except Exception:
                pass
        log.warning(message)

    # -- file handling -------------------------------------------------
    def _log_upload(self, path, ref, mime: str) -> None:
        uri = getattr(ref, "uri", "") or getattr(ref, "download_uri", "") or ""
        ref_mime = getattr(ref, "mime_type", "") or ""
        log.info("uploaded %s (%s) -> uri=%s mime=%s",
                 Path(path).name, mime, "yes" if uri else "MISSING",
                 ref_mime or f"MISSING (using {mime})")
        if not uri:
            log.warning("Files API returned no uri for %s — part will be retried inline",
                        Path(path).name)

    def build_media_part(self, path: str | Path, label: str = "",
                         progress: ProgressFn = None) -> tuple[Any, Any]:
        """``(part, ref)`` — a video part the SDK accepts on every version.

        Small proxies go inline (fast, no Files-API round trip); bigger ones use
        the Files API and the uri + mime type are re-checked, because passing
        whatever the SDK happened to return is exactly what broke analysis:

        * ``pathlib.Path`` as a part  → "Unsupported content part type" (or an
          empty part on google-genai 1.0.x, i.e. the video silently vanished)
        * ``File`` without ``mime_type`` → "file uri and mime_type are required."
        """
        path = Path(path)
        mime = video_mime(path)
        size = path.stat().st_size
        if size <= INLINE_PART_LIMIT:
            data = path.read_bytes()
            if self.is_new:
                types_mod = getattr(self.sdk, "types", None)
                part = types_mod.Part(inline_data=types_mod.Blob(data=data, mime_type=mime))
            else:  # legacy google-generativeai inline blob
                part = {"mime_type": mime, "data": data}
            log.info("inline part for %s (%.1f MB, %s)", path.name, size / 1048576, mime)
            return part, None

        ref = self.upload(path, progress, label)
        uri = str(getattr(ref, "uri", "") or getattr(ref, "download_uri", "") or "").strip()
        if self.is_new:
            ref_mime = str(getattr(ref, "mime_type", "") or "").strip() or mime
            if not uri:
                raise RuntimeError(
                    "Gemini Files API မှ file uri ပြန်မရပါ — google-genai ကို "
                    "အသစ်တင်ပါ (pip install -U google-genai) သို့မဟုတ် ခေတ္တမျှ ပြန်စမ်းပါ")
            types_mod = getattr(self.sdk, "types", None)
            part = types_mod.Part(file_data=types_mod.FileData(file_uri=uri, mime_type=ref_mime))
            return part, ref
        return ref, ref

    def upload(self, path: str | Path, progress: ProgressFn = None, label: str = ""):
        """Upload through the Files API and wait until it is ACTIVE.

        The mime type is always passed explicitly — an upload whose response
        lacks ``mime_type`` produced the *"file uri and mime_type are
        required."* failure the moment the part was handed back to the SDK.
        """
        mime = video_mime(path)
        if progress:
            progress(8, f"📤 {label or 'ဗီဒီယို'} ကို AI server သို့ ပေးပို့နေပါသည်...")
        if self.is_new:
            try:
                ref = self._client.files.upload(  # type: ignore[union-attr]
                    file=str(path), config={"mime_type": mime})
            except TypeError:  # very old SDK without the config argument
                ref = self._client.files.upload(file=str(path))  # type: ignore[union-attr]
            waited = 0.0
            while True:
                ref = self._client.files.get(name=ref.name)  # type: ignore[union-attr]
                state = str(getattr(ref, "state", "")).upper()
                if "PROCESSING" in state or "PENDING" in state:
                    time.sleep(2)
                    waited += 2
                    if progress:
                        progress(12, f"⏳ AI မှ {label or 'ဗီဒီယို'} ကို ပြင်ဆင်နေပါသည် ({int(waited)}s)...")
                    if waited > 600:
                        raise RuntimeError("AI server မှ ဗီဒီယို ပြင်ဆင်ချိန် ကြာလွန်းနေပါသည်")
                    continue
                if "ACTIVE" in state or state == "":
                    self._log_upload(path, ref, mime)
                    return ref
                raise RuntimeError(f"AI server video processing failed: {state}")
        ref = self.sdk.upload_file(path=str(path))  # type: ignore[attr-defined]
        waited = 0.0
        while True:
            ref = self.sdk.get_file(ref.name)  # type: ignore[attr-defined]
            state = str(getattr(ref.state, "name", ref.state)).upper()
            if "PROCESSING" in state:
                time.sleep(2)
                waited += 2
                if waited > 600:
                    raise RuntimeError("AI server မှ ဗီဒီယို ပြင်ဆင်ချိန် ကြာလွန်းနေပါသည်")
                continue
            if "ACTIVE" in state:
                self._log_upload(path, ref, mime)
                return ref
            raise RuntimeError(f"AI server video processing failed: {state}")

    def delete(self, ref) -> None:
        try:
            if ref is None:
                return
            if self.is_new:
                self._client.files.delete(name=ref.name)  # type: ignore[union-attr]
            else:
                self.sdk.delete_file(ref.name)  # type: ignore[attr-defined]
        except Exception as exc:  # quota cleanup is best effort
            log.debug("file delete failed: %s", exc)

    # -- generation ----------------------------------------------------
    def generate_json(self, model: str, parts: list, schema: dict | None = None,
                      temperature: float = 0.2, max_tokens: int | None = None,
                      progress: ProgressFn = None, cancel: Callable[[], bool] | None = None) -> str:
        model = MODEL_ALIASES.get(model, model)
        max_tokens = max_tokens or config.settings.ai_max_output_tokens
        attempt = 0
        key_switches = 0
        last_error: Exception | None = None
        while attempt < config.settings.ai_retries + key_switches:
            if cancel and cancel():
                raise CancelledError("အလုပ်ကို ရပ်တန့်လိုက်ပါပြီ")
            try:
                if self.is_new:
                    cfg: dict[str, Any] = {"temperature": temperature, "max_output_tokens": max_tokens}
                    if schema:
                        cfg["response_mime_type"] = "application/json"
                        cfg["response_schema"] = schema
                    res = self._client.models.generate_content(  # type: ignore[union-attr]
                        model=model, contents=parts, config=cfg,
                    )
                    text = (getattr(res, "text", "") or "").strip()
                else:
                    legacy_model = self.sdk.GenerativeModel(model)  # type: ignore[attr-defined]
                    gen_cfg: dict[str, Any] = {"temperature": temperature, "max_output_tokens": max_tokens}
                    if schema:
                        gen_cfg["response_mime_type"] = "application/json"
                    res = legacy_model.generate_content(
                        parts,
                        generation_config=self.sdk.types.GenerationConfig(**gen_cfg),  # type: ignore[attr-defined]
                    )
                    text = (getattr(res, "text", "") or "").strip()
                if text:
                    return text
                last_error = RuntimeError("AI မှ အလွတ် ပြန်လာပါသည်")
            except Exception as exc:
                last_error = exc
                msg = str(exc).lower()
                is_key_error = any(k in msg for k in _KEY_ERROR_MARKERS)
                # 1) a dead / quota-limited key is swapped out immediately
                if is_key_error and key_switches < 50 and self.rotate_key(exc):
                    key_switches += 1
                    if progress:
                        progress(20, f"🔑 Key #{self.slot} ဖြင့် ပြန်စမ်းနေပါသည်...")
                    continue
                retryable = any(k in msg for k in ("503", "unavailable", "429", "quota", "overloaded",
                                                   "deadline", "timeout", "500", "internal"))
                if not retryable or attempt >= config.settings.ai_retries - 1:
                    break
                wait = min(30, 4 * (attempt + 1) + (attempt * 2))
                if progress:
                    progress(20, f"⚠️ AI server အလုပ်များနေပါသည် - {wait}s စောင့်ပြီး ပြန်စမ်းပါမည် ({attempt + 1}/{config.settings.ai_retries})")
                time.sleep(wait)
            attempt += 1
        raise RuntimeError(f"AI request failed: {last_error}")


# ── JSON parsing ─────────────────────────────────────────────────────────
def parse_dialogues(raw: str) -> dict:
    """Forgiving JSON parser (models occasionally wrap output in prose)."""
    if not raw:
        raise ValueError("AI ပြန်ကြားချက် အလွတ် ဖြစ်နေပါသည်")
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    payload = None
    if start != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            payload = json.loads(candidate, strict=False)
        except Exception:
            payload = None
    if payload is None:
        payload = _regex_salvage(text)
    if payload is None:
        raise ValueError("AI JSON ပြန်ကြားချက် ဖတ်၍ မရပါ - model ကို ပြောင်း၍ ပြန်စမ်းပါ")

    dialogues = []
    for item in payload.get("dialogues") or payload.get("lines") or []:
        try:
            st = float(item.get("start", 0))
            en = float(item.get("end", 0))
        except (TypeError, ValueError):
            continue
        text_value = str(item.get("text") or item.get("line") or "").strip()
        if not text_value:
            continue
        if en <= st:
            en = st + max(1.0, estimate_speech_seconds(text_value))
        dialogues.append({
            "start": max(0.0, st),
            "end": max(0.0, en),
            "speaker": str(item.get("speaker") or item.get("character") or "").strip()[:40],
            "text": text_value,
        })
    return {
        "hook_line1": str(payload.get("hook_line1") or payload.get("hookLine1") or "").strip(),
        "hook_line2": str(payload.get("hook_line2") or payload.get("hookLine2") or "").strip(),
        "dialogues": dialogues,
    }


def _regex_salvage(text: str) -> Optional[dict]:
    """Last-resort extractor for truncated / malformed model output.

    Long recaps are the usual reason a response is cut off mid-JSON (the
    output-token ceiling), so this must cope with a *missing closing brace*
    and even an unterminated string - otherwise a nearly complete answer
    would be thrown away and the video would be left half dubbed.
    """
    dialogues = []
    for fragment in text.split("{"):
        match_start = re.search(r'"start"\s*:\s*([0-9.]+)', fragment)
        if not match_start:
            continue
        match_end = re.search(r'"end"\s*:\s*([0-9.]+)', fragment)
        match_text = (re.search(r'"(?:text|line)"\s*:\s*"((?:[^"\\]|\\.)*)"', fragment)
                      or re.search(r'"(?:text|line)"\s*:\s*"([^"]*)$', fragment))
        if not match_text:
            continue
        try:
            st = float(match_start.group(1))
            en = float(match_end.group(1)) if match_end else st + 2.5
        except ValueError:
            continue
        line = match_text.group(1).replace('\\"', '"').replace("\\n", " ").strip()
        if line:
            dialogues.append({"start": st, "end": en if en > st else st + 2.5,
                              "speaker": "", "text": line})
    if not dialogues:
        return None
    h1 = re.search(r'"hook_line1"\s*:\s*"([^"]*)"', text)
    h2 = re.search(r'"hook_line2"\s*:\s*"([^"]*)"', text)
    log.warning("JSON salvage recovered %d lines from malformed model output", len(dialogues))
    return {
        "hook_line1": h1.group(1) if h1 else "",
        "hook_line2": h2.group(1) if h2 else "",
        "dialogues": dialogues,
    }


# ── chunk planning ───────────────────────────────────────────────────────
def line_is_terse(text: str, window: float, lang: str = "my") -> bool:
    """A grammatically closed but caption-like line.

    It passes :func:`sentence_complete` (it ends with "။") yet speaks for
    under ~3.2s, or fills under half of a ≥4s window — e.g. "သူဟာ လူတိုင်း။"
    Models write these at the START of a video (titles, establishing
    shots) while the middle gets full prose, which is exactly the
    "beginning chopped, middle full" complaint. Recap narration must read
    as flowing prose everywhere, so terse lines get merged or expanded.

    v1.10: the old ≥4s gate blinded this check for fast-cut drama, where
    segments are 2-3s and caption scraps survive untouched. For short
    windows a line is terse when it speaks under ~3.2s AND fills under
    75% of its window — a full line that uses its whole short window
    is fine, a scrap is not.
    """
    text = (text or "").strip()
    if not text or window <= 0:
        return False
    est = estimate_speech_seconds(text, lang)
    if window >= 4.0:
        return est < 3.2 or est < window * 0.5
    return est < 3.2 and est < window * 0.75


def plan_chunks(duration: float, max_seconds: int | None = None,
                target_chunks: int | None = None) -> list[tuple[float, float]]:
    max_seconds = max_seconds or config.settings.max_chunk_seconds
    target_chunks = target_chunks or config.settings.target_chunks
    if duration <= 0:
        return []
    if duration <= max_seconds:
        return [(0.0, duration)]
    count = max(2, min(target_chunks, int(duration // max_seconds) + 1))
    length = duration / count
    chunks: list[tuple[float, float]] = []
    cursor = 0.0
    for index in range(count):
        end = duration if index == count - 1 else min(duration, cursor + length)
        chunks.append((round(cursor, 3), round(end, 3)))
        cursor = end
    return [(s, e) for s, e in chunks if e - s > 1.0]


def build_proxy(video_path: str | Path, start: float, end: float, out_path: str | Path,
                progress: ProgressFn = None) -> Path:
    """1 fps / 480p / mono 16 kHz proxy - the exact sampling Gemini uses.

    Cuts encode time by ~20x, makes uploads tiny and keeps the chunk start
    frame-accurate (``-ss`` after ``-i``), which is what makes absolute
    timestamps trustworthy.
    """
    length = max(0.5, end - start)
    run_ffmpeg([
        "-y",
        "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{length:.3f}",
        # `force_original_aspect_ratio=decrease` can override the implicit
        # `-2` rounding and produce 480x853 for a portrait 9:16 clip. x264's
        # yuv420p output requires both dimensions to be even. Explicitly round
        # down to multiples of two, then pad defensively before encoding.
        "-vf", "scale=480:-2:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=ceil(iw/2)*2:ceil(ih/2)*2,fps=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-ar", "16000",
        "-movflags", "+faststart",
        str(out_path),
    ], total_duration=length, label="proxy", progress=None, check=True)
    return Path(out_path)


# ── prompts ──────────────────────────────────────────────────────────────
# Genre is deliberately independent from task mode. Previously these prompts
# said "dub every spoken line" even in Recap mode, while the shared coverage
# rules told Dialogue Dubbing to narrate silent scenes. That made both choices
# sound almost identical.
MODE_PROMPTS = {
    "auto": "Identify the video's genre and use its conventions without changing the task mode.",
    "movie": "Genre: film, animation, or drama. Track character motivation and scene causality.",
    "experiment": "Genre: science, experiment, or review. Preserve observable steps and results.",
    "craft": "Genre: DIY, craft, or cooking. Preserve the order of visible actions and instructions.",
    "news": "Genre: news or documentary. Keep claims factual and distinguish what is shown from inference.",
}

FILL_MODE_PROMPTS = {
    "continuous": (
        "MODE = PROFESSIONAL MOVIE RECAP. You are narrating as an experienced movie-recap "
        "storyteller, not dubbing the actors. Watch the visual action in sequence: explain who does "
        "what, why the moment matters, and how it changes the story. Use vivid but natural spoken "
        "language, suspense and smooth transitions. Paraphrase dialogue only when it is essential "
        "to explain the plot; do not translate every spoken line. Cover the complete clip, including "
        "quiet visual scenes, but never invent a fact that is not supported by the video."
    ),
    "dialogue": (
        "MODE = PROFESSIONAL DIALOGUE DUBBING. Act as a professional dubbing translator and "
        "voice director. Listen to the source audio and use it as the authority for what was said; "
        "do not infer dialogue from the picture, subtitles, or plot. Transcribe the meaning internally "
        "then render a faithful, natural spoken translation in the requested target language, keeping "
        "the speaker's intent, emotion, register, and turn-taking. Output ONLY genuine audible speech: "
        "never describe the picture, summarize, add narration, or fill silence. Start at the audible "
        "speech onset and end at the audible speech offset (within 0.3 seconds when the boundary is "
        "clear). Keep each translated line concise enough to finish inside that same speech window; "
        "do not make it fast just to fit."
    ),
}


def _chunk_prompt(chunk_start: float, chunk_end: float, target_language: str,
                  mode_key: str, fill_mode: str, duration: float,
                  boundary_hint: str = "") -> str:
    lang_instruction = (
        "Burmese language (မြန်မာစကားပြော အသုံးအနှုန်းသီးသန့်၊ သဘာဝကျသော မြန်မာစကားဖြင့်)"
        if target_language == "my" else "natural spoken English"
    )
    style_block = ""
    if target_language == "my" and fill_mode == "continuous":
        try:
            from .lexicon import lexicon
            hint = "\n".join(lexicon.connectors_hint(26).splitlines())
            if hint:
                style_block = (
                    "\nNATURAL STORYTELLING CONNECTORS (use sparingly; avoid translationese):\n"
                    + hint + "\n"
                )
        except Exception:
            style_block = ""

    # ── voice-of-the-recap rules (owner demand: 100% natural, scene-exact,
    #    never generic) ──────────────────────────────────────────────────
    selected_mode = fill_mode if fill_mode in FILL_MODE_PROMPTS else "continuous"
    if target_language == "my" and selected_mode == "continuous":
        voice_block = '''
RECAP VOICE — NON-NEGOTIABLE (မြန်မာ):
- လူပြော မြန်မာစကား စစ်စစ် (စကားပြော ဟန်) နဲ့ပဲ ရေးပါ။ စာအုပ်/ရုံးသုံး ဟန်၊ ဘာသာပြန် ဟန် လုံးဝ မသုံးရ။
- ဘာသာပြန် မလုပ်ပါနဲ့ — ပြန်ပြောပါ: မူရင်း ဗီဒီယိုက အင်္ဂလိပ်လို (သို့ အခြား ဘာသာနဲ့) ပြောထားရင် စာကြောင်း တစ်ကြောင်းချင်းကို လိုက်ပြီး တိုက်ရိုက် ပြန်ခြင်း လုံးဝ မလုပ်ရ (အဲဒါက နားထောင်လို့ မကောင်းတဲ့ "တုံးတိ" ဘာသာပြန် ဖြစ်စေတယ်)။ အရင် အဖြစ်အပျက်ကို နားလည်ပါ၊ ပြီးမှ မြန်မာ ဇာတ်လမ်းပြောသူ တစ်ယောက် ကိုယ်ပိုင် စကားနဲ့ ပြန်ပြောသလို ရေးပါ။
- လိုင်းတိုင်းသည် ပြီးပြည့်စုံတဲ့ ဝါကျ ဖြစ်ရမည် — အဆုံးမှာ "။" အမြဲ ပါရမည်။ အတွေးတပိုင်း၊ ကြိယာ မပါတဲ့ အပိုင်းအစ၊ စကားစပ် အဆတ်အပြတ် မရေးရ။
- ဝါကျ တစ်ကြောင်းကို လိုင်း နှစ်လိုင်း ခွဲမရေးရ (အလွန် အရေးကြီး): လိုင်း တစ်ခုချင်းစီကို သီးခြား ဖတ်ကြည့်ရင် ဝါကျ အပြည့် ဖြစ်နေရမည်။ "…တစ်ဦးက။" "…ပါလာတော့။" "…ဘဝသစ်စဖို့။" "…မသိခဲ့ဘဲ။" "…သမုဒ္ဒရာကို။" "…ခေါ်ပြေးခဲ့တာ။" ဆိုတဲ့ subordinate အဆုံးသတ်တွေနဲ့ လိုင်း လုံးဝ မဆုံးရ — အဲဒါ ဇာတ်လမ်း ပြတ်သွားသလို ဖြစ်ပြီး နားထောင် မကောင်းဆုံး အမှားပါ။
- ပြည့်ပြည့်စုံစုံ ရေးပါ: ဇာတ်ဝင် တစ်ချက်စီမှာ ဘယ်သူက၊ ဘာလုပ်တယ်၊ ဘာကြောင့် လုပ်တယ်၊ ဘာဆက်ဖြစ်တယ် ဆိုတာ ပါအောင် ရေးပါ။ သတင်းခေါင်းစဉ် လို အချက် ၄-၅ လုံးတည်း စာရင်းချတာ မလုပ်ရ — နားထောင်သူ မြင်ကွင်းထဲ မြင်လာအောင် ပြည့်စုံတဲ့ ဝါကျ ရေးပါ။
- Hook (hook_line1) သည် ပထမ ဝါကျကတည်းက အန္တရာယ်/ရည်မှန်းချက် (ဘယ်သူက ဘာလိုချင်တယ်၊ မရရင် ဘာဖြစ်မယ်) ကို တန်းချရမည် — "ဒီနေ့တော့ ဇာတ်လမ်းတစ်ပုဒ် ကြည့်ကြရအောင်" လို အစချီ စကား လုံးဝ မသုံးရ။
- Caption ပုံစံ လိုင်း တိုတိုတုတ်တုတ် လုံးဝ မရေးရ: နာမည်/နေရာ/အချိန် ပြောရုံ လိုင်းမျိုး (ဥပမာ "သူဟာ လူတိုင်း။" "အဲဒီအချိန်မှာ။") တားမြစ်တယ် — အဲဒီ အချက်ကို ဘာဖြစ်တယ် ဆိုတာနဲ့ တစ်ဝါကျတည်း ပေါင်းရေးပါ။ လိုင်း တစ်ခုချင်းစီ သူ့ window ရဲ့ စာလုံး budget ရဲ့ အနည်းဆုံး ၇၀% လောက် ပြည့်အောင် ရေးပါ။ အထူးသဖြင့် ဇာတ်လမ်း အစပိုင်း လိုင်းတွေ — အစကို ခေါင်းစဉ်/မြင်ကွင်း ဖတ်သလို တိုတိုချည်း မရေးဘဲ အလယ်ပိုင်းနဲ့ အရည်အသွေး တစ်ပြေးညီ ပြည့်ပြည့်စုံစုံ ရေးပါ။
- ဥပမာ (မရေးရ → ရေးရမည်): "အာဖူက ဘဲကင်ဆိုင်သို့။" → "အာဖူက ဘဲကင်ဆိုင်ကို သွားပြီး ဘဲကင်တစ်ကောင် ဝယ်လိုက်တယ်။" / "အာဖူမျက်လုံးများ။" → "အာဖူက သူ့မျက်လုံးတွေကို ရေနဲ့ ဆေးကြောလိုက်တယ်။" — မူရင်း စာတန်းကို စကားလုံး အတိုင်း လိုက်မပြန်ဘဲ အဖြစ်အပျက်ကို ကိုယ်ပိုင် စကားနဲ့ ဝါကျအပြည့် ပြန်ပြောပါ။
- အပိုင်းအစတွေ မခွဲပါနဲ့: ဆက်စပ်တဲ့ အချက်တွေကို တစ်လိုင်းထဲ သဘာဝကျအောင် ပေါင်းပြောပါ။ ၅-၁၀ စက္ကန့်စာ ဇာတ်ဝင် တစ်ချက် = လိုင်း တစ်လိုင်း ကို ပိုနှစ်သက်ပါ — ၂-၃ စက္ကန့် တိုတို အစအနတွေ အများကြီး မလုပ်ရ။
- လိုင်းတိုင်းသည် အဲဒီ အချိန် screen ပေါ်မှာ တကယ် မြင်နေရတဲ့ အရာ (ဘယ်သူ၊ ဘာလုပ်နေတယ်၊ ဘယ်နေရာ၊ မျက်နှာ အမူအရာ) ကိုပဲ ကွက်တိ ပြောရမည်။ မမြင်ရတာ၊ မသေချာတာ မထည့်ရ။
- တားမြစ် generic စကားများ (လုံးဝ မသုံးရ): "ဇာတ်လမ်း စတင်ပါပြီ", "ဒီဇာတ်လမ်းမှာ", "အခုဆိုရင်", "ဆက်လက် စောင့်ကြည့်ရအောင်", "ဘာတွေ ဖြစ်လာမလဲ", "သူတို့ စကားပြောနေကြပါတယ်" (ဘာအကြောင်း ပြောနေတယ်ဆိုတာ မပါဘဲ), "အံ့ဩစရာ ကောင်းပါတယ်" (အကြောင်းရင်း မပါဘဲ)။
- အသေးစိတ် အချက်အလက် သုံးပါ: လူနာမည်၊ ပစ္စည်း၊ နေရာ၊ လုပ်ရပ်၊ စိတ်ခံစားမှု။ တစ်လိုင်း = ဇာတ်ဝင်ခန်း တစ်ခု။
- သဒ္ဒါ အမှန်: ကို/က/မှာ/ထဲ/ပေါ် နေရာ မှန်မှန် သုံးပါ။ စာကြောင်း အဆုံးသတ် ပုံစံ မျိုးစုံ သုံးပါ (တယ် / ပါတယ် / ပေါ့ / လေ / သွားတယ် / နေတယ် / မိတယ်) — "ဖြစ်ပါတယ်" နဲ့ လိုင်းတိုင်း မဆုံးရ။
- ရိုးရိုး စကားလုံး ရွေးပါ: စာပေ ဆန်တဲ့ ရှားပါး စကားလုံး အစား လူတိုင်း သုံးတဲ့ စကားလုံး။ နံပါတ်/အသက်/ငွေ ကို ပြောရ လွယ်တဲ့ ပုံစံနဲ့ ရေးပါ။
- စာကြောင်း ရှည်ကြီး မရေးရ — တစ်ချက် ပြောရင် အသက်ရှူ မကြပ်တဲ့ အတိုလေးတွေ ခွဲရေးပါ။'''
    elif target_language == "my":
        voice_block = '''
DUBBING VOICE — NON-NEGOTIABLE (မြန်မာ):
- မူရင်း အဓိပ္ပာယ်ကို လူပြော မြန်မာစကား စစ်စစ်နဲ့ ပြန်ပါ — တိုက်ရိုက် ဘာသာပြန် (translationese) လုံးဝ မဖြစ်ရ။ မြန်မာတွေ တကယ် ပြောတဲ့ ပုံစံ၊ သဒ္ဒါ အစီအစဉ် မှန်မှန်။
- ပြောသူရဲ့ စိတ်ခံစားမှု၊ အသံအနိမ့်အမြင့်၊ ရင်းနှီးမှု အဆင့် (ခင်မင်တဲ့ ပုံစံ/ရိုသေတဲ့ ပုံစံ) ကို ထိန်းထားပါ။
- လိုင်းတိုင်းသည် သီးခြား ဖတ်ကြည့်ရင် ဝါကျအပြည့် ဖြစ်ရမည် — ကြိယာ အပြည့်အစုံ (ပါတယ်/တယ်/သည်/ခဲ့/ပြီ/လား) နဲ့ ဆုံးရမည်။ မူရင်းအလှည့် တစ်ခု = ဘာသာပြန်လိုင်း တစ်လိုင်း — အလှည့်တစ်ခုကို နှစ်လိုင်းခွဲမရေးရ၊ အလှည့်နှစ်ခုကို တစ်လိုင်းပေါင်းမရေးရ။ အဲဒီအလှည့်မှာ တကယ်ပြောခဲ့တာကို တိတိကျကျ၊ အပြည့်အစုံ ပြန်ပါ။
- လုံးဝ မဆုံးရ: ကို/မှာ/တာ/ရဲ့/၏/ဆိုတာ/တော့/လည်း/ဖြစ် (တစ်လုံးတည်း) စတဲ့ အပိုင်းအစ အဆုံးသတ်တွေ။ ဥပမာ "…သူတွေကို။" "…ဆိုတာ။" "…အဘွားကြီးရဲ့။" ဆိုတဲ့ လိုင်းမျိုး လုံးဝ မရေးရ။
- "…" / "..." နဲ့ လုံးဝ မဆုံးရ — ဖြတ်ထားတဲ့ စကားကို ဝါကျအပြည့် ဖြစ်အောင် ပြန်ပါ။'''
    elif selected_mode == "continuous":
        voice_block = '''
RECAP VOICE — NON-NEGOTIABLE:
- Write like a real human recap narrator talking, not a summary writer. Every line must describe EXACTLY what is visible on screen in that moment — who, what action, where, what emotion. Never invent or generalise.
- RETELL, never translate: if the source speaks another language, understand the beat and retell it in your own natural words. Sentence-by-sentence literal translation produces wooden, choppy narration and is banned. Every line must be one COMPLETE sentence with final punctuation — no fragments, no thoughts split across two lines. Merge related micro-facts into one flowing 5-10 second line instead of many 2-3 second scraps.
- NEVER split one sentence across two entries, and never end an entry on a subordinate fragment ("...because he", "...to start a new life.", "...the Atlantic Ocean."). Read alone, every entry must still be a full grammatical sentence. Write FULLY: include who did what, why, and what came of it — a listener must be able to picture the beat from the line alone.
- No caption-style scraps: an entry that only names a person, place or time ("He was everyone.", "At that moment.") is banned — fold it into the sentence about what happens. Fill at least ~70% of each entry's character budget, especially the OPENING entries: the beginning must read with the same fullness as the middle, never like reading titles or scene labels.
- BANNED generic filler: "In this video", "Let's see what happens next", "Things take a turn", "Little did he know", "Meanwhile" (without concrete content), or any line that could fit any movie.
- Concrete details only: names, objects, places, actions. One line = one scene beat. Vary sentence endings; keep it punchy and spoken.'''
    else:
        voice_block = '''
DUBBING VOICE — NON-NEGOTIABLE:
- Translate into idiomatic spoken English a real actor would say — never literal/translationese. Keep the speaker's emotion, register and intent.'''

    from .tts import budget_chars, chars_per_second
    cps = chars_per_second(target_language)
    lang_word = "Burmese" if target_language == "my" else "English"
    four_sec, eight_sec = budget_chars(4.0, target_language), budget_chars(8.0, target_language)
    selected_mode = fill_mode if fill_mode in FILL_MODE_PROMPTS else "continuous"
    if selected_mode == "continuous":
        task_rules = f"""RECAP TIMELINE RULES
- Narrate continuously from the beginning to the end of this clip.
- The first entry should start within 4 seconds of {chunk_start:.2f}s; the final entry should reach
  the final scene near {chunk_end:.2f}s. No unexplained narration gap may exceed 4 seconds.
- During quiet scenes, narrate the visible action and its story consequence. This is the only mode
  in which you should add connective narration.
- Make each entry one concise, COMPLETE thought (a full sentence, never a fragment) with a
  start/end window that can be spoken naturally. Prefer fewer, fuller lines of 5-10 seconds
  over many tiny 2-3 second scraps about the same beat."""
    else:
        task_rules = f"""DIALOGUE DUB TIMELINE RULES
- Return an entry ONLY for real speech audible in the source audio. A quiet interval is correct;
  never add a line to satisfy clip coverage.
- Timestamp each utterance from the audio onset to audio offset using absolute full-video seconds.
  Use separate entries for separate speaker turns or clear pauses; do not merge distant speech.
- Translate EXACTLY what is spoken — one original turn = one translated entry, in speech order.
  Do not summarize, retell, merge turns, or leave a turn half-translated; keep names, intent and
  emotion, do not omit important information or add plot explanation.
- One spoken utterance = one COMPLETE translated line, start to finish. Never end a line on a
  dangling particle (က / တဲ့ / များ / တော့ / ကို / မှာ / ကြောင့် / ဖို့) — if the speaker's
  sentence continues, keep translating until the sentence ends with a finished verb.
- Label `speaker` with the character's name or role when identifiable — never the generic word
  'Speaker' for every entry.
- NEVER give two entries the same start time, and NEVER use the clip end or the full-video
  duration as a timestamp — stamp each entry from its own audible onset/offset. If a boundary
  is uncertain, order the entries by speech and space them inside the clip; a pile of entries
  at one timestamp is a defect.
- Never cut a word in half — finish every word fully.
- Keep the translation short enough for its own [start, end] speech window at a calm, natural pace.
  Do not use a later speaker's start as extra time and do not stretch a line into silence."""

    return f"""You are an award-winning film recap writer and professional dubbing director. Follow the
selected task exactly; Recap and Dialogue Dubbing are different products.

VIDEO SEGMENT: this is only part of a longer video. It starts at {chunk_start:.2f}s and ends at
{chunk_end:.2f}s. Full-video duration: {duration:.2f}s.
TARGET SPOKEN LANGUAGE: {lang_instruction}.
GENRE CONTEXT: {MODE_PROMPTS.get(mode_key, MODE_PROMPTS['auto'])}
{FILL_MODE_PROMPTS[selected_mode]}
{style_block}

MANDATORY RULES
1. TIMESTAMPS ARE ABSOLUTE full-video seconds. The first second of this clip is {chunk_start:.2f}s,
   not zero. Keep every line inside {chunk_start:.2f}s–{chunk_end:.2f}s.
2. {task_rules}
3. NATURAL SPEAKING BUDGET: a calm {lang_word} voice speaks about {cps:.0f} characters per second.
   A 4-second window is about {four_sec} characters; 8 seconds is about {eight_sec}. Do not exceed
   the actual line window. In Dubbing mode, preserve meaning first and use a concise, idiomatic
   translation rather than rushing; set both hook fields to empty strings.
3b. PER-MINUTE EXACT FIT (ကွက်တိ): plan your lines in 60-second blocks (0:00-1:00, 1:00-2:00, ...).
   The TOTAL of all lines inside one block must be speakable within that minute — at most about
   {int(60 * cps * 0.92)} {lang_word} characters per block, pauses included. Every line must START
   and END inside the same block; a line may never cross a minute boundary. A line that is too
   long for its scene is a defect: rewrite it shorter, never let the narrator overrun the scene —
   the picture changes but the voice would still be finishing the previous scene, which ruins
   the video. Shorter and exact always beats longer and richer.
4. STYLE: short, clear, active spoken language. Recap mode should feel cinematic, engaging and
   professionally told. Dubbing mode should sound like a skilled actor speaking the translated
   source line, with matching emotion and no narrator voice. Avoid filler and repeated phrases.
5. Do not include timecodes, camera directions, stage directions, or speaker labels inside `text`.
   Never invent names, places, numbers, dialogue, or events. Return STRICT JSON only.
{voice_block}
{boundary_hint}

JSON SHAPE
{{
  "hook_line1": "short truthful hook in the target language",
  "hook_line2": "optional second short hook",
  "dialogues": [
    {{"start": {chunk_start:.2f}, "end": {chunk_start + 4:.2f}, "speaker": "Speaker", "text": "spoken line"}}
  ]
}}"""

def _gap_fill_prompt(gaps: list[dict], lang: str, context_lines: list[dict]) -> str:
    lang_word = "Burmese" if lang == "my" else "English"
    gap_block = "\n".join(
        f"- index {g['index']}: {g['start']:.1f}s → {g['end']:.1f}s "
        f"({g['end'] - g['start']:.1f}s long, roughly {int(g['chars'])} {lang_word} characters)"
        for g in gaps
    )
    context_block = "\n".join(
        f"  [{c['start']:.1f}s] {c['text']}" for c in context_lines
    ) or "  (none)"
    return f"""You are finishing a {lang_word} recap voice-over. Some windows of the timeline have no
line yet and the narration must not fall silent there.

KNOWN LINES AROUND THE GAPS (for continuity only):
{context_block}

WINDOWS TO FILL:
{gap_block}

Rules:
1. Write ONE line per window that fits the requested speaking length (about 9-10 Burmese
   characters per second for Burmese, 13 for English) and never exceeds the window - the line is
   spoken by a narrator whose speed cannot be stretched far.
2. Continue the story smoothly from the surrounding lines. You may summarise or connect what the
   surrounding lines already established, but do NOT invent new names, numbers or events.
3. Plain {lang_word} only: no quotes, no brackets, no timestamps, no stage directions.
4. Return STRICT JSON only: {{"lines": [{{"index": 0, "text": "..."}}]}}"""


# ── demo mode ────────────────────────────────────────────────────────────
_DEMO_SENTENCES = [
    "ဒီဇာတ်လမ်းရဲ့ အစကတည်းက စိတ်ဝင်စားစရာတွေ ဖြစ်လာတယ်။",
    "ဇာတ်ကောင်တွေရဲ့ ဆုံးဖြတ်ချက်က ဇာတ်လမ်းကို လှန်လိုက်တယ်။",
    "အန္တရာယ်ကြီးတစ်ခု နီးလာတာကို သူတို့ သတိမထားမိကြဘူး။",
    "ဒီအခိုက်အတန့်မှာ အားလုံးရဲ့ အခြေအနေ ပြောင်းသွားတယ်။",
    "ဆက်ပြီးတော့ သူတို့ ဘယ်လိုရင်ဆိုင်မလဲ ကြည့်ရအောင်။",
    "အဆုံးသတ်မှာ မမျှော်လင့်တဲ့ အလှည့်အပြောင်း တစ်ခု ရှိလာတယ်။",
]


def demo_timeline(duration: float, lang: str = "my") -> dict:
    """Deterministic offline timeline (RECAP_DEMO_MODE) so the pipeline can be
    exercised without an API key - used by the smoke test and the UI demo."""
    lines = []
    slot = max(6.0, min(14.0, duration / 12.0 if duration else 10.0))
    t = 0.5
    idx = 0
    while t < max(1.0, duration - 1.0):
        text = _DEMO_SENTENCES[idx % len(_DEMO_SENTENCES)]
        length = min(slot - 0.6, max(2.0, estimate_speech_seconds(text, lang)))
        lines.append({
            "start": round(t, 2),
            "end": round(min(duration, t + length), 2),
            "speaker": "Demo",
            "text": text,
        })
        t += slot
        idx += 1
    return {
        "hook_line1": "ဒီဇာတ်လမ်းကို လွတ်သွားရင် နောင်တရလိမ့်မယ်",
        "hook_line2": "အဆုံးအထိ ကြည့်ကြည့်ပါ",
        "dialogues": lines,
    }


# ── merging / coverage ───────────────────────────────────────────────────
def normalise(dialogues: Iterable[dict], duration: float) -> list[dict]:
    ordered = sorted(dialogues, key=lambda d: float(d.get("start", 0)))
    # v1.15: detect physically impossible timestamps and rescale them.
    # A model sometimes writes seconds/100 (0.47 for 47s) or 0-1
    # fractions; the old code trusted them, so every line looked 30-50x
    # too long for its 0.02s window and minute-fit amputated every line
    # mid-word ("တိုက်တန်းနစ်" -> "တိုက်တန်းနစ။"). Rescaling triggers
    # ONLY when the text physically cannot fit the stamped span (total
    # speech time > 2x the span) on a video longer than 10s, so genuine
    # fast dialogue is never touched.
    if duration > 10 and len(ordered) >= 3:
        try:
            starts = [float(d.get("start", 0)) for d in ordered]
            max_start = max(starts)
            span = max_start - min(starts)
            total_est = sum(estimate_speech_seconds(str(d.get("text", "")))
                            for d in ordered)
            factor = 0.0
            if span > 0 and total_est > span * 2:
                if 0 < max_start and max_start * 100 <= duration * 1.1:
                    factor = 100.0
                elif max_start <= 1.05:
                    factor = float(duration)
            if factor:
                ordered = [dict(d,
                                start=float(d.get("start", 0)) * factor,
                                end=float(d.get("end", 0)) * factor)
                           for d in ordered]
        except (TypeError, ValueError):
            pass
    cleaned: list[dict] = []
    for item in ordered:
        try:
            st = max(0.0, min(float(item.get("start", 0)), duration))
            en = max(0.0, min(float(item.get("end", st + 2.0)), duration))
            # v1.17: round to 0.01s — computed timestamps like
            # 147.65300000000002 are unreadable in the editor and pastes.
            st, en = round(st, 2), round(en, 2)
        except (TypeError, ValueError):
            continue
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        if not text:
            continue
        if en <= st:
            en = min(duration, st + max(1.2, estimate_speech_seconds(text)))
        cleaned.append({"start": st, "end": en,
                        "speaker": str(item.get("speaker", ""))[:40], "text": text})

    # v1.14: spread pile-ups — the model sometimes stamps several lines
    # with one timestamp (often the video duration, e.g. six dubbing
    # turns all at 180.07), which made them all speak at once at the end
    # ("နောက်နေသလို"). Lines sharing a start get sequential windows:
    # forward from their stamp when there is room, otherwise backward
    # ending at the next line's start (or the duration).
    spread: list[dict] = []
    i = 0
    while i < len(cleaned):
        j = i + 1
        while j < len(cleaned) and abs(cleaned[j]["start"] - cleaned[i]["start"]) < 0.01:
            j += 1
        run = cleaned[i:j]
        nxt = cleaned[j]["start"] if j < len(cleaned) else duration
        prev_end = spread[-1]["end"] if spread else 0.0
        if len(run) == 1 and run[0]["start"] < duration:
            spread.append(run[0])
            i = j
            continue
        needs = [max(0.8, estimate_speech_seconds(d["text"])) for d in run]
        total = sum(needs) + 0.1 * (len(run) - 1)
        lo = max(prev_end + 0.05, 0.0)
        hi = nxt if nxt > lo else lo + total
        if total > hi - lo and hi > lo:
            # not enough room: split the span evenly, keep order
            step = (hi - lo) / len(run)
            for k, d in enumerate(run):
                d["start"] = lo + k * step
                d["end"] = lo + (k + 1) * step
        else:
            if run[0]["start"] >= lo and run[0]["start"] + total <= hi:
                cursor = run[0]["start"]
            else:
                cursor = max(lo, hi - total)
            for d, need in zip(run, needs):
                d["start"] = cursor
                d["end"] = min(hi, cursor + need)
                cursor = d["end"] + 0.1
        spread.extend(run)
        i = j
    cleaned = spread

    # de-overlap: a line may never run into the next line's start, and
    # no line may run past the video end (the old version pushed ends
    # past duration for lines stamped at the end)
    merged: list[dict] = []
    for idx, line in enumerate(cleaned):
        if idx + 1 < len(cleaned):
            nxt = cleaned[idx + 1]["start"]
            if line["end"] > nxt:
                line["end"] = min(duration, max(line["start"] + 0.4, nxt))
        if line["end"] > line["start"]:
            merged.append(line)
    return merged


def split_gaps(gaps: list[dict], max_len: float = 8.0,
                min_len: float = 1.2) -> list[dict]:
    """Break long silent windows into speakable chunks.

    One line can only cover a few seconds of speech: asking for a single line
    in a 40 s hole left the rest of the hole silent ("အသံ အစအဆုံး မသွင်းတဲ့
    ပြသာနာ"). A long window is therefore split into ~8 s slots, each getting
    its own short line, so the narrator keeps talking through the whole hole.
    """
    out: list[dict] = []
    for gap in gaps:
        start = float(gap["start"])
        end = float(gap["end"])
        length = end - start
        if length <= max_len:
            if length >= min_len:
                out.append({"start": round(start, 2), "end": round(end, 2)})
            continue
        count = max(2, int(math.ceil(length / max_len)))
        step = length / count
        for index in range(count):
            sub_start = start + index * step
            sub_end = end if index == count - 1 else start + (index + 1) * step
            if sub_end - sub_start < min_len:
                continue
            out.append({"start": round(sub_start, 2), "end": round(sub_end, 2)})
    return out


def find_gaps(dialogues: list[dict], duration: float, min_gap: float,
              include_head_tail: bool = True) -> list[dict]:
    gaps: list[dict] = []
    cursor = 0.0
    for line in dialogues:
        if line["start"] - cursor >= min_gap:
            gaps.append({"start": cursor, "end": line["start"]})
        cursor = max(cursor, line["end"])
    if include_head_tail and duration - cursor >= min_gap:
        gaps.append({"start": cursor, "end": duration})
    return gaps


def _big_dubbing_gaps(quiet: list[dict], threshold: float = 8.0,
                      cap: int = 6) -> list[dict]:
    """Gaps worth re-analysing for dubbing, longest first.

    A gap longer than the threshold is usually a missed exchange, not a
    natural pause — the model hears the actual video for the region and
    returns nothing if it is genuinely silent, so the only cost of a
    lower threshold is extra analysis calls (hence the cap).
    """
    return sorted([g for g in quiet if g["end"] - g["start"] > threshold],
                  key=lambda g: g["start"] - g["end"])[:cap]


# ── sentence completeness (Burmese fragments are the #1 quality killer) ──
#: endings that CLOSE a Burmese sentence (finite verb / final particle)
_MY_COMPLETE_SUFFIXES = (
    "ပါတယ်", "တယ်", "သည်", "မယ်", "ပါ", "လေ", "ပေါ့", "ပြီ", "ဘူး", "ရှိ",
    "ဖြစ်", "သွား", "လာ", "နေ", "ထား", "ပေး", "ယူ", "ရောက်", "ရ", "စေ",
)
#: endings that leave a Burmese sentence DANGLING (subordinate clauses,
#: case particles, nominalisers) — a line ending like this is a fragment
_MY_FRAGMENT_SUFFIXES = (
    "တော့", "ဖို့", "ကို", "မှာ", "ထဲ", "ပေါ်", "နဲ့", "ပြီး", "လို့", "မို့",
    "ရင်", "တဲ့", "တာ", "ဘဲ", "ပဲ", "လွန်း", "အောင်", "ထိ", "၍", "သော",
    "သည့်", "မည့်", "စွာ", "ချင်း", "ရင်း",
    # v1.10: verb-less endings that used to pass as "complete" — a line
    # ending on one of these has no predicate, however it is punctuated:
    # direction without a verb ("ဆိုင်သို့။"), a dangling conjunction
    # ("ယောက်ျားလေးနှင့်။"), a bare noun phrase ("မျက်လုံးများ။"),
    # or a subordinate participle ("ထိုင်ကာ။", "လုပ်လျက်။").
    "သို့", "နှင့်", "များ", "ကာ", "လျက်",
    # v1.12: a bare topic with no comment ("သူဟာ။", "သူမဟာ။") and a
    # trailing "also" ("ငါလည်း။") are fragments too — missing either
    # one broke whole merge chains, because the merge only continues
    # through lines it recognises as fragments.
    "ဟာ", "လည်း",
    # v1.13: a dangling possessive ("အဘွားကြီးရဲ့။", "သူ၏။") has no
    # predicate either — dubbing turns end on these constantly.
    "ရဲ့", "၏",
    # v1.19: a dangling "that's why" ("ဒါကြောင့်။") is a subordinate
    # clause with no main clause — the owner's latest paste ends turns
    # on it, and the detector's "complete" verdict broke the merge chain.
    "ကြောင့်",
)


def normalise_my_punctuation(text: str) -> str:
    """Burmese lines must end with "။", not a Latin ".".

    The model mixes "." into Burmese output ("ခေါ်ခဲ့."), which reads
    inconsistent and breaks the "။"-based flow. A "." followed by
    whitespace or end-of-line is a sentence boundary, so it becomes
    "။"; decimals like "၄.၅" ("." followed by a digit) are untouched.
    """
    return re.sub(r"\.(?=\s|$)", "။", text or "")


def sentence_complete(text: str, lang: str = "my") -> bool:
    """Heuristic: can this line stand alone as a finished sentence?

    Gemini loves splitting ONE Burmese sentence across several timestamp
    lines, each chunk closed with "။" — grammatically they are fragments
    ("အမေရိကမှာ ဘဝသစ်စဖို့။", "သူမကတော့ ဒီအကြောင်းတွေကို လုံးဝမသိခဲ့ဘဲ။") and
    they sound terrible read aloud. Punctuation alone cannot detect them;
    the sentence-final suffix can.
    """
    t = (text or "").strip()
    if not t:
        return True
    core = t.rstrip("။.!?…\"'” ")
    if not core:
        return True
    words = core.split()
    last = words[-1] if words else core
    if lang == "my":
        for suffix in _MY_COMPLETE_SUFFIXES:
            if last.endswith(suffix):
                return True
        for suffix in _MY_FRAGMENT_SUFFIXES:
            if last.endswith(suffix):
                return False
        # v1.13: a line trailing off on "…" / "..." was cut mid-thought
        # ("အသက်နှစ်ဆယ်...။") — never a finished sentence, even
        # though the old fallback accepted a trailing "…".
        if t.rstrip("။").endswith(("...", "…")):
            return False
        # subject marker "…က" (noun + က) with a vowel sign before it.
        # v1.19: the set was missing TALL AA (U+102B) — Burmese "ဒါ"
        # (this) is spelled with TALL AA, not AA (U+102C), so "ဒါက။"
        # slipped through as "complete" and neither merge nor repair ever
        # touched it. Written as an escape so the lookalike can't recur.
        if last.endswith("က") and len(last) >= 2 and last[-2] in "ာိီုူေဲဳံး့\u102b":
            return False
        # a bare verb root ("ကလေးပုံမြင်") is only a sentence when the
        # writer actually closed it with "။" — unpunctuated, it is a scrap
        return t.endswith(("။", ".", "!", "?", "…"))
    if not t.endswith((".", "!", "?")):
        return False
    return last.lower().strip(",;:") not in {
        "the", "a", "an", "and", "but", "or", "to", "of", "in", "on", "with",
        "because", "when", "while", "who", "which", "that", "as", "by",
        "for", "from",
    }




class TimelineExtractor:
    """Full pipeline: chunked analysis + coverage sweep."""

    def __init__(self, api_key: str = "", model: str = "", target_language: str = "my",
                 mode: str = "auto", fill_mode: str = "continuous",
                 progress: ProgressFn = None, cancel: Optional[Callable[[], bool]] = None,
                 log_fn: Optional[Callable[[str], None]] = None,
                 key_ring: KeyRing | None = None, keys: Optional[list[str]] = None):
        self.api_key = api_key
        self.key_ring = key_ring
        self.keys = [k for k in (keys or []) if k]
        self.model = MODEL_ALIASES.get(model or config.settings.default_model,
                                       model or config.settings.default_model)
        self.lang = target_language
        self.mode = mode
        self.fill_mode = fill_mode
        self.progress = progress
        self.cancel = cancel
        self.log = log_fn or (lambda msg: log.info(msg))
        # Demo mode is an *explicit* opt-in (RECAP_DEMO_MODE=1). A missing API
        # key must never silently produce a fake beep-narrated video.
        self.demo = bool(config.settings.demo_mode)
        self.client: Optional[GeminiClient] = None
        self.key_switches: list[str] = []

    def _emit(self, pct: float, message: str) -> None:
        if self.progress:
            try:
                self.progress(pct, message)
            except CancelledError:
                raise
            except Exception:
                pass

    def _check_cancel(self) -> None:
        if self.cancel and self.cancel():
            raise CancelledError("အလုပ်ကို ရပ်တန့်လိုက်ပါပြီ")

    def has_key(self) -> bool:
        if self.api_key:
            return True
        if self.key_ring and self.key_ring.has_key():
            return True
        return bool(self.keys)

    # ── main entry ─────────────────────────────────────────────────────
    def run(self, video_path: str, duration: float) -> dict:
        if self.demo:
            self.log("Demo mode: synthetic timeline (API key မလိုအပ်ပါ)")
            data = demo_timeline(duration, self.lang)
            report = self._coverage(data["dialogues"], duration, 0)
            return {"hook_line1": data["hook_line1"], "hook_line2": data["hook_line2"],
                    "dialogues": data["dialogues"], "coverage": report}

        if not self.has_key():
            raise ValueError(
                "Gemini API Key မထည့်ရသေးပါ။ ⚙️ Settings tab မှ Key #1 တွင် ထည့်သွင်းပေးပါ "
                "(Key #2/#3 ထည့်ထားပါက quota ပြည့်ချိန် အလိုအလျောက် ကူးပေးပါမည်)။"
            )
        self.client = GeminiClient(
            self.api_key, key_ring=self.key_ring,
            on_switch=lambda message: (self.key_switches.append(message), self.log(message)),
        )
        sdk_version = getattr(getattr(self.client, "sdk", None), "__version__", "?")
        self.log(f"🔑 Using Gemini key #{self.client.slot} • model {self.model} "
                 f"• google-genai {sdk_version}")
        chunks = plan_chunks(duration)
        self.log(f"Analysis chunks: {len(chunks)} → {[f'{s:.0f}-{e:.0f}s' for s, e in chunks]}")
        self._emit(6, f"🧠 ဗီဒီယိုကို အပိုင်း {len(chunks)} ပိုင်းခွဲ၍ AI ဖြင့် စစ်ဆေးနေပါသည်...")

        results: list[dict] = []
        per_chunk: list[dict] = []
        errors: list[str] = []
        workers = max(1, min(config.settings.analyze_workers, len(chunks)))
        done = 0
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="analyze") as pool:
            futures = {
                pool.submit(self._analyze_chunk, video_path, start, end, duration, idx, len(chunks)): (idx, start, end)
                for idx, (start, end) in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx, start, end = futures[future]
                self._check_cancel()
                try:
                    payload, count = future.result()
                    results.extend(payload["dialogues"])
                    per_chunk.append({"index": idx + 1, "start": round(start, 1),
                                      "end": round(end, 1), "lines": count})
                    if payload.get("hook_line1") or payload.get("hook_line2"):
                        results.append({"__hook__": payload})
                except Exception as exc:
                    errors.append(f"chunk {idx + 1} ({start:.0f}-{end:.0f}s): {exc}")
                    self.log(f"⚠️ chunk {idx + 1} failed: {exc}")
                done += 1
                self._emit(8 + (done / len(chunks)) * 62,
                           f"🧠 AI စစ်ဆေးပြီး {done}/{len(chunks)} အပိုင်း "
                           f"({start:.0f}-{end:.0f}s) ✅")

        if not results and errors:
            raise RuntimeError("AI analysis failed for every chunk:\n" + "\n".join(errors[:3]))

        hooks = [r for r in results if isinstance(r, dict) and "__hook__" in r]
        dialogues = normalise([r for r in results if "__hook__" not in r], duration)
        hook1 = next((h["__hook__"].get("hook_line1", "") for h in hooks if h["__hook__"].get("hook_line1")), "")
        hook2 = next((h["__hook__"].get("hook_line2", "") for h in hooks if h["__hook__"].get("hook_line2")), "")

        # v1.16: Burmese lines end with "။", never a Latin "." — the model
        # mixes them ("ခေါ်ခဲ့."), so normalise them before merge/repair
        # judge sentence completeness on them.
        if self.lang == "my":
            for d in dialogues:
                d["text"] = normalise_my_punctuation(str(d.get("text", "")))

        # ── coverage sweep ────────────────────────────────────────────
        dialogues, filled, longest = self._coverage_sweep(dialogues, duration, video_path)
        # ── merge chopped fragments (recap: any line; dubbing: same
        #    speaker only) + repair dangling lines in place ────────────
        dialogues = self._merge_fragments(dialogues)
        dialogues = self._repair_fragments(dialogues)
        # v1.21: drop repeated passages — the model re-emits the same
        # exchange twice inside one passage with different speaker labels
        dialogues = self._dedup_repeats(dialogues)
        # ── per-minute exact fit: no line may overrun its scene/minute ──
        dialogues, fit_report = self._enforce_minute_fit(dialogues, duration)
        report = self._coverage(dialogues, duration, filled, per_chunk, longest)
        if errors:
            report["warnings"] = errors
        self.log(f"Coverage: {report['coverage_percent']}% "
                 f"({report['spoken_seconds']}s / {report['total_seconds']}s), "
                 f"{report['lines']} lines, {filled} gaps filled, longest gap {longest:.1f}s")
        return {"hook_line1": hook1, "hook_line2": hook2, "dialogues": dialogues,
                "coverage": report, "fit": fit_report,
                "repair_unfixed": getattr(self, "repair_unfixed", 0)}

    # ── one chunk ──────────────────────────────────────────────────────
    def _analyze_chunk(self, video_path: str, start: float, end: float,
                       duration: float, index: int, total: int) -> tuple[dict, int]:
        self._check_cancel()
        assert self.client is not None
        proxy = Path(config.TMP_DIR) / f"proxy_{uuid.uuid4().hex[:8]}.mp4"
        ref = None
        try:
            build_proxy(video_path, start, end, proxy)
            # inline for small proxies, Files API for big ones — build_media_part
            # makes sure the SDK never sees a bare path or a file without mime
            part, ref = self.client.build_media_part(
                proxy, label=f"အပိုင်း {index + 1}/{total}", progress=self.progress)
            prompt = _chunk_prompt(start, end, self.lang, self.mode, self.fill_mode, duration)
            raw = self.client.generate_json(self.model, [part, prompt], DIALOGUE_SCHEMA,
                                            progress=self.progress, cancel=self.cancel)
            payload = parse_dialogues(raw)
            clip_len = max(0.5, end - start)
            lines = payload["dialogues"]
            # Models occasionally answer relative to the clip even when told
            # to use absolute video time. If *every* timestamp fits inside the
            # clip then the answer was relative -> shift it by the offset.
            looks_relative = bool(lines) and max(l["start"] for l in lines) < clip_len - 0.5
            shifted = []
            for line in lines:
                if looks_relative:
                    line_start = min(duration, start + line["start"])
                    line_end = min(duration, start + line["end"])
                else:
                    line_start = min(duration, max(start, line["start"]))
                    line_end = min(duration, max(line_start + 0.5, line["end"]))
                shifted.append({"start": line_start, "end": line_end,
                                "speaker": line.get("speaker", ""), "text": line["text"]})
            payload["dialogues"] = shifted

            # ── tail pass: never let a clip end in silence ─────────────
            # Models like to "summarise" the ending, so the last line often
            # stops far before the clip does and the dub fell silent there.
            if config.settings.ai_tail_pass and self.fill_mode == "continuous" and shifted:
                last_end = max(l["end"] for l in shifted)
                missing_tail = end - last_end
                if missing_tail > max(6.0, (end - start) * 0.12) and end - start > 12.0:
                    tail_start = max(start + 0.5, last_end - 1.0)
                    self.log(f"↻ chunk {index + 1}: tail {tail_start:.0f}-{end:.0f}s "
                             f"မပါသေးပါ — ပြန်မေးနေပါသည်")
                    try:
                        extra = self._ask_region(proxy, video_path, tail_start, end, duration,
                                                 focus=True)
                        before = len(shifted)
                        shifted.extend(extra)
                        self.log(f"✓ tail pass recovered {len(shifted) - before} lines "
                                 f"({tail_start:.0f}-{end:.0f}s)")
                    except CancelledError:
                        raise
                    except Exception as exc:
                        self.log(f"⚠️ tail pass failed ({start:.0f}-{end:.0f}s): {exc}")
            payload["dialogues"] = shifted
            return payload, len(shifted)
        finally:
            if ref is not None:
                self.client.delete(ref)
            proxy.unlink(missing_ok=True)

    def _ask_region(self, proxy: Path, video_path: str, start: float, end: float,
                    duration: float, focus: bool = False) -> list[dict]:
        """Ask the model about one window (used for tails and re-analysis)."""
        assert self.client is not None
        proxy_path = proxy
        temporary = False
        if not focus:
            # a fresh proxy because the caller's clip covers a wider window
            proxy_path = Path(config.TMP_DIR) / f"proxy_{uuid.uuid4().hex[:8]}.mp4"
            build_proxy(video_path, start, end, proxy_path)
            temporary = True
        ref = None
        try:
            part, ref = self.client.build_media_part(
                proxy_path, label=f"အပိုင်း {start:.0f}-{end:.0f}s", progress=self.progress)
            boundary_hint = (
                f"\n\nIMPORTANT: only the window {start:.2f}s–{end:.2f}s is missing from the "
                "timeline. Return entries that START inside this window and keep going until "
                f"{end:.2f}s."
            )
            prompt = _chunk_prompt(start, end, self.lang, self.mode, self.fill_mode,
                                   duration, boundary_hint=boundary_hint)
            raw = self.client.generate_json(self.model, [part, prompt], DIALOGUE_SCHEMA,
                                            progress=self.progress, cancel=self.cancel)
            payload = parse_dialogues(raw)
            clip_len = max(0.5, end - start)
            lines = payload["dialogues"]
            looks_relative = bool(lines) and max(l["start"] for l in lines) < clip_len - 0.5
            out: list[dict] = []
            for line in lines:
                if looks_relative:
                    line_start = min(duration, start + line["start"])
                    line_end = min(duration, start + line["end"])
                else:
                    line_start = min(duration, max(start - 1.0, line["start"]))
                    line_end = min(duration, max(line_start + 0.5, line["end"]))
                out.append({"start": line_start, "end": line_end,
                            "speaker": line.get("speaker", ""), "text": line["text"]})
            return out
        finally:
            if ref is not None:
                self.client.delete(ref)
            if temporary:
                proxy_path.unlink(missing_ok=True)

    # ── gap filling ────────────────────────────────────────────────────
    def _min_gap(self) -> float:
        # dialogue dubbing has no "gaps" to fill — silence is the correct
        # output there, so the sweep is skipped entirely (see _coverage_sweep)
        return 2.5 if self.fill_mode == "continuous" else 1e9

    def _coverage_sweep(self, dialogues: list[dict], duration: float,
                        video_path: str) -> tuple[list[dict], int, float]:
        """Make sure narration runs from the first to the last second.

        Three passes, each one cheaper than the last:
          1. big holes → re-analyse that piece of *video* (the model simply
             skipped a scene, text alone would invent things),
          2. remaining holes → short connective lines written in batches,
          3. anything the model still refuses to fill → a deterministic
             fallback line so continuous mode never goes silent.
        """
        if self.fill_mode != "continuous":
            # 🎭 Dialogue dubbing: only real speech is dubbed — never fill
            # quiet parts with invented narration. BUT a hole usually
            # means the model skipped speech, so re-analyse big holes
            # against the actual video: if the region is genuinely silent
            # the model returns nothing and the hole stays silent,
            # correctly. v1.22: threshold 20s → 8s — the owner's scripts
            # show many 7–13s holes where characters are audibly speaking
            # (e.g. 13s, 8.7s, 7.9s gaps in one 180s passage), and 20s
            # caught none of them; shorter gaps are usually natural
            # pauses between turns. Up to 6, longest first, bounds cost.
            quiet = find_gaps(dialogues, duration, 3.0)
            longest = max((g["end"] - g["start"] for g in quiet), default=0.0)
            big = _big_dubbing_gaps(quiet)
            region_lines: list[dict] = []
            if big and self.client is not None and not self.demo:
                for gap in big:
                    self._check_cancel()
                    try:
                        lines = self._ask_region(Path(""), video_path, gap["start"],
                                                 gap["end"], duration, focus=False)
                        region_lines.extend(lines)
                        self.log(f"↻ dubbing re-analysis {gap['start']:.0f}-{gap['end']:.0f}s "
                                 f"→ {len(lines)} lines")
                    except CancelledError:
                        raise
                    except Exception as exc:
                        self.log(f"⚠️ dubbing region re-analysis failed "
                                 f"({gap['start']:.0f}s): {exc}")
            if region_lines:
                dialogues = normalise(dialogues + region_lines, duration)
                quiet = find_gaps(dialogues, duration, 3.0)
                longest = max((g["end"] - g["start"] for g in quiet), default=0.0)
            self.log(f"🎭 Dialogue dub mode: {len(dialogues)} စကားပြောလိုင်း — "
                     f"တိတ်ဆိတ်ချိန်ကို ဖြည့်စွက်ခြင်း မလုပ်ပါ (အရှည်ဆုံး {longest:.1f}s)")
            return dialogues, 0, longest

        min_gap = self._min_gap()
        gaps = find_gaps(dialogues, duration, min_gap)
        longest = max((g["end"] - g["start"] for g in gaps), default=0.0)
        if not gaps or self.demo:
            return dialogues, 0, longest

        self._emit(72, f"🧩 လွတ်နေသော အပိုင်း {len(gaps)} ခုကို ဖြည့်စွက်နေပါသည်...")

        # 1) large gaps → re-analyse the region (up to 6, longest first)
        big_gaps = sorted([g for g in gaps if (g["end"] - g["start"]) > 20.0],
                          key=lambda g: g["start"] - g["end"])[:6]
        region_lines: list[dict] = []
        for gap in big_gaps:
            self._check_cancel()
            try:
                lines = self._ask_region(Path(""), video_path, gap["start"], gap["end"],
                                         duration, focus=False)
                region_lines.extend(lines)
                self.log(f"↻ re-analysed {gap['start']:.0f}-{gap['end']:.0f}s → {len(lines)} lines")
            except CancelledError:
                raise
            except Exception as exc:
                self.log(f"⚠️ region re-analysis failed ({gap['start']:.0f}s): {exc}")
        if region_lines:
            dialogues = normalise(dialogues + region_lines, duration)

        # 2) connective narration for everything that is still empty
        filled = 0
        for attempt in range(3):
            gaps = find_gaps(dialogues, duration, min_gap)
            if not gaps:
                break
            self._emit(74 + attempt * 2,
                       f"🧩 လွတ်နေသော နေရာ {len(gaps)} ခု အတွက် စကားပြောစာ ရေးနေပါသည်...")
            added = self._fill_gaps_with_text(dialogues, gaps, duration,
                                              allow_fallback=attempt > 0, split=True)
            filled += added
            if added:
                dialogues = normalise(dialogues, duration)
            if not added:
                break

        # 3) final safety net: continuous mode must not fall silent
        remaining = find_gaps(dialogues, duration, max(min_gap, 6.0))
        if remaining and self.fill_mode == "continuous":
            filled += self._fallback_fill(dialogues, split_gaps(remaining), duration)
            dialogues = normalise(dialogues, duration)

        longest = max((g["end"] - g["start"] for g in find_gaps(dialogues, duration, 3.0)),
                      default=0.0)
        return dialogues, filled, longest

    #: neutral connectors used only when the AI cannot answer (keeps the voice
    #: track continuous instead of leaving the viewer with silence)
    _FALLBACK_LINES = (
        "ဒီအခိုက်အတန့်မှာ ဇာတ်လမ်းက ဆက်လက် ဖြစ်ပျက်နေပါတယ်။",
        "ဆက်ပြီးတော့ ဘာတွေ ဖြစ်လာမလဲ ကြည့်ရအောင်ဗျာ။",
        "ဒီနေရာမှာ အရေးကြီးတဲ့ အပြောင်းအလဲ တစ်ခု ရှိလာပါတယ်။",
        "ဇာတ်ကောင်တွေ ဆက်ပြီး ဘယ်လို ရင်ဆိုင်မလဲ ဆက်ကြည့်ရအောင်။",
        "အခုတော့ နောက်ထပ် အရေးကြီးတဲ့ အခိုက်အတန့်တစ်ခု ရောက်လာပါပြီ။",
    )

    def _fallback_fill(self, dialogues: list[dict], gaps: list[dict],
                       duration: float) -> int:
        added = 0
        gaps = split_gaps(gaps)
        for idx, gap in enumerate(gaps[:30]):
            window = gap["end"] - gap["start"]
            if window < 3.0:
                continue
            text = self._FALLBACK_LINES[idx % len(self._FALLBACK_LINES)]
            est = estimate_speech_seconds(text, self.lang)
            if est > window:
                keep = max(12, int(len(text) * (window / est)))
                text = text[:keep].rstrip(" ။၊,")
            if not text:
                continue
            dialogues.append({"start": gap["start"], "end": gap["end"],
                              "speaker": "Recap", "text": text})
            added += 1
        if added:
            self.log(f"↳ AI မဖြည့်နိုင်သော ကွက် {added} ခုကို အလိုအလျောက် ဖြည့်လိုက်ပါသည်")
        return added

    def fill_windows(self, gaps: list[dict], duration: float) -> list[dict]:
        """Public helper for the TTS repair loop (pipeline).

        Given windows that ended up silent after synthesis, ask the model for
        one short line per window and return ready-to-use dialogue entries.
        Falls back to neutral connectors when the AI cannot answer.
        """
        if not gaps:
            return []
        self._check_cancel()
        if self.client is None and not self.demo:
            return []
        out: list[dict] = []
        windows = split_gaps(gaps)
        if not self.demo and self.client is not None:
            try:
                created = self._fill_gaps_with_text(out, windows, duration,
                                                    allow_fallback=False, log_prefix=True,
                                                    split=False)
                if created:
                    self.log(f"🧩 silence repair: AI wrote {created} lines")
            except CancelledError:
                raise
            except Exception as exc:
                self.log(f"⚠️ silence repair AI call failed: {exc}")
        got = {round(d["start"], 1) for d in out}
        missing = [g for g in windows if round(g["start"], 1) not in got]
        if missing:
            before = len(out)
            self._fallback_fill(out, missing, duration)
            self.log(f"↳ silence repair fallback added {len(out) - before} lines")
        return out

    def _fill_gaps_with_text(self, dialogues: list[dict], gaps: list[dict],
                             duration: float, allow_fallback: bool = False,
                             log_prefix: bool = False, split: bool = False) -> int:
        """Write one short line per gap (batched so long videos work too)."""
        assert self.client is not None
        if split:
            gaps = split_gaps(gaps)
        added = 0
        batch_size = 20
        for start_index in range(0, min(len(gaps), 120), batch_size):
            batch = gaps[start_index:start_index + batch_size]
            payload_gaps = []
            context: list[dict] = []
            for idx, gap in enumerate(batch):
                length = gap["end"] - gap["start"]
                payload_gaps.append({
                    "index": idx,
                    "start": gap["start"],
                    "end": gap["end"],
                    "chars": max(8, int(length * (9.5 if self.lang == "my" else 13.5))),
                })
                before = [d for d in dialogues if d["end"] <= gap["start"]][-2:]
                after = [d for d in dialogues if d["start"] >= gap["end"]][:2]
                for item in before + after:
                    context.append({"start": item["start"], "text": item["text"]})
            context = context[-24:]
            prompt = _gap_fill_prompt(payload_gaps, self.lang, context)
            try:
                raw = self.client.generate_json(self.model, [prompt], GAP_FILL_SCHEMA,
                                                temperature=0.4, max_tokens=8192,
                                                progress=self.progress, cancel=self.cancel)
                data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            except CancelledError:
                raise
            except Exception as exc:
                self.log(f"⚠️ gap fill failed ({start_index + 1}-{start_index + len(batch)}): {exc}")
                if allow_fallback:
                    added += self._fallback_fill(dialogues, batch, duration)
                continue
            for item in data.get("lines", []):
                try:
                    gap = payload_gaps[int(item["index"])]
                except (KeyError, ValueError, IndexError, TypeError):
                    continue
                text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
                if not text:
                    continue
                est = estimate_speech_seconds(text, self.lang)
                if est > (gap["end"] - gap["start"]) + 1.0:
                    text = text[: max(10, int(len(text) * (gap["end"] - gap["start"]) / est))]
                dialogues.append({"start": gap["start"], "end": gap["end"], "speaker": "Recap",
                                  "text": text})
                added += 1
        # also cover the first couple of seconds if the recap opens silent
        if dialogues and not any(d["start"] < 1.0 for d in dialogues) and duration > 6:
            dialogues.append({"start": 0.2, "end": min(4.0, duration), "speaker": "Recap",
                              "text": "ဒီဇာတ်လမ်းကို အခုပဲ ကြည့်လိုက်ရအောင်။"})
            added += 1
        if not log_prefix:
            self.log(f"Coverage sweep added {added} narration lines")
        return added

    def _merge_fragments(self, dialogues: list[dict]) -> list[dict]:
        """Join chopped fragments back into complete, listenable lines.

        Sentence-by-sentence output (worst with narration-dense sources)
        leaves half-thoughts like "ဒါပေမဲ့ ကလေးပုံမြင်" as standalone lines —
        wooden to hear and impossible to voice naturally. A line that is
        not a complete sentence (see :func:`sentence_complete` — Burmese
        fragments often DO end with "။", so punctuation alone is not
        enough), or that is merely a terse caption-style scrap (see
        :func:`line_is_terse`), and is immediately followed by its
        continuation is merged with it, as long as the combined line stays
        reasonable (≤16s window, ≤50% over the naive speech estimate —
        retime slows the picture to its 0.75x floor, 1.33x, and a short
        last-frame freeze absorbs the rest). In Dubbing mode only turns
        of the SAME
        speaker merge (a narrator's sentence split across turns joins
        back up; different speakers' turns never merge, so timing and
        turn-taking stay intact).
        """
        if self.fill_mode not in ("continuous", "dialogue"):
            return dialogues
        out: list[dict] = []
        merged = 0
        for item in sorted(dialogues, key=lambda d: float(d.get("start", 0.0))):
            d = dict(item)
            if out:
                prev = out[-1]
                prev_text = str(prev.get("text", "")).strip()
                prev_window = float(prev.get("end", 0.0)) - float(prev.get("start", 0.0))
                gap = float(d.get("start", 0.0)) - float(prev.get("end", 0.0))
                prev_frag = not sentence_complete(prev_text, self.lang)
                # v1.12: when the previous line is a FRAGMENT, its "။"
                # is a false boundary — drop it so the joined line reads
                # as ONE flowing sentence ("…နေချိန်မှာ ဒါပေမဲ့…")
                # instead of three chopped sentences in one line. A
                # merely terse (grammatically closed) line keeps its "။".
                base = (prev_text.rstrip("။.") if prev_frag else prev_text)
                joined = f"{base} {str(d.get('text', '')).strip()}".strip()
                window = float(d.get("end", 0.0)) - float(prev.get("start", 0.0))
                # v1.16: in Dubbing mode only same-speaker turns merge —
                # a narrator's sentence split across turns joins back up,
                # but two speakers' turns never merge into one line.
                if self.fill_mode == "dialogue":
                    sp_prev = str(prev.get("speaker", "")).strip()
                    sp_cur = str(d.get("speaker", "")).strip()
                    same_speaker = bool(sp_prev) and sp_prev == sp_cur
                else:
                    same_speaker = True
                if (prev_text and same_speaker
                        and (prev_frag
                             or line_is_terse(prev_text, prev_window, self.lang))
                        and gap <= 1.5 and 0 < window <= 16.0
                        and estimate_speech_seconds(joined, self.lang) <= window * 1.5):
                    prev["text"] = joined
                    prev["end"] = d.get("end", prev.get("end"))
                    merged += 1
                    continue
            out.append(d)
        if merged:
            self.log(f"🧩 Fragment merge: အပိုင်းအစ {merged} ခု ကို ဝါကျ အပြည့် လိုင်းများ အဖြစ် ပြန်ပေါင်းပြီးပါပြီ")
        return out

    def _dedup_repeats(self, dialogues: list[dict]) -> list[dict]:
        """Drop long lines that repeat an earlier line almost verbatim.

        The model sometimes re-emits the same exchange twice inside one
        passage (e.g. the chicken-gift scene at ~70s and again at ~111s,
        with different speaker labels). A long line (>=20 chars) is a
        duplication if, against an earlier line within 180s, either its
        overall similarity is >=85% or the two share one contiguous
        block of >=25 chars (the refusal pair scores only 0.745 overall
        because the model dropped its prefix and reworded its ending,
        yet its whole core clause repeats verbatim). Short repeats
        (laughter, chants, name calls) are left alone, and so are
        repeats far apart. Residual risk, stated plainly: two genuinely
        different speeches sharing a >=25-char opening inside 180s can
        be mistaken for a repeat — the editor can restore such a line.
        """
        import difflib

        def _norm(text: str) -> str:
            return re.sub(r"[\s။.!?,\"“”]+", "", str(text))

        kept: list[dict] = []
        dropped = 0
        for d in dialogues:
            norm = _norm(d.get("text", ""))
            dup = False
            if len(norm) >= 20:
                for k in kept:
                    if abs(float(d.get("start", 0.0)) - float(k.get("start", 0.0))) > 180.0:
                        continue
                    matcher = difflib.SequenceMatcher(None, _norm(k.get("text", "")), norm)
                    longest = max((b.size for b in matcher.get_matching_blocks()), default=0)
                    if matcher.ratio() >= 0.85 or longest >= 25:
                        dup = True
                        break
            if dup:
                dropped += 1
            else:
                kept.append(d)
        if dropped:
            self.log(f"🧹 Repeat dedup: စာတပိုဒ်ထဲ ထပ်နေတဲ့ လိုင်း {dropped} လိုင်း "
                     f"ဖျက်ပြီးပါပြီ")
        return kept

    def _repair_fragments(self, dialogues: list[dict]) -> list[dict]:
        """AI-rewrite lines that are STILL fragments — or terse scraps.

        Merging cannot fix every fragment (a video may end on one, or the
        continuation is too far away), and it cannot fix caption-style
        one-phrase lines that are grammatically closed but empty
        ("သူဟာ လူတိုင်း။"). One batched Gemini call rewrites each into a
        complete, full, natural sentence using its neighbours as context —
        same facts, finished verb, filled out to its window. Acceptance is
        judged against the line's WINDOW budget (not its old length: the
        old 1.45×-of-old-length cap rejected exactly the expansions the
        shortest lines needed). Recap and dubbing: merging stays
        recap-only (dubbing turns must keep their own speech windows),
        but repair completes dangling dubbing lines in place. Without a
        client the lines are left as-is.
        """
        # v1.19: how many lines stayed broken, surfaced to the UI (a plain
        # count, never the texts) — the log line alone was invisible on a
        # successful analysis, so the app never showed it.
        self.repair_unfixed = 0
        if self.fill_mode not in ("continuous", "dialogue") or self.client is None:
            return dialogues
        from .tts import chars_per_second
        cps = chars_per_second(self.lang)
        lines = sorted(dialogues, key=lambda d: float(d.get("start", 0.0)))

        def _needs_repair(d: dict) -> bool:
            text = str(d.get("text", ""))
            window = max(1.0, float(d.get("end", 0.0)) - float(d.get("start", 0.0)))
            return (not sentence_complete(text, self.lang)
                    or line_is_terse(text, window, self.lang))

        bad = [i for i, d in enumerate(lines) if _needs_repair(d)]
        if not bad:
            return lines
        batch = []
        for i in bad[:24]:
            d = lines[i]
            window = max(1.0, float(d.get("end", 0.0)) - float(d.get("start", 0.0)))
            batch.append({
                "index": i,
                "max_chars": int(window * cps * 1.2),
                "before": str(lines[i - 1].get("text", "")) if i > 0 else "",
                "text": str(d.get("text", "")),
                "after": str(lines[i + 1].get("text", "")) if i + 1 < len(lines) else "",
            })
        listing = "\n".join(
            f"- index {b['index']} (max {b['max_chars']} chars): \"{b['text']}\""
            f"  [context before: \"{b['before']}\" / after: \"{b['after']}\"]"
            for b in batch)
        lang_word = "Burmese" if self.lang == "my" else "English"
        kind = "recap" if self.fill_mode == "continuous" else "dubbing"
        mode_note = (" Each line is one speaker's turn in a dubbing script: keep it inside its own "
                     "speech window and complete it only from the neighbouring turns' content — "
                     "never move words between turns." if kind == "dubbing" else "")
        prompt = f"""These {lang_word} {kind} lines are BROKEN in one of two ways — each one either stops mid-thought
(a fragment ending on a subordinate clause or particle instead of a finished verb), or it is a
terse caption-style scrap that only names a person/place/time and says nothing ("သူဟာ လူတိုင်း။").
A narrator cannot read either kind naturally, and they stand out against the full lines around them.
{mode_note}
{listing}

Rewrite EACH line into ONE complete, full, natural, flowing sentence:
1. Keep its facts, names and meaning. Use the before/after lines ONLY as context to know
   what the finished thought is — do not copy them and do not move facts between lines.
2. The sentence must END with a finished verb and final punctuation. It may borrow the
   obvious missing ending (e.g. a fragment ending "...ဘဝသစ်စဖို့" becomes
   "...ဘဝသစ် စတင်ဖို့ ထွက်ခွာခဲ့ပါတယ်" only when the context supports it).
3. For a terse scrap, EXPAND it: attach the who/where/what-happens that the context makes
   obvious so the line carries a real beat like its neighbours do. Never invent new plot
   facts, names or events that are not in the line or its context.
4. Fill the line out toward its max characters (that is its time window) without padding
   or generic filler, and keep the natural spoken style.
5. NEVER return a line unchanged. Every line in the list is broken — a line ending on
   က / တဲ့ / များ / တော့ / ကို / မှာ / ကြောင့် / ဖို့ / ဟာ / လည်း is NOT a finished
   sentence no matter how natural it looks, and returning it as-is is a failure.
   Extend it to a finished verb using the context.
6. Return STRICT JSON only: {{"lines": [{{"index": 0, "text": "..."}}]}}"""
        schema = {"type": "object", "properties": {"lines": {
            "type": "array", "items": {"type": "object", "properties": {
                "index": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["index", "text"]}}}, "required": ["lines"]}
        try:
            raw = self.client.generate_json(self.model, [prompt], schema,
                                            temperature=0.3, max_tokens=8192,
                                            progress=self.progress, cancel=self.cancel)
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except CancelledError:
            raise
        except Exception as exc:
            self.log(f"⚠️ fragment repair failed: {exc}")
            self.repair_unfixed = len(batch)
            return lines
        max_chars = {b["index"]: b["max_chars"] for b in batch}
        fixed = 0
        for item in data.get("lines", []):
            try:
                i = int(item["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= i < len(lines)) or i not in max_chars:
                continue
            text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
            old = str(lines[i].get("text", ""))
            if (text and sentence_complete(text, self.lang)
                    and len(text) <= max_chars[i]
                    and len(text) >= len(old) * 0.9):
                lines[i]["text"] = text
                fixed += 1
        self.repair_unfixed = len(batch) - fixed
        if fixed:
            self.log(f"🪡 Fragment repair: အပိုင်းအစ {fixed} လိုင်း ကို ဝါကျ အပြည့် ပြန်ရေးပြီးပါပြီ")
        elif batch:
            # v1.18: silence here used to look identical to "nothing was
            # broken" — say plainly that lines stayed broken so the next
            # diagnosis starts from evidence, not another blind rebuild.
            self.log(f"⚠️ Fragment repair: အပိုင်းအစ {len(batch)} လိုင်း ပြင်လို့ မရပါ — "
                     f"model ပြန်ချက် လက်ခံလို့ မရခြင်း (ဝါကျမပြည့်/ window ကျော်) — "
                     f"editor မှာ ပြင်နိုင်ပါတယ်")
        return lines

    def _enforce_minute_fit(self, dialogues: list[dict],
                            duration: float) -> tuple[list[dict], dict]:
        """Per-minute exact fit (owner rule v1.4: "script ကို မိနစ်အလိုက်
        ကွက်တိ ဖြစ်အောင် ရေးခိုင်းပါ").

        A line whose natural speech overruns its own scene — or a minute
        block whose lines together overrun the minute — leaves the voice
        still reading the previous scene while the picture has moved on.
        Offenders are shortened ONCE here: first an AI rewrite that keeps
        the natural spoken style and the facts, then a deterministic
        clause-boundary cut for anything still too long. The voice itself
        is never sped up (that stays forbidden), and retime only has to
        absorb the small remainder.
        """
        from .tts import chars_per_second, condense_line
        cps = chars_per_second(self.lang)
        lines = sorted(dialogues, key=lambda d: float(d.get("start", 0.0)))
        report = {"checked": len(lines), "fixed_ai": 0, "fixed_cut": 0,
                  "blocks_adjusted": 0}

        def window(d: dict) -> float:
            return max(0.4, float(d.get("end", 0.0)) - float(d.get("start", 0.0)))

        targets: dict[int, int] = {}
        for pos, d in enumerate(lines):
            # v1.15: a sub-second window is a timestamp error, not a wordy
            # line — cutting text to fit it only amputates words mid-
            # syllable, so leave such lines alone (retime/the editor
            # handle them) instead of butchering them.
            if window(d) < 0.8:
                continue
            est = estimate_speech_seconds(str(d.get("text", "")), self.lang)
            # 25% slack: retime slows the picture to meet the voice, so only
            # genuine overruns are touched — rewriting healthy lines just to
            # satisfy a tight budget was making the narration choppy.
            if est > window(d) * 1.25:
                targets[pos] = max(10, int(window(d) * cps * 0.95))
        blocks: dict[int, list[int]] = {}
        for pos, d in enumerate(lines):
            blocks.setdefault(int(float(d.get("start", 0.0)) // 60), []).append(pos)
        for block, positions in blocks.items():
            block_len = max(1.0, min(60.0, duration - block * 60.0))
            total = sum(estimate_speech_seconds(str(lines[p].get("text", "")),
                                                self.lang) for p in positions)
            if total <= block_len * 1.10:
                continue
            report["blocks_adjusted"] += 1
            by_size = sorted(positions, key=lambda p: -estimate_speech_seconds(
                str(lines[p].get("text", "")), self.lang))
            for pos in by_size:
                if total <= block_len * 1.10:
                    break
                if window(lines[pos]) < 0.8:
                    continue
                est = estimate_speech_seconds(str(lines[pos].get("text", "")), self.lang)
                goal = max(10, int(len(str(lines[pos].get("text", ""))) * 0.85))
                if pos in targets:
                    goal = min(goal, targets[pos])
                targets[pos] = goal
                total -= est * 0.15
        if not targets:
            return lines, report

        # 1) AI rewrite of just the offending lines (style-preserving)
        remaining = dict(targets)
        if self.client is not None:
            batch = [{"index": pos, "max_chars": targets[pos],
                      "text": str(lines[pos].get("text", ""))}
                     for pos in sorted(targets)]
            listing = "\n".join(
                f"- index {b['index']} (max {b['max_chars']} characters): {b['text']}"
                for b in batch)
            lang_word = "Burmese" if self.lang == "my" else "English"
            prompt = f"""Shorten these {lang_word} recap lines. Each one is currently TOO LONG to be
spoken inside its own scene, so the narrator would still be reading it after the picture changed.

{listing}

Rules:
1. Rewrite each line to at most its max characters. Keep every fact, name and the natural,
   human spoken style — just say it tighter. Never add generic filler.
2. Every rewritten line must stay a COMPLETE, flowing sentence that ends with final
   punctuation — never a fragment, never cut off mid-thought. Shorten by dropping
   secondary details and merging clauses, never by chopping a sentence in half.
3. Plain text only: no quotes, no brackets, no timestamps.
4. Return STRICT JSON only: {{"lines": [{{"index": 0, "text": "..."}}]}}"""
            schema = {"type": "object", "properties": {"lines": {
                "type": "array", "items": {"type": "object", "properties": {
                    "index": {"type": "integer"}, "text": {"type": "string"}},
                    "required": ["index", "text"]}}}, "required": ["lines"]}
            try:
                raw = self.client.generate_json(self.model, [prompt], schema,
                                                temperature=0.3, max_tokens=8192,
                                                progress=self.progress,
                                                cancel=self.cancel)
                data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
                for item in data.get("lines", []):
                    try:
                        pos = int(item["index"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
                    if pos in remaining and text and len(text) <= remaining[pos] * 1.08:
                        lines[pos]["text"] = text
                        report["fixed_ai"] += 1
                        del remaining[pos]
            except CancelledError:
                raise
            except Exception as exc:
                self.log(f"⚠️ minute-fit rewrite failed: {exc}")
        # 2) deterministic clause cut — ONLY for extreme overruns (the AI
        # rewrite keeps language natural; a hard cut is a last resort and
        # was making mildly-long lines read like chopped fragments)
        for pos, max_chars in remaining.items():
            est = estimate_speech_seconds(str(lines[pos].get("text", "")), self.lang)
            if est <= window(lines[pos]) * 1.6:
                continue
            short, changed = condense_line(str(lines[pos].get("text", "")),
                                           max_chars / max(1.0, cps), self.lang)
            if changed and short:
                lines[pos]["text"] = short
                report["fixed_cut"] += 1
        if report["fixed_ai"] or report["fixed_cut"]:
            self.log(f"✂️ Minute-fit (ကွက်တိ): လိုင်း {report['fixed_ai'] + report['fixed_cut']} ကြောင်း "
                     f"ကို သူ့အခန်း/မိနစ် အတွင်း ဝင်အောင် ချုံ့ပြီးပါပြီ "
                     f"(AI rewrite {report['fixed_ai']}, cut {report['fixed_cut']}, "
                     f"blocks {report['blocks_adjusted']})")
        return lines, report

    def _coverage(self, dialogues: list[dict], duration: float, filled: int,
                  per_chunk: list[dict] | None = None, longest: float = 0.0) -> dict:
        spoken = sum(max(0.0, d["end"] - d["start"]) for d in dialogues)
        # v1.17: surface long silences in the report, not just the log —
        # a 65s hole in a dialogue episode usually means the model skipped
        # speech (or the section is action-only); the user must SEE it to
        # judge. 20s matches the recap re-analysis threshold.
        quiet = [{"start": round(g["start"], 2), "end": round(g["end"], 2),
                  "seconds": round(g["end"] - g["start"], 2)}
                 for g in find_gaps(dialogues, duration, 20.0)]
        report = CoverageReport(
            total_seconds=duration, spoken_seconds=spoken, lines=len(dialogues),
            gaps_filled=filled, longest_gap=longest, chunks=len(per_chunk or []),
            per_chunk=per_chunk or [], quiet_gaps=quiet,
        )
        return report.as_dict()


def extract_timeline(api_key: str, video_path: str, duration: float, *,
                     language: str = "my", mode: str = "auto", fill_mode: str = "continuous",
                     model: str = None, progress: ProgressFn = None,
                     cancel: Optional[Callable[[], bool]] = None,
                     log_fn: Optional[Callable[[str], None]] = None,
                     key_ring: KeyRing | None = None,
                     keys: Optional[list[str]] = None) -> dict:
    extractor = TimelineExtractor(
        api_key=api_key, model=model or config.settings.default_model,
        target_language=language, mode=mode, fill_mode=fill_mode,
        progress=progress, cancel=cancel, log_fn=log_fn,
        key_ring=key_ring, keys=keys,
    )
    return extractor.run(video_path, duration)


def repair_timeline(video_path: str, duration: float, *, language: str = "my",
                    mode: str = "auto", fill_mode: str = "continuous",
                    model: str = "", gaps: list[dict] | None = None,
                    progress: ProgressFn = None,
                    cancel: Optional[Callable[[], bool]] = None,
                    log_fn: Optional[Callable[[str], None]] = None,
                    key_ring: KeyRing | None = None) -> list[dict]:
    """Second AI pass used by the pipeline when synthesis left silences.

    Reuses the same extractor (and therefore the same key ring / failover) but
    only asks for the windows that are actually silent in the finished audio.
    """
    extractor = TimelineExtractor(
        api_key="", model=model or config.settings.default_model,
        target_language=language, mode=mode, fill_mode=fill_mode,
        progress=progress, cancel=cancel, log_fn=log_fn, key_ring=key_ring,
    )
    if not extractor.has_key() or extractor.demo:
        return []
    extractor.client = GeminiClient(key_ring=key_ring,
                                    on_switch=lambda m: extractor.log(m))
    return extractor.fill_windows(gaps or [], duration)


def suggest_thumbnail(video_path: str, duration: float, *, api_key: str = "",
                      model: str = "", language: str = "my",
                      key_ring: KeyRing | None = None) -> dict[str, Any]:
    """Choose a truthful high-CTR thumbnail moment and short on-image copy.

    Gemini reviews the full video proxy, not a caller-supplied still. The
    returned timestamp is in the original video timeline and is used by the
    renderer to extract the actual frame.
    """
    if duration <= 0:
        raise ValueError("Video duration must be positive")
    if config.settings.demo_mode:
        headline, subheadline = (
            ("မထင်မှတ်တဲ့ အလှည့်အပြောင်း", "ဇာတ်လမ်း အကျဉ်းချုပ်")
            if language == "my" else ("An Unexpected Twist", "Watch to the end")
        )
        return {
            "timestamp": round(min(duration, max(0.0, duration * 0.38)), 2),
            "headline": headline,
            "subheadline": subheadline,
            "text_position": "left",
            "rationale": "Demo mode: synthetic suggestion; enable Gemini for a real video review.",
            "demo": True,
        }

    proxy = Path(config.TMP_DIR) / f"thumb_proxy_{uuid.uuid4().hex[:8]}.mp4"
    ref = None
    client = GeminiClient(api_key=api_key, key_ring=key_ring)
    lang = "natural spoken Burmese" if language == "my" else "natural spoken English"
    prompt = f"""You are a senior YouTube thumbnail designer and audience-retention strategist.
Review the complete video carefully and choose ONE real, sharp, emotionally strong frame that
would earn a high click-through rate on a mobile feed. Prefer a clear face, decisive action,
strong reaction, or distinctive object; avoid black frames, transitions, blur, empty scenery,
and spoilers that the video does not support. Return its timestamp in seconds from the ORIGINAL
video start (0 to {duration:.2f}s).

Design the actual thumbnail around one truthful, curiosity-driving hook. Write a headline of at
most 4 short words in {lang}; an optional subheadline of at most 3 short words. Avoid generic
phrases, fake claims, and long sentences. Choose `text_position` as the side with the most empty
space opposite the main subject (`left` or `right`). Keep mobile readability and a single clear
focal point in mind. Explain the frame choice in one short rationale sentence.
Return only the requested JSON schema."""
    try:
        build_proxy(video_path, 0.0, duration, proxy)
        part, ref = client.build_media_part(proxy, label="thumbnail designer")
        raw = client.generate_json(
            model or config.settings.default_model,
            [part, prompt], THUMBNAIL_SCHEMA,
            temperature=0.35, max_tokens=700,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned,
                             flags=re.IGNORECASE)
        result = json.loads(cleaned)
        timestamp = float(result.get("timestamp", duration * 0.4))
        if not math.isfinite(timestamp):
            timestamp = duration * 0.4
        headline = re.sub(r"\s+", " ", str(result.get("headline", ""))).strip()[:120]
        subheadline = re.sub(r"\s+", " ", str(result.get("subheadline", ""))).strip()[:120]
        if not headline:
            raise ValueError("AI thumbnail designer returned an empty headline")
        return {
            "timestamp": round(max(0.0, min(duration, timestamp)), 2),
            "headline": headline,
            "subheadline": subheadline,
            "text_position": "right" if result.get("text_position") == "right" else "left",
            "rationale": re.sub(r"\s+", " ", str(result.get("rationale", ""))).strip()[:300],
            "demo": False,
        }
    finally:
        if ref is not None:
            client.delete(ref)
        proxy.unlink(missing_ok=True)
