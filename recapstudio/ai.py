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
- Make each entry one concise thought with a start/end window that can be spoken naturally."""
    else:
        task_rules = f"""DIALOGUE DUB TIMELINE RULES
- Return an entry ONLY for real speech audible in the source audio. A quiet interval is correct;
  never add a line to satisfy clip coverage.
- Timestamp each utterance from the audio onset to audio offset using absolute full-video seconds.
  Use separate entries for separate speaker turns or clear pauses; do not merge distant speech.
- Translate the audible meaning faithfully into {lang_instruction}. Keep names, intent and emotion;
  do not omit important information or add plot explanation.
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
4. STYLE: short, clear, active spoken language. Recap mode should feel cinematic, engaging and
   professionally told. Dubbing mode should sound like a skilled actor speaking the translated
   source line, with matching emotion and no narrator voice. Avoid filler and repeated phrases.
5. Do not include timecodes, camera directions, stage directions, or speaker labels inside `text`.
   Never invent names, places, numbers, dialogue, or events. Return STRICT JSON only.
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
    cleaned: list[dict] = []
    for item in sorted(dialogues, key=lambda d: float(d.get("start", 0))):
        try:
            st = max(0.0, min(float(item.get("start", 0)), duration))
            en = max(0.0, min(float(item.get("end", st + 2.0)), duration))
        except (TypeError, ValueError):
            continue
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        if not text:
            continue
        if en <= st:
            en = min(duration, st + max(1.2, estimate_speech_seconds(text)))
        cleaned.append({"start": st, "end": en,
                        "speaker": str(item.get("speaker", ""))[:40], "text": text})

    # de-overlap: a line may never run into the next line's start
    merged: list[dict] = []
    for idx, line in enumerate(cleaned):
        if idx + 1 < len(cleaned):
            nxt = cleaned[idx + 1]["start"]
            if line["end"] > nxt:
                line["end"] = max(line["start"] + 0.4, nxt)
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

        # ── coverage sweep ────────────────────────────────────────────
        dialogues, filled, longest = self._coverage_sweep(dialogues, duration, video_path)
        report = self._coverage(dialogues, duration, filled, per_chunk, longest)
        if errors:
            report["warnings"] = errors
        self.log(f"Coverage: {report['coverage_percent']}% "
                 f"({report['spoken_seconds']}s / {report['total_seconds']}s), "
                 f"{report['lines']} lines, {filled} gaps filled, longest gap {longest:.1f}s")
        return {"hook_line1": hook1, "hook_line2": hook2, "dialogues": dialogues,
                "coverage": report}

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
            # 🎭 Dialogue dubbing: only real speech is dubbed. Filling the
            # quiet parts with invented narration is exactly what this mode
            # must NOT do, so report the silence and stop.
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

    def _coverage(self, dialogues: list[dict], duration: float, filled: int,
                  per_chunk: list[dict] | None = None, longest: float = 0.0) -> dict:
        spoken = sum(max(0.0, d["end"] - d["start"]) for d in dialogues)
        report = CoverageReport(
            total_seconds=duration, spoken_seconds=spoken, lines=len(dialogues),
            gaps_filled=filled, longest_gap=longest, chunks=len(per_chunk or []),
            per_chunk=per_chunk or [],
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
