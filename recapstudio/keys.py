"""Gemini API key management — up to three slots, with automatic failover.

Why this exists
---------------
A long recap job can spend 10-30 minutes on Gemini calls. A single key that
hits its daily quota at minute 20 used to kill the whole job ("job stopped in
the middle", "API key ပျောက်သွားတယ်"). The studio now keeps a small key ring:

* **Slot 1-3** — each slot holds one key. Slots can be filled in the UI
  (⚙️ Settings) or with ``GEMINI_API_KEY`` / ``GEMINI_API_KEY_2`` /
  ``GEMINI_API_KEY_3`` in the server ``.env``. Server-provided keys are marked
  read-only so a browser cannot overwrite the deployment's credentials.
* **Active slot** — the key a new job starts with; the user can switch it with
  one click (also while a job is running: the next AI call picks it up).
* **Failover** — when a key answers 429 / quota / permission error it is put on
  cooldown and the next healthy slot is used, so a job keeps going. Every
  switch is written to the job log and surfaced in the UI.

The ring is stored in ``DATA_DIR/.recap_config.json`` (next to the old single
key) and is backwards compatible: a saved ``gemini_api_key`` is migrated into
slot 1 on first read.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from . import config
from .util import atomic_write_json, get_logger, read_json

log = get_logger("recap.keys")


def mask(key: str, keep: int = 4) -> str:
    """``AIzaSyABCDEF…IJKL`` → safe to render in the UI."""
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) <= 12:
        return "•" * len(key)
    return f"{key[:6]}••••{key[-keep:]}"


def looks_like_key(key: str) -> bool:
    """Loose sanity check (Google keys are long and start with ``AIza``)."""
    key = (key or "").strip()
    if len(key) < 20 or any(ch.isspace() for ch in key):
        return False
    return True


@dataclass
class KeySlot:
    index: int                 # 1-based slot number (what the UI shows)
    key: str = ""
    source: str = "empty"      # env | saved | empty
    cooldown_until: float = 0.0
    last_error: str = ""

    @property
    def present(self) -> bool:
        return bool(self.key)

    @property
    def read_only(self) -> bool:
        """Env-provided keys cannot be edited from the browser."""
        return self.present and self.source == "env"

    def healthy(self) -> bool:
        return self.present and time.time() >= self.cooldown_until

    def to_public(self) -> dict:
        return {
            "slot": self.index,
            "set": self.present,
            "masked": mask(self.key),
            "source": self.source,
            "read_only": self.read_only,
            "cooldown_seconds": max(0, int(self.cooldown_until - time.time())) if self.present else 0,
            "last_error": self.last_error[:160],
        }


class KeyRing:
    """Thread-safe ring of API keys + the currently active slot."""

    def __init__(self, config_file=config.CONFIG_FILE, size: int | None = None):
        self.config_file = config_file
        self.size = max(1, min(100, size or config.settings.max_api_keys))
        self._lock = threading.RLock()
        self._active = 0
        self._cooldowns: dict[int, tuple[float, str]] = {}
        self._load()

    # ── persistence ────────────────────────────────────────────────────
    def _env_keys(self) -> list[str]:
        raw = [config.settings.gemini_api_key, config.settings.gemini_api_key_2,
               config.settings.gemini_api_key_3]
        return [str(k or "").strip() for k in raw][: self.size]

    def _read_config(self) -> dict:
        data = read_json(self.config_file, {}) or {}
        if not isinstance(data, dict):
            data = {}
        # migrate the single-key format used before v4.1
        legacy = str(data.get("gemini_api_key") or "").strip()
        stored = data.get("gemini_api_keys")
        if not isinstance(stored, list):
            stored = []
        stored = [str(k or "").strip() for k in stored]
        if legacy and legacy not in stored:
            stored.insert(0, legacy)
        data["gemini_api_keys"] = stored[: self.size]
        return data

    def _load(self) -> None:
        with self._lock:
            data = self._read_config()
            stored = list(data.get("gemini_api_keys") or [])
            env_keys = self._env_keys()
            self._slots: list[KeySlot] = []
            for idx in range(self.size):
                env_key = env_keys[idx] if idx < len(env_keys) else ""
                saved_key = stored[idx] if idx < len(stored) else ""
                if env_key:
                    self._slots.append(KeySlot(index=idx + 1, key=env_key, source="env"))
                elif saved_key:
                    self._slots.append(KeySlot(index=idx + 1, key=saved_key, source="saved"))
                else:
                    self._slots.append(KeySlot(index=idx + 1, source="empty"))
            active = data.get("active_key_index", 0)
            try:
                self._active = max(0, min(self.size - 1, int(active)))
            except (TypeError, ValueError):
                self._active = 0

    def _persist(self) -> None:
        data = read_json(self.config_file, {}) or {}
        if not isinstance(data, dict):
            data = {}
        data["gemini_api_keys"] = [slot.key for slot in self._slots]
        data["active_key_index"] = self._active
        # keep the legacy field in sync so older builds still see a key
        data["gemini_api_key"] = self.active_key()[0] or ""
        atomic_write_json(self.config_file, data)

    def reload(self) -> None:
        with self._lock:
            self._load()

    # ── queries ────────────────────────────────────────────────────────
    @property
    def slots(self) -> list[KeySlot]:
        with self._lock:
            return list(self._slots)

    @property
    def active_index(self) -> int:
        with self._lock:
            return self._active

    def active_key(self) -> tuple[str, int]:
        """``(key, slot_number)`` of the selected slot (slot 0 when empty)."""
        with self._lock:
            slot = self._slots[self._active]
            if slot.present:
                return slot.key, slot.index
            # selected slot empty → first healthy key so the UI never dead-ends
            for candidate in self._slots:
                if candidate.present:
                    return candidate.key, candidate.index
            return "", 0

    def all_keys(self) -> list[str]:
        """Every usable key, active first (used for automatic failover)."""
        with self._lock:
            ordered: list[KeySlot] = [self._slots[self._active]]
            ordered += [s for s in self._slots if s.index != self._slots[self._active].index]
            healthy = [s for s in ordered if s.healthy()]
            cooling = [s for s in ordered if s.present and not s.healthy()]
            return [s.key for s in (healthy + cooling) if s.present]

    def has_key(self) -> bool:
        return any(slot.present for slot in self.slots)

    # ── mutation ───────────────────────────────────────────────────────
    def update(self, entries: list[dict]) -> dict:
        """Apply ``[{slot: 1, key: "AIza..."}]`` (empty key clears a slot)."""
        with self._lock:
            for item in entries or []:
                try:
                    index = int(item.get("slot", 0)) - 1
                except (TypeError, ValueError):
                    continue
                if index < 0 or index >= self.size:
                    continue
                slot = self._slots[index]
                if slot.read_only:
                    continue  # .env owns this key
                if "key" in item:
                    value = str(item.get("key") or "").strip()
                    slot.key = value
                    slot.source = "saved" if value else "empty"
                    if value:
                        self.clear_cooldown(index + 1)
            self._persist()
            return self.describe()

    def clear_cooldown(self, slot_number: int) -> None:
        self._cooldowns.pop(slot_number, None)
        with self._lock:
            for slot in self._slots:
                if slot.index == slot_number:
                    slot.cooldown_until = 0.0
                    slot.last_error = ""

    def activate(self, slot_number: int) -> dict:
        with self._lock:
            index = max(0, min(self.size - 1, int(slot_number) - 1))
            if not self._slots[index].present:
                raise ValueError(f"Key #{index + 1} တွင် key မထည့်ရသေးပါ")
            self._active = index
            self._persist()
            log.info("active Gemini key switched to slot %d", index + 1)
            return self.describe()

    def rotate_after_failure(self, slot_number: int, error: str,
                             message_fn=None) -> Optional[str]:
        """Put ``slot_number`` on cooldown and return the next usable key."""
        if not config.settings.api_key_failover:
            return None
        with self._lock:
            cooldown = max(30, int(config.settings.api_key_cooldown))
            for slot in self._slots:
                if slot.index == slot_number:
                    slot.cooldown_until = time.time() + cooldown
                    slot.last_error = str(error)[:200]
                    log.warning("key slot %d failed (%s) - cooling down %ds",
                                slot_number, str(error)[:120], cooldown)
            candidates = [s for s in self._slots
                          if s.present and s.index != slot_number and s.healthy()]
            if not candidates:
                return None
            nxt = candidates[0]
            self._active = nxt.index - 1
            self._persist()
            if message_fn:
                try:
                    message_fn(f"🔑 Key #{slot_number} အလုပ်မလုပ်တော့ပါ — Key #{nxt.index} သို့ "
                               f"အလိုအလျောက် ပြောင်းလိုက်ပါသည်")
                except Exception:
                    pass
            return nxt.key

    # ── UI payload ─────────────────────────────────────────────────────
    def describe(self) -> dict:
        with self._lock:
            active = self._active + 1 if self._slots[self._active].present else 0
            return {
                "keys": [slot.to_public() for slot in self._slots],
                "active_slot": active,
                "max_keys": self.size,
                "failover": bool(config.settings.api_key_failover),
                "any_key": self.has_key(),
                "masked": mask(self.active_key()[0]),
            }


#: module level singleton (the app is a single process / single data dir)
key_ring = KeyRing()


def validate_key(key: str, model: str = "") -> tuple[bool, str]:
    """Check a key against the Gemini API (used by the "Test" button).

    Returns ``(ok, message)``. Never raises: the UI shows the message verbatim,
    which is what turns "key error" into an actionable hint (invalid key vs
    project quota vs network blocked).
    """
    key = (key or "").strip()
    if not key:
        return False, "Key အလွတ် ဖြစ်နေပါသည်"
    if not looks_like_key(key):
        return False, "Key ပုံစံ မမှန်ပါ (AIza… ဖြင့် စသည့် key အရှည် ၃၉ လုံးခန့် ဖြစ်ရပါမည်)"
    try:
        from google import genai  # type: ignore
    except ImportError:
        return False, "google-genai package မထည့်သွင်းထားပါ (pip install google-genai)"
    try:
        client = genai.Client(api_key=key)
        names = []
        for item in client.models.list():
            names.append(getattr(item, "name", "") or "")
            if len(names) >= 3:
                break
        if names:
            return True, f"✅ Key အလုပ်လုပ်ပါသည် (model {len(names)}+ ရရှိပါသည်)"
        return True, "✅ Key ချိတ်ဆက်မှု အောင်မြင်ပါသည်"
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        low = text.lower()
        if "api key not valid" in low or "api_key_invalid" in low or "401" in low or "invalid" in low:
            return False, f"⛔ Key မမှန်ကန်ပါ — {text[:180]}"
        if "permission" in low or "403" in low:
            return False, f"⛔ Key တွင် ခွင့်ပြုချက် မရှိပါ (project/billing စစ်ပါ) — {text[:160]}"
        if "quota" in low or "429" in low or "resource_exhausted" in low:
            return False, f"⚠️ Key quota ပြည့်နေပါသည် — {text[:160]}"
        return False, f"⛔ စစ်ဆေး၍ မရပါ — {text[:200]}"



# ── v5 Free Edition: unlimited in-memory key pool (Streamlit) ─────────────
class UnlimitedKeyPool:
    """Duck-type replacement for KeyRing with NO slot limit.

    Built for the Streamlit edition, where keys arrive from the UI / Secrets
    at runtime instead of a server .env file. Keys are kept in memory only
    (never written to disk), selected least-used-first, and a key that hits
    quota/429/403 is rested for ``cooldown_seconds`` while the next healthy
    key takes over — the interface mirrors KeyRing so GeminiClient can use
    it unchanged (has_key / active_key / active_index / rotate_after_failure).
    """

    def __init__(self, keys: list[str], cooldown_seconds: int = 900):
        seen: set[str] = set()
        self._keys: list[str] = []
        for k in keys or []:
            k = str(k or "").strip()
            if k and k not in seen:
                seen.add(k)
                self._keys.append(k)
        self.cooldown_seconds = max(30, int(cooldown_seconds))
        self._lock = threading.RLock()
        self._active = 0
        self._uses: dict[int, int] = {i: 0 for i in range(len(self._keys))}
        self._cooldown_until: dict[int, float] = {}
        self._last_error: dict[int, str] = {}

    def __len__(self) -> int:
        return len(self._keys)

    # ── queries ──────────────────────────────────────────────────────
    def has_key(self) -> bool:
        return bool(self._keys)

    @property
    def active_index(self) -> int:
        with self._lock:
            return self._active

    def _healthy(self, idx: int) -> bool:
        return time.time() >= self._cooldown_until.get(idx, 0.0)

    def _pick_least_used(self, exclude: int | None = None) -> int | None:
        candidates = [i for i in range(len(self._keys))
                      if i != exclude and self._healthy(i)]
        if not candidates:
            return None
        return min(candidates, key=lambda i: self._uses.get(i, 0))

    def active_key(self) -> tuple[str, int]:
        """(key, 1-based slot) of the least-used healthy key."""
        with self._lock:
            if not self._keys:
                return "", 0
            if not self._healthy(self._active):
                nxt = self._pick_least_used()
                if nxt is not None:
                    self._active = nxt
            idx = self._active
            self._uses[idx] = self._uses.get(idx, 0) + 1
            return self._keys[idx], idx + 1

    def all_keys(self) -> list[str]:
        with self._lock:
            healthy = [self._keys[i] for i in range(len(self._keys)) if self._healthy(i)]
            cooling = [self._keys[i] for i in range(len(self._keys)) if not self._healthy(i)]
            return healthy + cooling

    # ── failover ─────────────────────────────────────────────────────
    def rotate_after_failure(self, slot_number: int, error: str,
                             message_fn=None) -> Optional[str]:
        with self._lock:
            failed = int(slot_number) - 1
            if 0 <= failed < len(self._keys):
                self._cooldown_until[failed] = time.time() + self.cooldown_seconds
                self._last_error[failed] = str(error)[:200]
            nxt = self._pick_least_used(exclude=failed)
            if nxt is None:
                return None
            self._active = nxt
            if message_fn:
                try:
                    message_fn(f"🔑 Key #{slot_number} quota/error ထိသဖြင့် နားထားပြီး "
                               f"Key #{nxt + 1} သို့ အလိုအလျောက် ပြောင်းလိုက်ပါသည်")
                except Exception:
                    pass
            return self._keys[nxt]

    # ── UI payload ───────────────────────────────────────────────────
    def describe(self) -> list[dict]:
        with self._lock:
            now = time.time()
            out = []
            for i, k in enumerate(self._keys):
                out.append({
                    "slot": i + 1,
                    "masked": mask(k),
                    "uses": self._uses.get(i, 0),
                    "active": i == self._active,
                    "cooldown_seconds": max(0, int(self._cooldown_until.get(i, 0) - now)),
                    "last_error": self._last_error.get(i, "")[:120],
                })
            return out
