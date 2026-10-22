"""Accounts, sessions and multi-user policy (Phase 2 — #12).

What this gives the deployment
------------------------------
* **Real accounts** — username + PBKDF2 password hash, roles (``admin`` /
  ``user``), enable/disable, forced password change.
* **Server-side sessions** — the browser only ever holds a random
  ``recap_session`` cookie; the DB stores its SHA-256 hash, an expiry, an
  idle timeout and a per-session CSRF token. Logging out (or disabling an
  account) revokes it immediately on the server.
* **Brute-force protection** — failed logins are recorded per username *and*
  per IP; after ``RECAP_LOGIN_MAX_ATTEMPTS`` inside the window the account is
  locked for ``RECAP_LOGIN_LOCKOUT_MINUTES`` with the same generic error
  message (no username enumeration).
* **Audit trail** — logins, job creation/deletion, key changes and every
  admin action land in ``audit_log``.
* **Per-user Gemini keys** — stored encrypted (see :mod:`recapstudio.crypto`)
  so one user's quota/keys are never visible to another.
* **Quotas** — per-user concurrent jobs, daily job count and disk budget.

Backwards compatibility is deliberate: with no accounts in the DB the studio
behaves exactly like v4.1 (open, or the single shared ``RECAP_ACCESS_PASSWORD``
gate), so an existing AWS box keeps working the moment it is updated and only
switches to accounts when the admin creates one.
"""
from __future__ import annotations

import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from . import config, db
from .crypto import (constant_time_equals, decrypt_secret, encrypt_secret, hash_password,
                     hash_token, needs_rehash, verify_password)
from .util import get_logger

log = get_logger("recap.auth")

USERNAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{2,31}$")
ROLES = ("admin", "user")

#: passwords that are rejected regardless of length
COMMON_PASSWORDS = {
    "password", "password1", "password123", "12345678", "123456789", "1234567890",
    "qwerty123", "admin123", "letmein123", "welcome123", "recapstudio", "myanmar123",
    "changeme", "changeme123", "iloveyou123", "adminadmin", "administrator",
}

GENERIC_LOGIN_ERROR = ("အသုံးပြုသူအမည် သို့မဟုတ် စကားဝှက် မမှန်ကန်ပါ "
                       "(Username or password is incorrect)")


# ── principal ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Principal:
    """Who is making the request (also used in 'legacy'/'open' modes)."""

    user_id: str = "public"
    username: str = "public"
    role: str = "admin"
    mode: str = "open"            # users | legacy | open
    session_token: str = ""
    via_cookie: bool = False
    must_change_password: bool = False

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def scope(self) -> str:
        """Filesystem/ownership scope — shared unless real accounts are on."""
        return self.user_id if self.mode == "users" else "shared"

    def to_public(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "mode": self.mode,
            "is_admin": self.is_admin,
            "must_change_password": self.must_change_password,
        }


ANONYMOUS = Principal(user_id="", username="", role="", mode="users")


# ── mode ─────────────────────────────────────────────────────────────────
def user_count() -> int:
    db.init()
    row = db.query_one("SELECT COUNT(*) AS n FROM users WHERE status = 'active'")
    return int(row["n"]) if row else 0


def auth_mode() -> str:
    """``users`` (accounts), ``legacy`` (shared key) or ``open`` (no gate)."""
    configured = (config.settings.auth_mode or "auto").strip().lower()
    if configured in {"users", "legacy", "open"}:
        return configured
    if user_count() > 0:
        return "users"
    if config.settings.access_password:
        return "legacy"
    return "open"


# ── password policy ──────────────────────────────────────────────────────
def check_password_policy(password: str, username: str = "") -> Optional[str]:
    pw = password or ""
    minimum = max(8, int(config.settings.password_min_length))
    if len(pw) < minimum:
        return f"စကားဝှက်သည် အနည်းဆုံး {minimum} လုံး ရှိရပါမည်"
    if pw.lower() in COMMON_PASSWORDS:
        return "ဤစကားဝှက်သည် အလွန်ရိုးရှင်းပါသည် — တခြားတစ်ခု ရွေးပါ"
    if username and username.lower() in pw.lower():
        return "စကားဝှက်ထဲတွင် အသုံးပြုသူအမည် မပါရပါ"
    classes = sum([
        bool(re.search(r"[a-z]", pw)), bool(re.search(r"[A-Z]", pw)),
        bool(re.search(r"[0-9]", pw)), bool(re.search(r"[^a-zA-Z0-9]", pw)),
    ])
    if classes < 2:
        return "စကားဝှက်တွင် စာလုံးနှင့် ဂဏန်း (သို့) သင်္ကေတ ရောထားရပါမည်"
    return None


def check_username(username: str) -> Optional[str]:
    if not USERNAME_RE.match(username or ""):
        return ("အသုံးပြုသူအမည်သည် 3-32 လုံး၊ a-z / 0-9 / . _ - သာ ပါရပါမည် "
                "(အစတွင် စာလုံး သို့ ဂဏန်း)")
    return None


# ── users CRUD ───────────────────────────────────────────────────────────
def _row_to_user(row) -> dict:
    if row is None:
        return {}
    data = dict(row)
    data.pop("password_hash", None)
    data["is_admin"] = data.get("role") == "admin"
    return data


def get_user(user_id: str) -> dict:
    db.init()
    return _row_to_user(db.query_one("SELECT * FROM users WHERE id = ?", (user_id,)))


def get_user_by_name(username: str) -> dict:
    db.init()
    return _row_to_user(db.query_one("SELECT * FROM users WHERE username_lower = ?",
                                     ((username or "").strip().lower(),)))


def _raw_user_by_name(username: str):
    db.init()
    return db.query_one("SELECT * FROM users WHERE username_lower = ?",
                        ((username or "").strip().lower(),))


def list_users(limit: int = 200) -> list[dict]:
    db.init()
    rows = db.query("SELECT * FROM users ORDER BY created_at ASC LIMIT ?", (int(limit),))
    return [_row_to_user(r) for r in rows]


def create_user(username: str, password: str, *, role: str = "user",
                display_name: str = "", must_change_password: bool = False,
                quota_bytes: int = 0, max_concurrent_jobs: int = 0,
                daily_job_limit: int = 0, skip_policy: bool = False) -> dict:
    db.init()
    username = (username or "").strip()
    error = check_username(username)
    if error:
        raise ValueError(error)
    if role not in ROLES:
        raise ValueError(f"role သည် {' / '.join(ROLES)} ထဲမှ တစ်ခု ဖြစ်ရပါမည်")
    if not skip_policy:
        error = check_password_policy(password, username)
        if error:
            raise ValueError(error)
    if get_user_by_name(username):
        raise ValueError(f"'{username}' အမည်ဖြင့် အကောင့် ရှိနှင့်ပြီးသား ဖြစ်ပါသည်")
    now = time.time()
    user_id = f"u_{uuid.uuid4().hex[:12]}"
    db.execute(
        """INSERT INTO users (id, username, username_lower, display_name, password_hash,
                              role, status, created_at, updated_at, password_changed_at,
                              must_change_password, quota_bytes, max_concurrent_jobs,
                              daily_job_limit)
           VALUES (?,?,?,?,?,?,'active',?,?,?,?,?,?,?)""",
        (user_id, username, username.lower(), display_name or username,
         hash_password(password), role, now, now, now,
         1 if must_change_password else 0, int(quota_bytes), int(max_concurrent_jobs),
         int(daily_job_limit)),
    )
    log.info("user created: %s (role=%s)", username, role)
    return get_user(user_id)


def set_password(user_id: str, password: str, *, must_change: bool = False,
                 skip_policy: bool = False) -> None:
    db.init()
    user = get_user(user_id)
    if not user:
        raise ValueError("အကောင့် ရှာမတွေ့ပါ")
    if not skip_policy:
        error = check_password_policy(password, user.get("username", ""))
        if error:
            raise ValueError(error)
    now = time.time()
    db.execute("UPDATE users SET password_hash = ?, password_changed_at = ?, updated_at = ?, "
               "must_change_password = ? WHERE id = ?",
               (hash_password(password), now, now, 1 if must_change else 0, user_id))
    revoke_user_sessions(user_id)      # a password change logs every device out


def set_status(user_id: str, status: str) -> None:
    if status not in {"active", "disabled"}:
        raise ValueError("status သည် active / disabled ဖြစ်ရပါမည်")
    db.init()
    db.execute("UPDATE users SET status = ?, updated_at = ? WHERE id = ?",
               (status, time.time(), user_id))
    if status == "disabled":
        revoke_user_sessions(user_id)


def set_role(user_id: str, role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"role သည် {' / '.join(ROLES)} ဖြစ်ရပါမည်")
    db.init()
    if role != "admin" and _is_last_admin(user_id):
        raise ValueError("နောက်ဆုံး admin ကို သာမန် user အဖြစ် မပြောင်းနိုင်ပါ")
    db.execute("UPDATE users SET role = ?, updated_at = ? WHERE id = ?",
               (role, time.time(), user_id))


def set_limits(user_id: str, *, quota_bytes: Optional[int] = None,
               max_concurrent_jobs: Optional[int] = None,
               daily_job_limit: Optional[int] = None) -> None:
    db.init()
    sets, params = [], []
    if quota_bytes is not None:
        sets.append("quota_bytes = ?")
        params.append(int(quota_bytes))
    if max_concurrent_jobs is not None:
        sets.append("max_concurrent_jobs = ?")
        params.append(int(max_concurrent_jobs))
    if daily_job_limit is not None:
        sets.append("daily_job_limit = ?")
        params.append(int(daily_job_limit))
    if not sets:
        return
    sets.append("updated_at = ?")
    params.extend([time.time(), user_id])
    db.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", params)  # noqa: S608


def _is_last_admin(user_id: str) -> bool:
    row = db.query_one("SELECT COUNT(*) AS n FROM users "
                       "WHERE role = 'admin' AND status = 'active' AND id <> ?", (user_id,))
    return int(row["n"] or 0) == 0 if row else True


def delete_user(user_id: str) -> None:
    db.init()
    if _is_last_admin(user_id):
        raise ValueError("နောက်ဆုံး admin အကောင့်ကို မဖျက်နိုင်ပါ")
    # Preference rows intentionally support the non-account `shared` scope,
    # so clean them explicitly rather than relying on a foreign-key cascade.
    db.execute("DELETE FROM user_preferences WHERE scope = ?", (user_id,))
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))


# ── login throttling ─────────────────────────────────────────────────────
def _record_attempt(username: str, ip: str, ok: bool) -> None:
    db.execute("INSERT INTO login_attempts (username, ip, ok, created_at) VALUES (?,?,?,?)",
               ((username or "").lower(), ip or "", 1 if ok else 0, time.time()))
    # keep the table small
    db.execute("DELETE FROM login_attempts WHERE created_at < ?", (time.time() - 7 * 86400,))


def failed_attempts(username: str, ip: str) -> int:
    window = time.time() - max(60, config.settings.login_window_minutes * 60)
    row = db.query_one(
        "SELECT COUNT(*) AS n FROM login_attempts "
        "WHERE ok = 0 AND created_at > ? AND (username = ? OR ip = ?)",
        (window, (username or "").lower(), ip or "\x00"))
    return int(row["n"]) if row else 0


def lockout_remaining(username: str, ip: str) -> float:
    """Seconds the caller must wait, 0 when not locked out."""
    limit = max(3, int(config.settings.login_max_attempts))
    if failed_attempts(username, ip) < limit:
        return 0.0
    row = db.query_one(
        "SELECT MAX(created_at) AS t FROM login_attempts "
        "WHERE ok = 0 AND (username = ? OR ip = ?)",
        ((username or "").lower(), ip or "\x00"))
    last = float(row["t"] or 0) if row else 0.0
    unlock = last + max(60, config.settings.login_lockout_minutes * 60)
    return max(0.0, unlock - time.time())


def clear_attempts(username: str, ip: str) -> None:
    db.execute("DELETE FROM login_attempts WHERE username = ? OR ip = ?",
               ((username or "").lower(), ip or "\x00"))


# ── sessions ─────────────────────────────────────────────────────────────
def create_session(user_id: str, ip: str = "", user_agent: str = "") -> dict:
    db.init()
    token = secrets.token_urlsafe(40)
    csrf = secrets.token_urlsafe(24)
    now = time.time()
    expires = now + max(1.0, float(config.settings.session_ttl_hours)) * 3600
    db.execute(
        """INSERT INTO sessions (token_hash, user_id, csrf_token, created_at, expires_at,
                                 last_seen_at, ip, user_agent)
           VALUES (?,?,?,?,?,?,?,?)""",
        (hash_token(token), user_id, csrf, now, expires, now, ip or "",
         (user_agent or "")[:200]),
    )
    _prune_sessions()
    return {"token": token, "csrf_token": csrf, "expires_at": expires}


def _prune_sessions() -> None:
    db.execute("DELETE FROM sessions WHERE expires_at < ? OR revoked_at > 0",
               (time.time(),))


def resolve_token(token: str) -> tuple[dict, dict]:
    """``(user, session)`` for a valid token, otherwise ``({}, {})``."""
    if not token:
        return {}, {}
    db.init()
    row = db.query_one(
        """SELECT s.token_hash, s.user_id, s.csrf_token, s.expires_at, s.last_seen_at,
                  u.*
           FROM sessions s JOIN users u ON u.id = s.user_id
           WHERE s.token_hash = ? AND s.revoked_at = 0""",
        (hash_token(token),))
    if row is None:
        return {}, {}
    now = time.time()
    if float(row["expires_at"]) < now:
        db.execute("DELETE FROM sessions WHERE token_hash = ?", (row["token_hash"],))
        return {}, {}
    idle_limit = max(0.0, float(config.settings.session_idle_hours)) * 3600
    if idle_limit and (now - float(row["last_seen_at"])) > idle_limit:
        db.execute("DELETE FROM sessions WHERE token_hash = ?", (row["token_hash"],))
        return {}, {}
    if (row["status"] or "active") != "active":
        return {}, {}
    # sliding refresh, but only once a minute (avoids a write per poll)
    if now - float(row["last_seen_at"]) > 60:
        db.execute("UPDATE sessions SET last_seen_at = ? WHERE token_hash = ?",
                   (now, row["token_hash"]))
    session = {"token_hash": row["token_hash"], "csrf_token": row["csrf_token"],
               "expires_at": row["expires_at"], "user_id": row["user_id"]}
    return _row_to_user(row), session


def revoke_token(token: str) -> None:
    if not token:
        return
    db.init()
    db.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_token(token),))


def revoke_user_sessions(user_id: str) -> int:
    db.init()
    cur = db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    return int(cur.rowcount or 0)


def list_sessions(user_id: str) -> list[dict]:
    db.init()
    rows = db.query("SELECT created_at, expires_at, last_seen_at, ip, user_agent "
                    "FROM sessions WHERE user_id = ? ORDER BY last_seen_at DESC", (user_id,))
    return [dict(r) for r in rows]


# ── authentication ───────────────────────────────────────────────────────
def authenticate(username: str, password: str, ip: str = "") -> tuple[dict, str]:
    """``(user, "")`` on success, ``({}, message)`` otherwise."""
    db.init()
    username = (username or "").strip()
    wait = lockout_remaining(username, ip)
    if wait > 0:
        return {}, (f"စမ်းသပ်မှု အကြိမ်များလွန်းပါသည် — {int(wait // 60) + 1} မိနစ် စောင့်ပြီး "
                    "ပြန်လည် ကြိုးစားပါ")
    row = _raw_user_by_name(username)
    if row is None:
        # still burn time so a missing user is not distinguishable
        verify_password(password, "pbkdf2_sha256$200000$AAAA$AAAA")
        _record_attempt(username, ip, False)
        return {}, GENERIC_LOGIN_ERROR
    if (row["status"] or "active") != "active":
        _record_attempt(username, ip, False)
        return {}, "ဤအကောင့်ကို ပိတ်ထားပါသည် — admin ထံ ဆက်သွယ်ပါ"
    if not verify_password(password, row["password_hash"]):
        _record_attempt(username, ip, False)
        return {}, GENERIC_LOGIN_ERROR
    if needs_rehash(row["password_hash"]):
        db.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                   (hash_password(password), row["id"]))
    now = time.time()
    db.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, row["id"]))
    _record_attempt(username, ip, True)
    clear_attempts(username, ip)
    return _row_to_user(row), ""


def change_own_password(user_id: str, current_password: str, new_password: str) -> None:
    db.init()
    row = db.query_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if row is None:
        raise ValueError("အကောင့် ရှာမတွေ့ပါ")
    if not verify_password(current_password, row["password_hash"]):
        raise ValueError("လက်ရှိ စကားဝှက် မမှန်ကန်ပါ")
    if constant_time_equals(current_password, new_password):
        raise ValueError("စကားဝှက် အသစ်သည် အဟောင်းနှင့် မတူရပါ")
    set_password(user_id, new_password)


# ── audit log ────────────────────────────────────────────────────────────
def audit(action: str, *, principal: Optional[Principal] = None, target: str = "",
          detail: str = "", ip: str = "", ok: bool = True) -> None:
    try:
        db.init()
        db.execute(
            "INSERT INTO audit_log (created_at, user_id, username, ip, action, target, detail, ok)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), getattr(principal, "user_id", "") or "",
             getattr(principal, "username", "") or "", ip or "", action[:60],
             str(target)[:200], str(detail)[:400], 1 if ok else 0))
        retention = max(1, int(config.settings.audit_retention_days))
        db.execute("DELETE FROM audit_log WHERE created_at < ?",
                   (time.time() - retention * 86400,))
    except Exception as exc:  # auditing must never break a request
        log.debug("audit write failed: %s", exc)


def recent_audit(limit: int = 100, user_id: str = "") -> list[dict]:
    db.init()
    if user_id:
        rows = db.query("SELECT * FROM audit_log WHERE user_id = ? "
                        "ORDER BY created_at DESC LIMIT ?", (user_id, int(limit)))
    else:
        rows = db.query("SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?",
                        (int(limit),))
    return [dict(r) for r in rows]


# ── per-user API keys ────────────────────────────────────────────────────
def set_user_key(user_id: str, slot: int, key: str) -> None:
    db.init()
    slot = int(slot)
    if slot < 1 or slot > max(1, int(config.settings.max_api_keys)):
        raise ValueError("key slot မမှန်ကန်ပါ")
    key = (key or "").strip()
    if not key:
        db.execute("DELETE FROM user_secrets WHERE user_id = ? AND slot = ?", (user_id, slot))
        return
    db.execute("INSERT OR REPLACE INTO user_secrets (user_id, slot, value, updated_at) "
               "VALUES (?,?,?,?)", (user_id, slot, encrypt_secret(key), time.time()))


def get_user_keys(user_id: str) -> list[str]:
    """Decrypted keys, ordered by slot (empty list when the user has none)."""
    db.init()
    rows = db.query("SELECT slot, value FROM user_secrets WHERE user_id = ? ORDER BY slot",
                    (user_id,))
    keys = [decrypt_secret(r["value"]) for r in rows]
    return [k for k in keys if k]


def describe_user_keys(user_id: str) -> dict:
    from .keys import mask
    db.init()
    rows = db.query("SELECT slot, value, updated_at FROM user_secrets WHERE user_id = ? "
                    "ORDER BY slot", (user_id,))
    slots = []
    present = {int(r["slot"]): r for r in rows}
    for index in range(1, max(1, int(config.settings.max_api_keys)) + 1):
        row = present.get(index)
        key = decrypt_secret(row["value"]) if row else ""
        slots.append({"slot": index, "set": bool(key), "masked": mask(key),
                      "source": "user" if key else "empty", "read_only": False,
                      "updated_at": float(row["updated_at"]) if row else 0.0})
    return {"keys": slots, "any_key": any(s["set"] for s in slots),
            "max_keys": max(1, int(config.settings.max_api_keys))}


# ── job ownership + quotas ───────────────────────────────────────────────
def register_job(job_id: str, user_id: str, kind: str = "recap",
                 status: str = "queued") -> None:
    db.init()
    now = time.time()
    db.execute("INSERT OR REPLACE INTO job_index (job_id, user_id, kind, status, "
               "created_at, updated_at) VALUES (?,?,?,?,"
               "COALESCE((SELECT created_at FROM job_index WHERE job_id = ?), ?), ?)",
               (job_id, user_id or "", kind, status, job_id, now, now))


def job_owner(job_id: str) -> str:
    db.init()
    row = db.query_one("SELECT user_id FROM job_index WHERE job_id = ?", (job_id,))
    return str(row["user_id"]) if row else ""


def forget_job(job_id: str) -> None:
    db.init()
    db.execute("DELETE FROM job_index WHERE job_id = ?", (job_id,))


def jobs_today(user_id: str) -> int:
    db.init()
    row = db.query_one("SELECT COUNT(*) AS n FROM job_index WHERE user_id = ? AND created_at > ?",
                       (user_id, time.time() - 86400))
    return int(row["n"]) if row else 0


def user_limits(user: dict) -> dict:
    """Effective limits = per-user override or the global default."""
    return {
        "max_concurrent_jobs": int(user.get("max_concurrent_jobs") or
                                   config.settings.user_max_concurrent_jobs or 0),
        "daily_job_limit": int(user.get("daily_job_limit") or
                               config.settings.user_daily_jobs or 0),
        "quota_bytes": int(user.get("quota_bytes") or config.settings.user_quota_bytes or 0),
    }


# ── rate limiting (in-process, per worker) ───────────────────────────────
class RateLimiter:
    """Simple sliding-window limiter — enough to stop a hammering script."""

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = max(1, int(limit))
        self.window = float(window_seconds)
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """``(allowed, retry_after_seconds)``."""
        now = time.time()
        with self._lock:
            bucket = [t for t in self._hits.get(key, []) if now - t < self.window]
            if len(bucket) >= self.limit:
                self._hits[key] = bucket
                return False, max(0.5, self.window - (now - bucket[0]))
            bucket.append(now)
            self._hits[key] = bucket
            if len(self._hits) > 4096:        # bounded memory
                cutoff = now - self.window
                self._hits = {k: [t for t in v if t > cutoff]
                              for k, v in self._hits.items() if any(t > cutoff for t in v)}
            return True, 0.0

    def reset(self, key: str = "") -> None:
        with self._lock:
            if key:
                self._hits.pop(key, None)
            else:
                self._hits.clear()


# ── bootstrap ────────────────────────────────────────────────────────────
@dataclass
class BootstrapResult:
    created: bool = False
    username: str = ""
    warnings: list[str] = field(default_factory=list)


def bootstrap() -> BootstrapResult:
    """Create the first admin from the environment, if asked to.

    ``RECAP_ADMIN_USER`` + ``RECAP_ADMIN_PASSWORD`` make an unattended AWS
    install possible: the very first boot creates the admin, later boots do
    nothing (the password in the environment is *not* re-applied, so changing
    it in the UI sticks).
    """
    result = BootstrapResult()
    db.init()
    username = (config.settings.bootstrap_admin_user or "").strip()
    password = config.settings.bootstrap_admin_password or ""
    if username and password:
        if get_user_by_name(username):
            result.username = username
        else:
            try:
                create_user(username, password, role="admin",
                            display_name=username, skip_policy=False)
                result.created = True
                result.username = username
                log.warning("bootstrap admin '%s' created from RECAP_ADMIN_USER/"
                            "RECAP_ADMIN_PASSWORD — remove the password from .env "
                            "after the first login", username)
                audit("bootstrap_admin", target=username, detail="created from env")
            except ValueError as exc:
                result.warnings.append(f"bootstrap admin မဖန်တီးနိုင်ပါ: {exc}")
                log.error("bootstrap admin failed: %s", exc)
    mode = auth_mode()
    if mode == "open":
        result.warnings.append(
            "Authentication မဖွင့်ရသေးပါ — မည်သူမဆို ဝင်နိုင်ပါသည်။ "
            "'python -m recapstudio.useradmin create <name> --admin' ဖြင့် အကောင့် ဖန်တီးပါ။")
    elif mode == "legacy":
        result.warnings.append(
            "Shared access key (RECAP_ACCESS_PASSWORD) ဖြင့်သာ ကာကွယ်ထားပါသည် — "
            "အကောင့်များ ဖန်တီးပြီး multi-user mode သို့ ပြောင်းရန် အကြံပြုပါသည်။")
    return result


def security_report() -> dict:
    """Used by /api/system and tests/verify_deployment.py."""
    from .crypto import encryption_backend, secret_is_persistent
    mode = auth_mode()
    return {
        "auth_mode": mode,
        "users": user_count(),
        "signup_enabled": bool(config.settings.allow_signup),
        "secret_persistent": secret_is_persistent(),
        "encryption": encryption_backend(),
        "session_ttl_hours": config.settings.session_ttl_hours,
        "cookie_secure": config.settings.cookie_secure,
        "rate_limit_per_minute": config.settings.api_rate_limit,
        "login_max_attempts": config.settings.login_max_attempts,
        "db": {k: v for k, v in db.stats().items() if k != "path"},
    }
