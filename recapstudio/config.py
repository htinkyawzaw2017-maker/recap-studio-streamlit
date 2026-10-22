"""Environment driven configuration and paths.

Everything that used to be hard-coded (workspace/output folders, thread
counts, timeouts, quality presets) lives here so an AWS deployment can be
tuned purely with environment variables / ECS task definitions - no code
change required.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


def _load_env_files() -> list[str]:
    """Load .env files before any os.getenv() call.

    systemd deployments pass variables via EnvironmentFile and docker via the
    compose file, but a plain ``uvicorn app:app`` run (or the installer's tmux
    fallback) would otherwise start *without* the API key the user saved in
    ``.env`` — which is exactly the "key ပျောက်" report. Real environment
    variables always win (override=False).
    """
    loaded: list[str] = []
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:  # python-dotenv not installed — env vars only
        return loaded
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / ".env",
        Path(os.getenv("RECAP_DATA_DIR", str(root))) / ".env",
        Path.cwd() / ".env",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            path = candidate.resolve()
        except OSError:
            continue
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if load_dotenv(path, override=False):
            loaded.append(str(path))
    return loaded


ENV_FILES_LOADED = _load_env_files()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y", "enabled"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


APP_ROOT = Path(__file__).resolve().parent.parent

# Where runtime data (uploads, renders, cache) is written. On AWS ECS point
# this at a mounted EFS volume / big EBS scratch disk, e.g. RECAP_DATA_DIR=/data
DATA_DIR = Path(os.getenv("RECAP_DATA_DIR", str(APP_ROOT))).resolve()
if not DATA_DIR.is_absolute():
    DATA_DIR = (APP_ROOT / DATA_DIR).resolve()

WORKSPACE_DIR = DATA_DIR / "workspace"
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = WORKSPACE_DIR / "cache"
UPLOAD_DIR = WORKSPACE_DIR / "uploads"
INCOMING_DIR = WORKSPACE_DIR / "incoming"
TASK_DIR = WORKSPACE_DIR / "tasks"
LOGO_DIR = WORKSPACE_DIR / "logos"
TMP_DIR = WORKSPACE_DIR / "tmp"

USERS_DIR = WORKSPACE_DIR / "users"          # per-account upload sandbox
STATIC_DIR = APP_ROOT / "static"
ASSETS_DIR = APP_ROOT / "assets"
BUNDLED_FONT_DIR = ASSETS_DIR / "fonts"

CONFIG_FILE = DATA_DIR / ".recap_config.json"
#: accounts / sessions / audit log (Phase 2). Created with mode 0600.
DB_PATH = Path(os.getenv("RECAP_DB_PATH", str(DATA_DIR / "recap.db")))
if not DB_PATH.is_absolute():
    DB_PATH = (DATA_DIR / DB_PATH).resolve()
#: auto-generated server secret (encrypts stored API keys) when
#: RECAP_SECRET_KEY is not provided.
SECRET_FILE = DATA_DIR / ".recap_secret"


@dataclass(frozen=True)
class Settings:
    """Runtime tunables (all overridable via env)."""

    # ── AI ────────────────────────────────────────────────────────────────
    default_model: str = os.getenv("RECAP_DEFAULT_MODEL", "gemini-2.5-flash")
    models: tuple = (
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    )
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    #: extra keys so a quota-exhausted key does not kill a long job.
    #: Slot 2 / 3 can be given in the UI or with GEMINI_API_KEY_2 / _3.
    gemini_api_key_2: str = os.getenv("GEMINI_API_KEY_2", "")
    gemini_api_key_3: str = os.getenv("GEMINI_API_KEY_3", "")
    #: how many key slots the UI is allowed to fill (1 = classic behaviour)
    max_api_keys: int = max(1, _env_int("RECAP_MAX_API_KEYS", 30))
    #: when a key answers 429 / quota / 403 the engine switches to the next
    #: slot automatically instead of failing the whole job
    api_key_failover: bool = _env_bool("RECAP_API_KEY_FAILOVER", True)
    #: seconds a failed key is skipped before it is tried again
    api_key_cooldown: int = _env_int("RECAP_API_KEY_COOLDOWN", 90)
    #: longest single video chunk handed to Gemini (Files API / context safe)
    max_chunk_seconds: int = _env_int("RECAP_MAX_CHUNK_SECONDS", 480)
    #: target number of chunks for very long videos
    target_chunks: int = _env_int("RECAP_TARGET_CHUNKS", 8)
    analyze_workers: int = _env_int("RECAP_ANALYZE_WORKERS", 2)
    ai_max_output_tokens: int = _env_int("RECAP_AI_MAX_TOKENS", 32768)
    ai_retries: int = _env_int("RECAP_AI_RETRIES", 4)
    ai_timeout_seconds: int = _env_int("RECAP_AI_TIMEOUT", 600)
    #: second pass: re-ask a chunk whose answer stopped before the clip end
    ai_tail_pass: bool = _env_bool("RECAP_AI_TAIL_PASS", True)
    #: max narration silence tolerated between two lines (continuous mode)
    max_narration_gap: float = _env_float("RECAP_MAX_NARRATION_GAP", 5.0)
    #: rounds of "fill the silence → re-synthesise" repair after TTS
    coverage_repair_rounds: int = _env_int("RECAP_COVERAGE_REPAIR_ROUNDS", 2)

    # ── TTS ───────────────────────────────────────────────────────────────
    tts_workers: int = _env_int("RECAP_TTS_WORKERS", 8)
    tts_retries: int = _env_int("RECAP_TTS_RETRIES", 3)
    max_tempo: float = _env_float("RECAP_MAX_TEMPO", 1.6)
    max_lag_seconds: float = _env_float("RECAP_MAX_LAG", 0.35)
    default_voice_key: str = "thiha"

    # ── Render ────────────────────────────────────────────────────────────
    render_workers: int = _env_int("RECAP_RENDER_WORKERS", 2)
    video_preset: str = os.getenv("RECAP_X264_PRESET", "veryfast")
    video_crf: int = _env_int("RECAP_X264_CRF", 23)
    audio_bitrate: str = os.getenv("RECAP_AUDIO_BITRATE", "192k")
    loudness_normalize: bool = _env_bool("RECAP_LOUDNORM", True)
    #: 0 = let ffmpeg decide, otherwise cap x264 threads (small EC2 instances
    #: stay responsive while a render runs)
    ffmpeg_threads: int = _env_int("RECAP_FFMPEG_THREADS", 0)
    #: a render that produces no output for this many seconds is treated as
    #: stuck (killed + reported) instead of hanging the job forever
    ffmpeg_stall_seconds: int = _env_int("RECAP_FFMPEG_STALL_SECONDS", 900)

    # ── Limits / housekeeping ─────────────────────────────────────────────
    max_upload_bytes: int = _env_int("RECAP_MAX_UPLOAD_BYTES", 20 * 1024 * 1024 * 1024)
    upload_chunk_bytes: int = _env_int("RECAP_UPLOAD_CHUNK_BYTES", 8 * 1024 * 1024)
    min_free_disk_bytes: int = _env_int("RECAP_MIN_FREE_DISK_BYTES", 2 * 1024 * 1024 * 1024)
    workspace_ttl_hours: float = _env_float("RECAP_WORKSPACE_TTL_HOURS", 12.0)
    output_ttl_hours: float = _env_float("RECAP_OUTPUT_TTL_HOURS", 72.0)
    janitor_interval_seconds: int = _env_int("RECAP_JANITOR_INTERVAL", 1800)
    max_concurrent_jobs: int = _env_int("RECAP_MAX_CONCURRENT_JOBS", 2)

    # ── Access / security ─────────────────────────────────────────────────
    access_password: str = os.getenv("RECAP_ACCESS_PASSWORD", "")
    allowed_origins: tuple = field(
        default_factory=lambda: tuple(
            o.strip() for o in os.getenv("RECAP_ALLOWED_ORIGINS", "").split(",") if o.strip()
        )
    )
    ytdlp_cookies_file: str = os.getenv("RECAP_YTDLP_COOKIES", "")
    ytdlp_proxy: str = os.getenv("RECAP_YTDLP_PROXY", "")

    # ── Auth / accounts / multi-user (Phase 2 — #12) ──────────────────────
    #: auto = accounts when the DB has any, else the shared key, else open.
    #: users | legacy | open force one specific mode.
    auth_mode: str = os.getenv("RECAP_AUTH_MODE", "auto").strip().lower()
    #: encrypts stored API keys; falls back to DATA_DIR/.recap_secret
    secret_key: str = os.getenv("RECAP_SECRET_KEY", "")
    session_cookie: str = os.getenv("RECAP_SESSION_COOKIE", "recap_session")
    csrf_cookie: str = os.getenv("RECAP_CSRF_COOKIE", "recap_csrf")
    session_ttl_hours: float = _env_float("RECAP_SESSION_TTL_HOURS", 168.0)   # 7 days
    session_idle_hours: float = _env_float("RECAP_SESSION_IDLE_HOURS", 72.0)
    #: auto = Secure flag when the request arrived over https
    cookie_secure: str = os.getenv("RECAP_COOKIE_SECURE", "auto").strip().lower()
    cookie_samesite: str = os.getenv("RECAP_COOKIE_SAMESITE", "lax").strip().lower()
    allow_signup: bool = _env_bool("RECAP_ALLOW_SIGNUP", False)
    signup_code: str = os.getenv("RECAP_SIGNUP_CODE", "")
    bootstrap_admin_user: str = os.getenv("RECAP_ADMIN_USER", "")
    bootstrap_admin_password: str = os.getenv("RECAP_ADMIN_PASSWORD", "")
    password_min_length: int = _env_int("RECAP_PASSWORD_MIN_LENGTH", 10)
    pbkdf2_iterations: int = _env_int("RECAP_PBKDF2_ITERATIONS", 240_000)
    login_max_attempts: int = _env_int("RECAP_LOGIN_MAX_ATTEMPTS", 8)
    login_window_minutes: int = _env_int("RECAP_LOGIN_WINDOW_MINUTES", 15)
    login_lockout_minutes: int = _env_int("RECAP_LOGIN_LOCKOUT_MINUTES", 15)
    audit_retention_days: int = _env_int("RECAP_AUDIT_RETENTION_DAYS", 90)
    #: generic API rate limit per client per minute (0 = off)
    api_rate_limit: int = _env_int("RECAP_RATE_LIMIT_PER_MINUTE", 300)
    login_rate_limit: int = _env_int("RECAP_LOGIN_RATE_LIMIT_PER_MINUTE", 12)
    #: per-user quotas (0 = unlimited / fall back to the global setting)
    user_max_concurrent_jobs: int = _env_int("RECAP_USER_MAX_CONCURRENT_JOBS", 1)
    user_daily_jobs: int = _env_int("RECAP_USER_DAILY_JOBS", 0)
    user_quota_bytes: int = _env_int("RECAP_USER_QUOTA_BYTES", 0)
    #: response hardening headers (HSTS only when the request is https)
    security_headers: bool = _env_bool("RECAP_SECURITY_HEADERS", True)
    #: none | self | * — '*' allows embedding the UI in another site's iframe
    frame_ancestors: str = os.getenv("RECAP_FRAME_ANCESTORS", "self").strip().lower()
    hsts_seconds: int = _env_int("RECAP_HSTS_SECONDS", 0)
    #: comma separated Host allow-list (empty = accept any Host header)
    trusted_hosts: tuple = field(
        default_factory=lambda: tuple(
            h.strip().lower() for h in os.getenv("RECAP_TRUSTED_HOSTS", "").split(",")
            if h.strip()
        )
    )

    # ── Dev / test switches (never enable on a public deployment) ──────────
    demo_mode: bool = _env_bool("RECAP_DEMO_MODE", False)          # fake AI timeline
    fake_tts: bool = _env_bool("RECAP_FAKE_TTS", False)            # beep instead of speech
    force_reencode: bool = _env_bool("RECAP_FORCE_REENCODE", False)

    app_version: str = "4.4.0"


settings = Settings()


def ensure_dirs() -> None:
    for path in (
        WORKSPACE_DIR, OUTPUT_DIR, CACHE_DIR, UPLOAD_DIR, INCOMING_DIR,
        TASK_DIR, LOGO_DIR, TMP_DIR, USERS_DIR, STATIC_DIR / "fonts",
    ):
        path.mkdir(parents=True, exist_ok=True)


# ── per-user sandboxes (multi-user hardening) ────────────────────────────
_SCOPE_RE = re.compile(r"[^a-zA-Z0-9_-]")


def safe_scope(scope: str | None) -> str:
    """Normalise an ownership scope into something usable as a folder name."""
    cleaned = _SCOPE_RE.sub("", str(scope or "").strip())[:48]
    return cleaned or "shared"


def user_workspace_dir(scope: str | None, create: bool = True) -> Path:
    """Where this account's uploads live (``workspace/users/<scope>``).

    ``shared`` keeps the pre-Phase-2 layout so single-user installs and old
    links keep working after an upgrade.
    """
    name = safe_scope(scope)
    path = WORKSPACE_DIR if name == "shared" else (USERS_DIR / name)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def user_logo_dir(scope: str | None, create: bool = True) -> Path:
    name = safe_scope(scope)
    path = LOGO_DIR if name == "shared" else (USERS_DIR / name / "logos")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def user_output_dir(scope: str | None, create: bool = True) -> Path:
    """Rendered artefacts (``output/u_<scope>``) — never shared between users."""
    name = safe_scope(scope)
    path = OUTPUT_DIR if name == "shared" else (OUTPUT_DIR / f"u_{name}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def owned_roots(scope: str | None) -> tuple[Path, ...]:
    """Directories a given scope is allowed to read files from."""
    name = safe_scope(scope)
    if name == "shared":
        return (DATA_DIR.resolve(),)
    return (
        user_workspace_dir(name, create=False).resolve(),
        user_output_dir(name, create=False).resolve(),
        CACHE_DIR.resolve(),
    )


def path_is_owned(path: os.PathLike | str, scope: str | None) -> bool:
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    for root in owned_roots(scope):
        if resolved == root or root in resolved.parents:
            return True
    return False


def resolve_owned(rel_or_abs: str, scope: str | None) -> Path:
    """:func:`resolve` + "the caller actually owns this file".

    Raises ``ValueError`` for traversal *and* for a valid path that belongs to
    somebody else — the second half is what stops user A from streaming user
    B's upload by guessing its name.
    """
    resolved = resolve(rel_or_abs)
    if not path_is_owned(resolved, scope):
        raise ValueError("ဤဖိုင်ကို ကြည့်ခွင့် မရှိပါ (another account's file)")
    return resolved


def rel(path: os.PathLike | str) -> str:
    """Path relative to DATA_DIR when possible (keeps API payloads short)."""
    try:
        p = Path(path).resolve()
        return str(p.relative_to(DATA_DIR))
    except Exception:
        return str(path)


def resolve(rel_or_abs: str) -> Path:
    """Resolve an API supplied path safely inside DATA_DIR.

    Raises ValueError for anything that tries to escape the data root
    (path traversal hardening - important on a public AWS deployment).
    """
    if not rel_or_abs:
        raise ValueError("empty path")
    raw = Path(rel_or_abs)
    candidate = raw if raw.is_absolute() else (DATA_DIR / raw)
    resolved = candidate.resolve()
    root = DATA_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path outside data dir: {rel_or_abs}")
    return resolved
