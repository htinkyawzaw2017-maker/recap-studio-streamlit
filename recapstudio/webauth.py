"""FastAPI glue for :mod:`recapstudio.auth` — cookies, CSRF, principals.

Kept out of ``app.py`` so the routing file stays readable and so the auth
rules can be unit-tested without starting a server.

Request → principal resolution
------------------------------
1. ``mode = open``   → everybody is the shared ``public`` admin (v4.1 behaviour)
2. ``mode = legacy`` → ``X-Access-Key`` must equal ``RECAP_ACCESS_PASSWORD``
3. ``mode = users``  → ``recap_session`` cookie **or**
   ``Authorization: Bearer <session-token>`` (handy for curl / CI)

Cookie sessions additionally require a CSRF token on every unsafe method, so
another site cannot make a logged-in browser start renders in the background.
"""
from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request, Response

from . import auth, config
from .auth import Principal
from .crypto import constant_time_equals
from .util import get_logger

log = get_logger("recap.webauth")

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

LOGIN_REQUIRED = "ဝင်ရောက်ရန် လိုအပ်ပါသည် — အကောင့်ဖြင့် login ဝင်ပါ (Sign in required)"
ACCESS_KEY_REQUIRED = ("Access Key မှန်ကန်မှု မရှိပါ — ⚙️ Settings → Server Access Key "
                       "တွင် server ၏ RECAP_ACCESS_PASSWORD ကို ထည့်ပါ")
CSRF_FAILED = "CSRF token မမှန်ကန်ပါ — စာမျက်နှာကို refresh လုပ်ပြီး ပြန်ကြိုးစားပါ"


# ── request helpers ──────────────────────────────────────────────────────
def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:60]
    return (request.client.host if request.client else "")[:60]


def request_is_https(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return proto == "https" or request.url.scheme == "https"


def cookie_secure_flag(request: Request) -> bool:
    setting = (config.settings.cookie_secure or "auto").lower()
    if setting in {"1", "true", "yes", "on"}:
        return True
    if setting in {"0", "false", "no", "off"}:
        return False
    return request_is_https(request)


def set_session_cookie(response: Response, request: Request, token: str, csrf: str,
                       max_age: int) -> None:
    secure = cookie_secure_flag(request)
    samesite = config.settings.cookie_samesite if \
        config.settings.cookie_samesite in {"lax", "strict", "none"} else "lax"
    response.set_cookie(
        config.settings.session_cookie, token, max_age=max_age, httponly=True,
        secure=secure, samesite=samesite, path="/",
    )
    # readable by JS on purpose: the SPA echoes it back in X-CSRF-Token
    response.set_cookie(
        config.settings.csrf_cookie, csrf, max_age=max_age, httponly=False,
        secure=secure, samesite=samesite, path="/",
    )


def clear_session_cookie(response: Response) -> None:
    for name in (config.settings.session_cookie, config.settings.csrf_cookie):
        response.delete_cookie(name, path="/")


def bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


# ── principal resolution ─────────────────────────────────────────────────
def resolve_principal(request: Request) -> Optional[Principal]:
    """The caller, or ``None`` when the request is not authenticated."""
    mode = auth.auth_mode()
    if mode == "open":
        return Principal(user_id="public", username="public", role="admin", mode="open")
    if mode == "legacy":
        supplied = request.headers.get("x-access-key", "") or bearer_token(request)
        expected = config.settings.access_password
        if expected and not constant_time_equals(supplied.strip(), expected):
            return None
        return Principal(user_id="shared", username="shared", role="admin", mode="legacy")

    token = request.cookies.get(config.settings.session_cookie, "")
    via_cookie = bool(token)
    if not token:
        token = bearer_token(request)
    if not token:
        return None
    user, session = auth.resolve_token(token)
    if not user:
        return None
    return Principal(
        user_id=user["id"], username=user["username"], role=user.get("role", "user"),
        mode="users", session_token=token, via_cookie=via_cookie,
        must_change_password=bool(user.get("must_change_password")),
    )


def check_csrf(request: Request, principal: Principal) -> None:
    if principal.mode != "users" or not principal.via_cookie:
        return
    if request.method.upper() in SAFE_METHODS:
        return
    sent = request.headers.get("x-csrf-token", "")
    expected = request.cookies.get(config.settings.csrf_cookie, "")
    _, session = auth.resolve_token(principal.session_token)
    server_token = session.get("csrf_token", "") if session else ""
    if not sent or not server_token or not constant_time_equals(sent, server_token):
        # a cookie-only double submit is still better than nothing, but the
        # server-side value is authoritative
        if not (sent and expected and constant_time_equals(sent, expected)):
            raise HTTPException(status_code=403, detail=CSRF_FAILED)


def require_principal(request: Request) -> Principal:
    """FastAPI dependency: 401 unless the caller is authenticated."""
    principal = resolve_principal(request)
    if principal is None:
        mode = auth.auth_mode()
        detail = ACCESS_KEY_REQUIRED if mode == "legacy" else LOGIN_REQUIRED
        headers = {"WWW-Authenticate": "Bearer"} if mode == "users" else None
        raise HTTPException(status_code=401, detail=detail, headers=headers)
    check_csrf(request, principal)
    request.state.principal = principal
    return principal


def require_admin(request: Request) -> Principal:
    principal = require_principal(request)
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="ဤလုပ်ဆောင်ချက်အတွက် admin ဖြစ်ရပါမည်")
    return principal


def optional_principal(request: Request) -> Optional[Principal]:
    """Never raises — used by endpoints that must answer before login."""
    try:
        return resolve_principal(request)
    except Exception:  # pragma: no cover - defensive
        return None
