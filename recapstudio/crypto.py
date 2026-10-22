"""Secret handling: server key, password hashing, at-rest encryption.

Everything here is stdlib only (``hashlib``/``hmac``/``secrets``) so a small
EC2 box does not need extra wheels.

* **Server secret** — ``RECAP_SECRET_KEY`` or an auto-generated
  ``DATA_DIR/.recap_secret`` file (0600). Session cookies are random tokens
  (not signed blobs), so rotating the secret only invalidates stored API
  keys, never the login state.
* **Passwords** — PBKDF2-HMAC-SHA256, per-user salt, configurable iteration
  count, stored as ``pbkdf2_sha256$iterations$salt$hash``. Verification is
  constant time and reports when a hash should be upgraded.
* **At-rest encryption** — used for the per-user Gemini API keys. AES is not
  in the stdlib, so this is an *encrypt-then-MAC* construction built on
  HMAC-SHA256 (HMAC used as a PRF in counter mode + an HMAC-SHA256 tag over
  the ciphertext). Keys never touch the disk in clear text and a tampered
  blob is rejected. If ``cryptography`` happens to be installed we use
  Fernet (AES-128-CBC + HMAC) instead and mark the blob ``v2:``.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import stat
from pathlib import Path
from typing import Optional

from . import config
from .util import get_logger

log = get_logger("recap.crypto")

_SECRET_CACHE: Optional[bytes] = None


# ── server secret ────────────────────────────────────────────────────────
def server_secret() -> bytes:
    """32 bytes of key material shared by this deployment."""
    global _SECRET_CACHE
    if _SECRET_CACHE:
        return _SECRET_CACHE
    env = (config.settings.secret_key or "").strip()
    if env:
        _SECRET_CACHE = hashlib.sha256(env.encode("utf-8")).digest()
        return _SECRET_CACHE
    path = Path(config.SECRET_FILE)
    try:
        if path.exists():
            raw = path.read_text(encoding="utf-8").strip()
            if len(raw) >= 32:
                _SECRET_CACHE = hashlib.sha256(raw.encode("utf-8")).digest()
                return _SECRET_CACHE
        path.parent.mkdir(parents=True, exist_ok=True)
        generated = secrets.token_urlsafe(48)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(generated, encoding="utf-8")
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, path)
        log.warning("generated a new server secret at %s — set RECAP_SECRET_KEY "
                    "in .env to keep it stable across re-installs", path)
        _SECRET_CACHE = hashlib.sha256(generated.encode("utf-8")).digest()
        return _SECRET_CACHE
    except Exception as exc:  # pragma: no cover - read-only filesystem
        log.error("could not persist a server secret (%s) — using an ephemeral one", exc)
        _SECRET_CACHE = secrets.token_bytes(32)
        return _SECRET_CACHE


def reset_secret_cache() -> None:
    global _SECRET_CACHE
    _SECRET_CACHE = None


def secret_is_persistent() -> bool:
    return bool((config.settings.secret_key or "").strip()) or Path(config.SECRET_FILE).exists()


# ── password hashing ─────────────────────────────────────────────────────
def hash_password(password: str, iterations: int | None = None) -> str:
    if not isinstance(password, str) or not password:
        raise ValueError("password မထည့်ရသေးပါ")
    rounds = int(iterations or config.settings.pbkdf2_iterations)
    rounds = max(50_000, rounds)
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return "pbkdf2_sha256${}${}${}".format(
        rounds,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, salt_b64, hash_b64 = (stored or "").split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        digest = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"),
                                     salt, int(rounds))
    except Exception:
        return False
    return hmac.compare_digest(digest, expected)


def needs_rehash(stored: str) -> bool:
    try:
        algo, rounds, _, _ = (stored or "").split("$", 3)
    except ValueError:
        return True
    return algo != "pbkdf2_sha256" or int(rounds) < int(config.settings.pbkdf2_iterations)


def hash_token(token: str) -> str:
    """Session tokens are stored hashed — a DB leak cannot replay a login."""
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest((a or "").encode("utf-8"), (b or "").encode("utf-8"))


# ── at-rest encryption (per-user API keys) ───────────────────────────────
def _fernet():
    try:  # optional dependency, nothing breaks when it is missing
        from cryptography.fernet import Fernet  # type: ignore
    except Exception:
        return None
    key = base64.urlsafe_b64encode(server_secret())
    try:
        return Fernet(key)
    except Exception:  # pragma: no cover
        return None


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hmac.new(key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def encrypt_secret(plain: str) -> str:
    """Encrypt a small secret (API key) for storage in SQLite."""
    if plain is None:
        plain = ""
    data = plain.encode("utf-8")
    fernet = _fernet()
    if fernet is not None:
        return "v2:" + fernet.encrypt(data).decode("ascii")
    master = server_secret()
    nonce = secrets.token_bytes(16)
    enc_key = hashlib.sha256(b"enc" + master + nonce).digest()
    mac_key = hashlib.sha256(b"mac" + master + nonce).digest()
    cipher = bytes(a ^ b for a, b in zip(data, _keystream(enc_key, nonce, len(data))))
    tag = hmac.new(mac_key, nonce + cipher, hashlib.sha256).digest()[:16]
    return "v1:{}:{}:{}".format(
        base64.b64encode(nonce).decode("ascii"),
        base64.b64encode(cipher).decode("ascii"),
        base64.b64encode(tag).decode("ascii"),
    )


def decrypt_secret(blob: str) -> str:
    """Inverse of :func:`encrypt_secret`; returns "" for tampered/unknown data."""
    blob = (blob or "").strip()
    if not blob:
        return ""
    try:
        if blob.startswith("v2:"):
            fernet = _fernet()
            if fernet is None:
                log.warning("encrypted secret needs the 'cryptography' package to be read")
                return ""
            return fernet.decrypt(blob[3:].encode("ascii")).decode("utf-8")
        if blob.startswith("v1:"):
            _, nonce_b64, cipher_b64, tag_b64 = blob.split(":", 3)
            nonce = base64.b64decode(nonce_b64)
            cipher = base64.b64decode(cipher_b64)
            tag = base64.b64decode(tag_b64)
            master = server_secret()
            enc_key = hashlib.sha256(b"enc" + master + nonce).digest()
            mac_key = hashlib.sha256(b"mac" + master + nonce).digest()
            expected = hmac.new(mac_key, nonce + cipher, hashlib.sha256).digest()[:16]
            if not hmac.compare_digest(tag, expected):
                log.warning("stored secret failed its integrity check — ignoring")
                return ""
            data = bytes(a ^ b for a, b in zip(cipher, _keystream(enc_key, nonce, len(cipher))))
            return data.decode("utf-8")
    except Exception as exc:
        log.warning("could not decrypt a stored secret: %s", exc)
    return ""


def encryption_backend() -> str:
    return "fernet(cryptography)" if _fernet() is not None else "hmac-sha256-ctr(stdlib)"
