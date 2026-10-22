"""Resumable, chunked uploads.

Why uploads used to fail
------------------------
* The browser handler never cleared ``<input type=file>.value``; picking the
  **same file again** after an error does not fire a ``change`` event, so the
  user had to "upload twice" before anything happened.
* A single multi-hundred-MB ``POST`` had no progress, no retry and was easily
  killed by an AWS ALB idle/body limit or a flaky mobile connection - the
  whole transfer restarted from zero.
* Failures returned an HTML 500 page while the client called ``res.json()``,
  so every error surfaced as the same useless "Upload မအောင်မြင်ပါ".

Now: ``init`` → ``chunk`` (8 MB, retried by the client with progress) →
``complete`` (assembles, validates with ffprobe, dedupes by hash).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Optional

from . import config
from .media import get_video_info
from .util import get_logger, human_bytes, safe_rmtree, slugify_filename

log = get_logger("recap.uploads")

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".ts", ".3gp", ".flv", ".wmv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".avif", ".tif", ".tiff"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac"}

MAX_UPLOAD_AGE_SECONDS = 24 * 3600


@dataclass
class UploadSession:
    upload_id: str
    kind: str                      # video | logo | audio
    filename: str
    safe_name: str
    total_size: int
    chunk_size: int
    received: set[int] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    completed_path: str = ""
    #: account that started the upload (multi-user isolation). "" / "shared"
    #: keeps the single-user behaviour.
    owner: str = ""

    @property
    def expected_chunks(self) -> int:
        if self.total_size <= 0:
            return 0
        return (self.total_size + self.chunk_size - 1) // self.chunk_size

    def to_dict(self) -> dict:
        return {
            "upload_id": self.upload_id,
            "kind": self.kind,
            "filename": self.filename,
            "total_size": self.total_size,
            "chunk_size": self.chunk_size,
            "received_chunks": sorted(self.received),
            "expected_chunks": self.expected_chunks,
            "completed_path": self.completed_path,
            "progress": round(len(self.received) / max(1, self.expected_chunks) * 100, 1),
        }

    def owned_by(self, owner: Optional[str]) -> bool:
        if owner is None:                       # internal / legacy caller
            return True
        return (self.owner or "shared") == (owner or "shared")


class UploadManager:
    def __init__(self, incoming_dir: Path = config.INCOMING_DIR,
                 workspace_dir: Path = config.WORKSPACE_DIR):
        self.incoming = Path(incoming_dir)
        self.incoming.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(workspace_dir)
        self.sessions: dict[str, UploadSession] = {}

    # ── helpers ────────────────────────────────────────────────────────
    def _session_dir(self, upload_id: str) -> Path:
        return self.incoming / upload_id

    def _meta_path(self, upload_id: str) -> Path:
        return self._session_dir(upload_id) / "meta.json"

    def _save_meta(self, session: UploadSession) -> None:
        meta = session.to_dict()
        meta["created_at"] = session.created_at
        meta["safe_name"] = session.safe_name
        meta["owner"] = session.owner
        try:
            with open(self._meta_path(session.upload_id), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False)
        except Exception as exc:
            log.warning("could not persist upload meta: %s", exc)

    def _load_session(self, upload_id: str) -> Optional[UploadSession]:
        if upload_id in self.sessions:
            return self.sessions[upload_id]
        meta_path = self._meta_path(upload_id)
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            session = UploadSession(
                upload_id=upload_id,
                kind=meta.get("kind", "video"),
                filename=meta.get("filename", "file"),
                safe_name=meta.get("safe_name", "file"),
                total_size=int(meta.get("total_size", 0)),
                chunk_size=int(meta.get("chunk_size", config.settings.upload_chunk_bytes)),
                received=set(int(i) for i in meta.get("received_chunks", [])),
                created_at=float(meta.get("created_at", time.time())),
                completed_path=meta.get("completed_path", ""),
                owner=str(meta.get("owner", "") or ""),
            )
            self.sessions[upload_id] = session
            return session
        except Exception as exc:
            log.warning("could not read upload meta %s: %s", upload_id, exc)
            return None

    def _validate_extension(self, filename: str, kind: str) -> str:
        ext = Path(filename or "").suffix.lower()
        allowed = {"video": VIDEO_EXT, "logo": IMAGE_EXT, "audio": AUDIO_EXT}.get(kind, VIDEO_EXT)
        if ext not in allowed:
            raise ValueError(
                f"'{ext or 'unknown'}' ဖိုင်အမျိုးအစားကို ပံ့ပိုးမထားပါ။ "
                f"ခွင့်ပြုထားသည်: {', '.join(sorted(allowed))}"
            )
        return ext

    # ── public API ─────────────────────────────────────────────────────
    def find_resumable(self, filename: str, total_size: int, kind: str,
                       owner: str = "") -> Optional[UploadSession]:
        """An unfinished session for the same file (browser refresh / retry)."""
        safe_name = slugify_filename(filename)
        cutoff = time.time() - MAX_UPLOAD_AGE_SECONDS
        try:
            entries = list(self.incoming.iterdir())
        except Exception:
            return None
        for entry in entries:
            if not entry.is_dir():
                continue
            session = self._load_session(entry.name)
            if session is None or session.completed_path:
                continue
            if (session.safe_name == safe_name and session.total_size == int(total_size)
                    and session.kind == kind and session.created_at >= cutoff
                    and session.owned_by(owner)):
                return session
        return None

    def init(self, filename: str, total_size: int, kind: str = "video",
             resume: bool = True, owner: str = "") -> UploadSession:
        kind = kind if kind in {"video", "logo", "audio"} else "video"
        self._validate_extension(filename, kind)
        if total_size <= 0:
            raise ValueError("ဖိုင် size မမှန်ကန်ပါ")
        if total_size > config.settings.max_upload_bytes:
            raise ValueError(
                f"ဖိုင် အရမ်းကြီးနေပါသည် ({human_bytes(total_size)})။ "
                f"အများဆုံး {human_bytes(config.settings.max_upload_bytes)} အထိ ပံ့ပိုးပါသည် "
                "(လိုအပ်ပါက RECAP_MAX_UPLOAD_BYTES ကို ပြင်နိုင်ပါသည်)။"
            )
        if resume:
            existing = self.find_resumable(filename, total_size, kind, owner)
            if existing is not None and existing.received:
                # A refresh / dropped connection should continue where it
                # stopped instead of sending gigabytes again.
                log.info("resuming upload %s (%d/%d chunks already on disk)",
                         existing.upload_id, len(existing.received), existing.expected_chunks)
                return existing
        upload_id = f"up_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        session = UploadSession(
            upload_id=upload_id,
            kind=kind,
            filename=filename,
            safe_name=slugify_filename(filename),
            total_size=int(total_size),
            chunk_size=int(config.settings.upload_chunk_bytes),
            owner=owner or "",
        )
        self._session_dir(upload_id).mkdir(parents=True, exist_ok=True)
        self.sessions[upload_id] = session
        self._save_meta(session)
        log.info("upload %s started (%s, %s, %d chunks)",
                 upload_id, filename, human_bytes(total_size), session.expected_chunks)
        return session

    def session_state(self, upload_id: str, owner: Optional[str] = None) -> Optional[dict]:
        """Public progress of a session (used by /api/upload/status)."""
        session = self._load_session(upload_id)
        if session is None:
            return None
        if not session.owned_by(owner):
            raise PermissionError("ဤ upload session ကို သုံးခွင့် မရှိပါ")
        return session.to_dict()

    def save_chunk(self, upload_id: str, index: int, stream: BinaryIO,
                   owner: Optional[str] = None) -> UploadSession:
        session = self._load_session(upload_id)
        if session is None:
            raise KeyError("Upload session ရှာမတွေ့ပါ (session အဟောင်း ဖြစ်နိုင်ပါသည်)။ ပြန်လည် စတင်ပေးပါ။")
        if not session.owned_by(owner):
            raise PermissionError("ဤ upload session ကို သုံးခွင့် မရှိပါ")
        if session.completed_path:
            return session
        if index < 0 or index >= max(1, session.expected_chunks):
            raise ValueError(f"chunk index {index} သည် အပိုင်းအခြား ပြင်ပ ဖြစ်နေပါသည်")
        chunk_path = self._session_dir(upload_id) / f"part_{index:05d}"
        tmp_path = chunk_path.with_suffix(".tmp")
        written = 0
        with open(tmp_path, "wb") as fh:
            while True:
                buf = stream.read(1024 * 1024)
                if not buf:
                    break
                written += len(buf)
                fh.write(buf)
        if written == 0:
            tmp_path.unlink(missing_ok=True)
            raise ValueError("chunk အလွတ် ရောက်လာပါသည် (retry လုပ်ပေးပါ)")
        os.replace(tmp_path, chunk_path)
        session.received.add(index)
        self._save_meta(session)
        return session

    def complete(self, upload_id: str, owner: Optional[str] = None) -> dict:
        session = self._load_session(upload_id)
        if session is None:
            raise KeyError("Upload session ရှာမတွေ့ပါ။ ပြန်လည် စတင်ပေးပါ။")
        if not session.owned_by(owner):
            raise PermissionError("ဤ upload session ကို သုံးခွင့် မရှိပါ")
        if session.completed_path and Path(config.DATA_DIR / session.completed_path).exists():
            return self._describe(session)

        missing = [i for i in range(session.expected_chunks) if i not in session.received]
        if missing:
            raise ValueError(
                f"chunk {len(missing)} ခု မပြည့်စုံသေးပါ (ဥပမာ #{missing[:5]})။ ပြန်လည် ပေးပို့ပါ။"
            )

        session_dir = self._session_dir(upload_id)
        # Multi-user: every account writes into its own sandbox folder so the
        # asset/download endpoints can enforce ownership by path alone.
        if session.kind == "logo":
            target_dir = config.user_logo_dir(session.owner)
        elif session.owner and config.safe_scope(session.owner) != "shared":
            target_dir = config.user_workspace_dir(session.owner)
        else:
            target_dir = self.workspace
        target_dir.mkdir(parents=True, exist_ok=True)
        final_name = f"{session.kind}_{int(time.time())}_{uuid.uuid4().hex[:6]}_{session.safe_name}"
        final_path = target_dir / final_name

        with open(final_path, "wb") as out:
            for index in range(session.expected_chunks):
                part = session_dir / f"part_{index:05d}"
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, out, length=1024 * 1024)

        actual = final_path.stat().st_size
        if session.total_size and abs(actual - session.total_size) > 1024:
            log.warning("upload %s size mismatch: expected %s got %s",
                        upload_id, session.total_size, actual)

        session.completed_path = str(final_path.relative_to(config.DATA_DIR))
        self._save_meta(session)
        safe_rmtree(session_dir)
        log.info("upload %s completed -> %s (%s)", upload_id, final_path, human_bytes(actual))
        return self._describe(session)

    def _describe(self, session: UploadSession) -> dict:
        path = config.DATA_DIR / session.completed_path
        payload = {
            "status": "ok",
            "upload_id": session.upload_id,
            "kind": session.kind,
            "filename": session.filename,
            "path": session.completed_path,
            "size": path.stat().st_size if path.exists() else 0,
            "preview_url": f"/api/asset?path={session.completed_path}",
        }
        if session.kind == "video" and path.exists():
            info = get_video_info(path)
            payload["video_path"] = session.completed_path  # API compatibility alias
            payload.update({
                "duration": round(info["duration"], 3),
                "width": info["width"],
                "height": info["height"],
                "fps": round(info["fps"], 3),
                "has_audio": info["has_audio"],
                "has_video": info["has_video"],
                "video_codec": info["video_codec"],
                "audio_codec": info["audio_codec"],
            })
            if not info["has_video"]:
                raise ValueError("ဒီဖိုင်ထဲမှာ video stream မတွေ့ပါ။ အသံဖိုင် သို့မဟုတ် ပျက်နေသော ဖိုင် ဖြစ်နိုင်ပါသည်။")
        if session.kind == "logo" and path.exists():
            payload.update(self.logo_assets(path, owner=session.owner))
        return payload

    # ── logos ──────────────────────────────────────────────────────────
    def logo_assets(self, raw_path: Path, owner: str = "") -> dict:
        """Crop a logo into a circular badge, always returning a usable file.

        Previously a failed badge conversion still returned ``status: ok``
        with a path that did not exist, and the whole render then died with a
        cryptic ffmpeg error. Now we always fall back to a resized copy of the
        original image.
        """
        from PIL import Image

        clean_path = config.user_logo_dir(owner) / f"logo_{uuid.uuid4().hex[:10]}.png"
        try:
            img = Image.open(raw_path).convert("RGBA")
            if max(img.size) > 1400:  # keep GPU/CPU overlay cheap
                img.thumbnail((1400, 1400), Image.LANCZOS)
            side = min(img.size)
            left = (img.width - side) // 2
            top = (img.height - side) // 2
            img = img.crop((left, top, left + side, top + side))

            self._strip_flat_background(img)

            from PIL import ImageDraw
            mask = Image.new("L", (side, side), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, side, side), fill=255)
            badge = Image.new("RGBA", (side, side), (0, 0, 0, 0))
            badge.paste(img, (0, 0), mask=mask)

            ring = ImageDraw.Draw(badge)
            width = max(2, side // 40)
            ring.ellipse((width // 2, width // 2, side - width // 2, side - width // 2),
                         outline=(0, 242, 254, 235), width=width)
            badge.save(clean_path, "PNG", optimize=True)
            return {
                "logo_path": str(clean_path.relative_to(config.DATA_DIR)),
                "logo_preview_url": f"/api/asset?path={clean_path.relative_to(config.DATA_DIR)}",
                "logo_processed": True,
            }
        except Exception as exc:
            log.warning("logo processing failed (%s) - falling back to resized copy", exc)
            try:
                img = Image.open(raw_path).convert("RGBA")
                img.thumbnail((600, 600), Image.LANCZOS)
                img.save(clean_path, "PNG")
                return {
                    "logo_path": str(clean_path.relative_to(config.DATA_DIR)),
                    "logo_preview_url": f"/api/asset?path={clean_path.relative_to(config.DATA_DIR)}",
                    "logo_processed": False,
                    "logo_warning": "Logo ကို စက်ဝိုင်း badge အဖြစ် ပြောင်း၍ မရပါ - မူရင်းပုံအတိုင်း အသုံးပြုပါမည်။",
                }
            except Exception as exc2:
                raise ValueError(f"Logo ဖိုင် ဖတ်၍ မရပါ: {exc2}") from exc2

    @staticmethod
    def _strip_flat_background(img) -> None:
        """Make flat white/black logo backgrounds transparent (vectorised)."""
        try:
            import numpy as np
            arr = np.array(img)
            if arr.ndim != 3 or arr.shape[2] < 4:
                return
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            white = (r > 215) & (g > 215) & (b > 215)
            black = (r < 28) & (g < 28) & (b < 28)
            white_ratio = white.mean()
            black_ratio = black.mean()
            # only strip when the colour is clearly a background, otherwise a
            # dark or bright logo would be erased entirely.
            if 0.18 < white_ratio < 0.95:
                arr[white] = [255, 255, 255, 0]
            if 0.18 < black_ratio < 0.95:
                arr[black] = [0, 0, 0, 0]
            from PIL import Image
            img.paste(Image.fromarray(arr, "RGBA"), (0, 0))
        except Exception:
            return

    # ── janitor ────────────────────────────────────────────────────────
    def cleanup_stale(self) -> int:
        removed = 0
        cutoff = time.time() - MAX_UPLOAD_AGE_SECONDS
        for entry in self.incoming.iterdir() if self.incoming.exists() else []:
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    safe_rmtree(entry)
                    self.sessions.pop(entry.name, None)
                    removed += 1
            except Exception:
                continue
        return removed


def sha1_of_file(path: str | os.PathLike, limit_mb: int = 512) -> str:
    digest = hashlib.sha1()
    read = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            read += len(block)
            if read > limit_mb * 1024 * 1024:
                break
    return digest.hexdigest()


upload_manager = UploadManager()
