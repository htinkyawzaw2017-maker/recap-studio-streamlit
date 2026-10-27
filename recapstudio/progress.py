"""Project persistence (v1.11): a page refresh must NOT send the user back
to step 1.

Streamlit keeps workflow state in memory, so a browser refresh starts a
fresh session and the whole 6-step flow (upload → split → analysis →
script edit → render) is lost. This module stores the workflow state on
disk under ``<data_dir>/project_<id>/progress.json``; the project id
travels in the URL (``?project=...``), which survives refreshes. On a
fresh session the app restores the saved state before rendering, so the
user lands back on the exact step they were on — with their video,
script and finished outputs intact.

What is NOT persisted: Gemini keys (they live in browser localStorage by
design), in-editor keystrokes that were never confirmed (only the last
confirmed script + the latest editor draft are saved), and a render that
was mid-flight when the refresh happened (renders run inside the
session, so an interrupted render restarts — but from the confirmed
script, not from step 1).
"""

from __future__ import annotations

import json
from pathlib import Path

# Session-state keys that make up a restorable project. Anything not
# listed here is either secret (keys), derived (step pills), or
# transient (busy/logs).
PERSIST_KEYS = (
    "video_path", "video_info", "parts", "current_path", "current_label",
    "dialogues", "hook1", "hook2", "coverage", "fit", "repair_unfixed",
    "script_version",
    "script_confirmed", "edited_dialogues", "draft_dialogues", "outputs",
    "render_cfg",
)


def project_dir(data_dir: str | Path, pid: str) -> Path:
    return Path(data_dir) / f"project_{pid}"


def progress_path(data_dir: str | Path, pid: str) -> Path:
    return project_dir(data_dir, pid) / "progress.json"


def save_progress(data_dir: str | Path, pid: str, state: dict) -> bool:
    """Write the restorable slice of ``state`` to disk. Never raises."""
    if not pid:
        return False
    try:
        d = project_dir(data_dir, pid)
        d.mkdir(parents=True, exist_ok=True)
        data = {k: state.get(k) for k in PERSIST_KEYS}
        progress_path(data_dir, pid).write_text(
            json.dumps(data, ensure_ascii=False, default=str))
        return True
    except Exception:
        return False


def load_progress(data_dir: str | Path, pid: str) -> dict | None:
    """Read a saved project back. None when missing or unreadable —
    a corrupt save must never block the app from starting fresh."""
    if not pid:
        return None
    try:
        data = json.loads(progress_path(data_dir, pid).read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return {k: data[k] for k in PERSIST_KEYS if k in data}
