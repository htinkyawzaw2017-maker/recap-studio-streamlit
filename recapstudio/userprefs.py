"""Per-account Studio preferences stored in the shared SQLite database.

Only presentation/workflow choices belong here; API secrets are stored by
:mod:`recapstudio.auth` in their encrypted key table. In account mode the
scope is the user's immutable database id, so preferences follow that user
between browsers and devices.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from . import db

DEFAULTS: dict[str, Any] = {
    "lang": "my",
    "mode": "auto",
    "fill_mode": "continuous",
    "voice": "thiha",
    "quality": "balanced",
    "output_aspect": "9:16",
    "reframe_mode": "Smart Blur Background",
    "enable_subtitles": True,
    "sub_font_size": 42,
    "sub_color_hex": "#00f2fe",
    "sub_bg_style": "Solid Box",
    "sub_v_pos_percent": 22,
    "sub_width_percent": 90,
}

_ALLOWED = {
    "lang": {"my", "en"},
    "mode": {"auto", "movie", "experiment", "craft", "news"},
    "fill_mode": {"continuous", "dialogue"},
    "voice": {"thiha", "nilar", "christopher", "jenny", "guy", "aria"},
    "quality": {"fast", "balanced", "quality"},
    "output_aspect": {"9:16", "1:1", "16:9", "original"},
    "reframe_mode": {"Smart Blur Background", "Center Crop", "Original (no reframe)"},
    "sub_bg_style": {"Solid Box", "Semi Transparent Box", "Outline Only"},
}
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def _clamp_int(value: Any, low: int, high: int, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return max(low, min(high, parsed))


def normalise(payload: dict[str, Any] | None,
              base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a partial preference update and merge it with current values."""
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("Preferences must be an object")
    result = {**DEFAULTS, **(base or {})}
    for key, choices in _ALLOWED.items():
        if key not in payload:
            continue
        value = str(payload[key])
        if value not in choices:
            raise ValueError(f"Invalid value for {key}")
        result[key] = value
    if "enable_subtitles" in payload:
        value = payload["enable_subtitles"]
        if not isinstance(value, bool):
            raise ValueError("enable_subtitles must be true or false")
        result["enable_subtitles"] = value
    if "sub_font_size" in payload:
        result["sub_font_size"] = _clamp_int(payload["sub_font_size"], 24, 72,
                                             "sub_font_size")
    if "sub_v_pos_percent" in payload:
        result["sub_v_pos_percent"] = _clamp_int(
            payload["sub_v_pos_percent"], 3, 92, "sub_v_pos_percent")
    if "sub_width_percent" in payload:
        result["sub_width_percent"] = _clamp_int(
            payload["sub_width_percent"], 60, 96, "sub_width_percent")
    if "sub_color_hex" in payload:
        color = str(payload["sub_color_hex"]).strip()
        if not _COLOR.fullmatch(color):
            raise ValueError("sub_color_hex must be a #RRGGBB colour")
        result["sub_color_hex"] = color.lower()
    return result


def get(scope: str) -> dict[str, Any]:
    db.init()
    row = db.query_one("SELECT preferences_json FROM user_preferences WHERE scope = ?",
                       (str(scope),))
    if not row:
        return dict(DEFAULTS)
    try:
        stored = json.loads(row["preferences_json"])
    except (TypeError, json.JSONDecodeError):
        stored = {}
    return normalise(stored)


def save(scope: str, payload: dict[str, Any]) -> dict[str, Any]:
    db.init()
    current = get(scope)
    merged = normalise(payload, current)
    db.execute(
        """INSERT INTO user_preferences (scope, preferences_json, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT(scope) DO UPDATE SET
             preferences_json = excluded.preferences_json,
             updated_at = excluded.updated_at""",
        (str(scope), json.dumps(merged, ensure_ascii=False, separators=(",", ":")), time.time()),
    )
    return merged


def delete(scope: str) -> None:
    db.init()
    db.execute("DELETE FROM user_preferences WHERE scope = ?", (str(scope),))
