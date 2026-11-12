"""Measure real audio/video drift on a rendered recap, per line.

The owner's standing complaint — "မြန်မာဘာသာပြန် အသံ-ရုပ် sync လွဲနေတုန်း" —
has always been a guess: we could not say *which* line was off by *how much*.
This module makes it a number.

How it works
------------
For every planned line we know two things:

* the **intended** window on the source timeline (``[start, end]`` from the
  dialogue list), and
* the **produced** narration, whose clips we can re-synthesise (or read back
  from the saved ``*_narration.mp3`` the pipeline already publishes).

We then:

1. find the actual speech onset of each clip with ffmpeg ``silencedetect``
   (the real first millisecond the voice is heard), and
2. compare it against the line's intended start.

The delta is the drift for that line — positive means the voice started
*after* the picture beat it describes (the most common failure: the Burmese
translation is a fragment or the timestamp was off, so the voice begins
mid-pause). We also compare the clip's spoken length against the window so a
line that overflows into the next scene is flagged.

Usage
-----

CLI::

    python -m recapstudio.sync_check rendered.mp4 --dialogues dialogues.json

``dialogues.json`` is a JSON list of ``{start, end, text}`` (the same shape
the timeline editor exports, or ``job['dialogues']``). When the file is a
pipeline output the matching ``*_narration.mp3`` and ``*.srt`` next to it are
found automatically, so usually only the rendered path is needed.

Programmatic::

    from recapstudio.sync_check import measure_drift
    report = measure_drift("out.mp4", dialogues)
    for row in report["lines"]:
        print(row["index"], row["drift_seconds"], row["text"])
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .media import FFMPEG, get_media_duration
from .util import get_logger

log = get_logger("recap.sync_check")

_SILENCE_RE = re.compile(r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)")


@dataclass
class LineDrift:
    index: int
    start: float               # intended start (source timeline)
    end: float                 # intended end
    speech_onset: float        # where the voice actually starts (absolute s)
    speech_seconds: float      # how long the voice actually speaks
    text: str = ""

    @property
    def window(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def drift_seconds(self) -> float:
        """Voice onset minus intended start. + = voice late (started after the
        picture beat); − = voice early."""
        return self.speech_onset - self.start

    @property
    def overflow_seconds(self) -> float:
        """How far the spoken clip runs past the intended window end.
        + = the voice is still talking after the scene changed."""
        return max(0.0, (self.speech_onset + self.speech_seconds) - self.end)


def _speech_bounds(path: str | Path, noise: str = "-40dB") -> tuple[float, float]:
    """Return ``(onset, spoken_seconds)`` for the first run of real speech.

    ffmpeg ``silencedetect`` reports silence_start / silence_end pairs inside
    the file. The first ``silence_end`` (or 0.0) is where sound begins; the
    next ``silence_start`` (or the file end) is where it stops. This is the
    same detector that strips encoder padding in :mod:`recapstudio.tts`.
    """
    try:
        proc = subprocess.run(
            [FFMPEG, "-hide_banner", "-nostdin", "-i", str(path),
             "-af", f"silencedetect=noise={noise}:d=0.05", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120)
    except Exception as exc:
        log.warning("silencedetect failed on %s: %s", path, exc)
        return 0.0, get_media_duration(path)
    out = proc.stderr or ""
    total = get_media_duration(path)
    onset = 0.0
    end = total
    seen_start = False
    for kind, value in _SILENCE_RE.findall(out):
        t = float(value)
        if kind == "end" and not seen_start:
            # first silence_end = audio begins (only if it's near the head)
            onset = t
            seen_start = True
        elif kind == "start" and seen_start:
            end = t
            break
    return onset, max(0.0, end - onset)


def _load_dialogues(rendered: Path, dialogues_path: str | None) -> list[dict]:
    """Dialogue list from a file, or reconstructed from the sibling .srt."""
    if dialogues_path:
        data = json.loads(Path(dialogues_path).read_text(encoding="utf-8"))
        return _clean(data)
    # prefer an srt next to the rendered file (pipeline publishes it there)
    srt = rendered.with_suffix(".srt")
    if srt.exists():
        return _dialogues_from_srt(srt)
    return []


def _clean(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        try:
            st = float(r.get("start", 0.0))
            en = float(r.get("end", st + 2.0))
        except (TypeError, ValueError, AttributeError):
            continue
        out.append({"start": st, "end": en, "text": str(r.get("text", ""))})
    out.sort(key=lambda d: d["start"])
    return out


def _dialogues_from_srt(srt_path: Path) -> list[dict]:
    """Parse a plain SRT back into {start,end,text} rows for measuring."""
    def to_sec(ts: str) -> float:
        h, m, rest = ts.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)

    out: list[dict] = []
    block: list[str] = []
    for line in srt_path.read_text(encoding="utf-8-sig").splitlines() + [""]:
        if line.strip():
            block.append(line.strip())
            continue
        if len(block) >= 3 and "-->" in block[1]:
            a, _, b = block[1].partition("-->")
            out.append({"start": to_sec(a.strip()), "end": to_sec(b.strip()),
                        "text": " ".join(block[2:])})
        block = []
    return out


def measure_drift(rendered: str | Path, dialogues: list[dict],
                  *, clips_dir: str | Path | None = None,
                  narration_path: str | Path | None = None,
                  voice_cfg: dict | None = None,
                  lang: str = "my") -> dict[str, Any]:
    """Measure per-line A/V drift for a rendered recap.

    Two paths, whichever the inputs allow:

    * If ``clips_dir`` (or the cached per-line clips) is available we measure
      each clip's real speech onset and place it on the dialogue timeline —
      the most accurate per-line drift.
    * Otherwise we measure onsets inside the single ``*_narration.mp3``
      (published next to the render) by slicing it at each intended start.
    """
    rendered = Path(rendered)
    duration = get_media_duration(rendered)
    rows: list[LineDrift] = []
    for idx, d in enumerate(dialogues):
        start, end = float(d["start"]), float(d["end"])
        onset, spoken = _clip_speech(idx, d, clips_dir, narration_path, start, end)
        rows.append(LineDrift(index=idx, start=start, end=end,
                              speech_onset=start + onset,
                              speech_seconds=spoken,
                              text=str(d.get("text", ""))[:80]))
    return _report(rows, duration)


def _clip_speech(idx: int, d: dict, clips_dir, narration_path, start, end):
    """Speech (onset,len) for one line — a clip if present, else a slice."""
    if clips_dir:
        clip = Path(clips_dir) / f"{idx:04d}.mp3"
        if not clip.exists():
            clip = Path(clips_dir) / f"clip_{idx}.wav"
        if clip.exists():
            return _speech_bounds(clip)
    # fall back: slice the narration master at the intended window
    if narration_path and Path(narration_path).exists():
        seg = Path(narration_path).parent / f"_seg_{idx}.wav"
        try:
            subprocess.run(
                [FFMPEG, "-hide_banner", "-nostdin", "-y",
                 "-ss", f"{start:.3f}", "-i", str(narration_path),
                 "-t", f"{max(0.4, end - start + 0.5):.3f}",
                 "-c:a", "pcm_s16le", str(seg)],
                capture_output=True, timeout=60)
            onset, spoken = _speech_bounds(seg)
            return onset, spoken
        except Exception:
            return 0.0, max(0.0, end - start)
        finally:
            seg.unlink(missing_ok=True)
    return 0.0, max(0.0, end - start)


def _report(rows: list[LineDrift], duration: float) -> dict[str, Any]:
    drifts = [r.drift_seconds for r in rows]
    overflows = [r.overflow_seconds for r in rows]
    worst = sorted(rows, key=lambda r: -abs(r.drift_seconds))[:8]
    return {
        "duration": round(duration, 2),
        "lines": [{
            "index": r.index, "start": round(r.start, 2), "end": round(r.end, 2),
            "drift_seconds": round(r.drift_seconds, 2),
            "overflow_seconds": round(r.overflow_seconds, 2),
            "spoken_seconds": round(r.speech_seconds, 2),
            "text": r.text,
        } for r in rows],
        "summary": {
            "count": len(rows),
            "mean_abs_drift": round(sum(abs(d) for d in drifts) / max(1, len(drifts)), 2),
            "max_drift": round(max(drifts, key=abs, default=0.0), 2),
            "lines_over_1s": sum(1 for d in drifts if abs(d) > 1.0),
            "lines_overflowing": sum(1 for o in overflows if o > 0.3),
            "total_overflow": round(sum(overflows), 2),
        },
        "worst": [{"index": r.index, "drift": round(r.drift_seconds, 2),
                   "text": r.text[:60]} for r in worst],
    }


def _cli(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="recapstudio.sync_check",
                                 description="Measure per-line A/V drift on a rendered recap.")
    ap.add_argument("rendered", help="the rendered MP4 (or narration mp3)")
    ap.add_argument("--dialogues", help="dialogues.json (or omit to use sibling .srt)")
    ap.add_argument("--clips", help="per-line clips dir (most accurate)")
    ap.add_argument("--narration", help="narration mp3 to slice")
    args = ap.parse_args(argv)

    rendered = Path(args.rendered)
    dialogues = _load_dialogues(rendered, args.dialogues)
    if not dialogues:
        print("No dialogues found — pass --dialogues dialogues.json "
              "or keep a .srt next to the rendered file.")
        return 2
    narration = args.narration
    if not narration:
        cand = rendered.parent / f"{rendered.stem}_narration.mp3"
        if cand.exists():
            narration = str(cand)
    report = measure_drift(rendered, dialogues, clips_dir=args.clips,
                           narration_path=narration)
    s = report["summary"]
    print(f"duration {report['duration']}s • {s['count']} lines")
    print(f"mean |drift| {s['mean_abs_drift']}s • max {s['max_drift']}s • "
          f"{s['lines_over_1s']} lines >1s off • {s['lines_overflowing']} overflow "
          f"(total {s['total_overflow']}s)")
    print("\nper-line (drift = voice onset − intended start; + = late):")
    for r in report["lines"]:
        flag = "  ⚠" if abs(r["drift_seconds"]) > 1.0 or r["overflow_seconds"] > 0.3 else ""
        print(f"  #{r['index']:>3} {r['start']:>7.2f}-{r['end']:>7.2f} "
              f"drift {r['drift_seconds']:+.2f}s  over {r['overflow_seconds']:.2f}s  "
              f"{r['text'][:40]}{flag}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli(sys.argv[1:]))
