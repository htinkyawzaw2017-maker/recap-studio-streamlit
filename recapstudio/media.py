"""ffmpeg / ffprobe helpers with real progress reporting.

Why this module exists
----------------------
The previous implementation shelled out to ffmpeg with
``subprocess.run(..., stderr=PIPE)`` and produced **no progress at all**.
Because a recap render is CPU bound and can run for several minutes the UI
looked frozen ("rendering ကြာနေတယ်") even though ffmpeg was working.
Every long running command here streams ``-progress pipe:1`` (or parses
stderr) so the job store can show a live percentage, ETA and speed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from . import config
from .util import CancelledError, get_logger, human_time

log = get_logger("recap.media")

FFMPEG = os.getenv("FFMPEG_BINARY", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BINARY", "ffprobe")
ProgressFn = Optional[Callable[[float, str], None]]

_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
_SPEED_RE = re.compile(r"speed=\s*([\d.]+)x")
_OUT_TIME_RE = re.compile(r"out_time_ms=(\d+)")
_OUT_TIME_US_RE = re.compile(r"out_time_us=(\d+)")


def ffmpeg_available() -> bool:
    return shutil.which(FFMPEG) is not None


def ffprobe_available() -> bool:
    return shutil.which(FFPROBE) is not None


_FILTER_FEATURES: dict[str, bool] = {}


def ffmpeg_filter_supports(filter_name: str, option: str) -> bool:
    """Runtime capability probe (e.g. ``amix`` gained ``normalize`` only in
    ffmpeg 5.1). Keeps the render command working on older distro builds such
    as Debian 11 / Ubuntu 20.04 without hard failing."""
    key = f"{filter_name}:{option}"
    if key not in _FILTER_FEATURES:
        try:
            out = subprocess.run([FFMPEG, "-hide_banner", "-h", f"filter={filter_name}"],
                                 capture_output=True, text=True, timeout=20)
            blob = (out.stdout or "") + (out.stderr or "")
            _FILTER_FEATURES[key] = option in blob
        except Exception:
            _FILTER_FEATURES[key] = False
        log.info("ffmpeg capability %s = %s", key, _FILTER_FEATURES[key])
    return _FILTER_FEATURES[key]


def ffmpeg_version() -> str:
    try:
        out = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True, timeout=20)
        return out.stdout.splitlines()[0] if out.stdout else "unknown"
    except Exception:
        return "missing"


def _run(cmd: Sequence[str], timeout: int | None = 300, check: bool = False) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(
            list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="ignore", timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {timeout}s: {cmd[0]}") from exc
    if check and proc.returncode != 0:
        tail = (proc.stderr or "")[-1500:]
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}): {tail}")
    return proc


class ProcessRegistry:
    """Live ffmpeg processes, so a cancel can kill them immediately.

    Previously "ရပ်မည်" only set a flag that was checked between pipeline
    stages: a 40 minute render kept burning CPU and the button looked broken.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._procs: dict[int, subprocess.Popen] = {}
        self._owners: dict[int, str] = {}

    def add(self, proc: subprocess.Popen, owner: str = "") -> None:
        with self._lock:
            self._procs[proc.pid] = proc
            self._owners[proc.pid] = owner

    def remove(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.pop(proc.pid, None)
            self._owners.pop(proc.pid, None)

    def kill_owner(self, owner: str) -> int:
        """Kill every process belonging to ``owner`` (a job id). Returns count."""
        if not owner:
            return 0
        with self._lock:
            targets = [(pid, proc) for pid, proc in self._procs.items()
                       if self._owners.get(pid) == owner]
        killed = 0
        for pid, proc in targets:
            try:
                proc.kill()
                killed += 1
                log.info("killed ffmpeg pid=%s for job %s", pid, owner)
            except Exception:
                continue
        return killed

    def count(self) -> int:
        with self._lock:
            return len(self._procs)


process_registry = ProcessRegistry()


def run_ffmpeg(
    args: Iterable[str],
    *,
    total_duration: float = 0.0,
    label: str = "",
    progress: ProgressFn = None,
    timeout: int | None = None,
    check: bool = True,
    cancel: Optional[Callable[[], bool]] = None,
    owner: str = "",
) -> subprocess.CompletedProcess:
    """Run ffmpeg while streaming progress to ``progress`` (0-100, message).

    ``cancel`` is polled on every progress line (and at least every 2 s), so a
    cancelled job stops the encoder within a couple of seconds.
    ``owner`` links the process to a job id for the ProcessRegistry.
    """
    # ``-progress pipe:1`` is what makes a long encode observable: ffmpeg
    # writes machine readable key=value blocks (out_time_us / speed) to
    # stdout, which we translate into a live percentage + ETA.
    threads = config.settings.ffmpeg_threads
    thread_args = ["-threads", str(threads)] if threads and threads > 0 else []
    cmd = [FFMPEG, "-hide_banner", "-nostdin", *thread_args,
           "-progress", "pipe:1", *[str(a) for a in args]]
    started = time.time()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="ignore", bufsize=1,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg binary မတွေ့ပါ။ Docker image ထဲတွင် ffmpeg ထည့်သွင်းထားကြောင်း စစ်ဆေးပါ "
            "(packages.txt / Dockerfile)။"
        ) from exc

    process_registry.add(proc, owner)
    tail: list[str] = []
    last_emit = 0.0
    stall_seconds = max(120, int(config.settings.ffmpeg_stall_seconds))
    state = {"last_output": time.time(), "stalled": False, "cancelled": False, "stop": False}

    def _watchdog() -> None:
        """Kills a hung encoder (and answers cancel) even when ffmpeg writes
        no output at all — without this the read loop below could block
        forever and the job would look frozen at e.g. 73%."""
        while not state["stop"] and proc.poll() is None:
            time.sleep(1.0)
            if state["stop"]:
                return
            if cancel and cancel():
                state["cancelled"] = True
                try:
                    proc.kill()
                except Exception:
                    pass
                return
            if time.time() - state["last_output"] > stall_seconds:
                state["stalled"] = True
                log.error("ffmpeg produced no output for %ss - killing pid %s",
                          stall_seconds, proc.pid)
                try:
                    proc.kill()
                except Exception:
                    pass
                return

    watchdog = threading.Thread(target=_watchdog, daemon=True,
                                name=f"ffmpeg-watchdog-{proc.pid}")
    watchdog.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            state["last_output"] = time.time()
            tail.append(line)
            if len(tail) > 120:
                del tail[:60]

            if state["cancelled"] or (cancel and cancel()):
                proc.kill()
                raise CancelledError("အလုပ်ကို ရပ်တန့်လိုက်ပါပြီ")
            if state["stalled"]:
                raise RuntimeError(
                    f"FFmpeg သည် {stall_seconds}s ကြာ တုံ့ပြန်မှု မရှိပါ - ရပ်လိုက်ပါသည် "
                    "(ဖိုင် ပျက်နိုင်/disk ပြည့်နိုင်)။"
                )

            if not progress or total_duration <= 0:
                continue
            now = time.time()
            if now - last_emit < 0.4:
                continue
            seconds = None
            m = _OUT_TIME_US_RE.search(line) or _OUT_TIME_RE.search(line)
            if m:
                if _OUT_TIME_US_RE.search(line):
                    seconds = int(m.group(1)) / 1_000_000.0
                else:
                    seconds = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            if seconds is None:
                continue
            pct = max(0.0, min(100.0, seconds / total_duration * 100.0))
            speed_match = _SPEED_RE.search(line)
            speed = float(speed_match.group(1)) if speed_match else 0.0
            eta = ""
            if speed > 0.01 and seconds < total_duration:
                eta = f" • ~{human_time((total_duration - seconds) / speed)} ကျန်ပါသည်"
            last_emit = now
            msg = f"{label} {pct:.0f}%{eta}" if label else f"{pct:.0f}%{eta}"
            if progress:
                # A cancel raised inside the callback must abort the encode -
                # swallowing it here is why "ရပ်မည်" used to do nothing.
                progress(round(pct, 1), msg)
        proc.wait(timeout=timeout)
        if state["cancelled"]:
            raise CancelledError("အလုပ်ကို ရပ်တန့်လိုက်ပါပြီ")
        if state["stalled"]:
            raise RuntimeError(
                f"FFmpeg သည် {stall_seconds}s ကြာ တုံ့ပြန်မှု မရှိပါ - ရပ်လိုက်ပါသည် "
                "(ဖိုင် ပျက်နိုင်/disk ပြည့်နိုင်)။"
            )
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"ffmpeg timed out after {timeout}s")
    finally:
        state["stop"] = True
        process_registry.remove(proc)
        if proc.poll() is None:
            proc.kill()

    stderr_tail = "".join(tail)[-2500:]
    if proc.returncode != 0 and cancel and cancel():
        # ffmpeg died *because* the job was cancelled (the job-level watchdog
        # kills by owner) — report a cancel, never a crash: otherwise the
        # renderer logs "retrying without burn-in" and starts a second encode
        # for a job the user just stopped.
        raise CancelledError("အလုပ်ကို ရပ်တန့်လိုက်ပါပြီ")
    if check and proc.returncode != 0:
        log.error("ffmpeg failed (%s): %s", proc.returncode, stderr_tail)
        raise RuntimeError(f"FFmpeg error: {stderr_tail}")
    log.debug("ffmpeg finished in %.1fs (%s)", time.time() - started, label)
    return subprocess.CompletedProcess(cmd, proc.returncode, "", stderr_tail)


# ── probing ──────────────────────────────────────────────────────────────
def probe_json(path: str | os.PathLike, timeout: int = 60) -> dict:
    if not path or not Path(path).exists():
        return {}
    proc = _run(
        [FFPROBE, "-v", "error", "-show_format", "-show_streams",
         "-of", "json", str(path)],
        timeout=timeout,
    )
    try:
        return json.loads(proc.stdout or "{}")
    except Exception:
        return {}


def get_media_duration(path: str | os.PathLike) -> float:
    """Duration in seconds (0.0 when unknown/unreadable)."""
    if not path or not Path(path).exists():
        return 0.0
    info = probe_json(path)
    fmt = info.get("format") or {}
    for key in ("duration", "DURATION"):
        try:
            value = float(fmt.get(key))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    for stream in info.get("streams", []):
        try:
            value = float(stream.get("duration", 0))
            if value > 0:
                return value
        except (TypeError, ValueError):
            continue
    # Last resort: decode header only (fast) for containers without duration.
    try:
        proc = _run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=nb_frames,avg_frame_rate", "-of", "json", str(path)])
        data = json.loads(proc.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        frames = float(stream.get("nb_frames") or 0)
        rate = (stream.get("avg_frame_rate") or "0/1").split("/")
        fps = (float(rate[0]) / float(rate[1])) if len(rate) == 2 and float(rate[1]) else 0
        if frames and fps:
            return frames / fps
    except Exception:
        pass
    return 0.0


def get_video_info(path: str | os.PathLike) -> dict:
    info = probe_json(path)
    result = {
        "duration": 0.0, "width": 0, "height": 0, "fps": 0.0,
        "has_audio": False, "has_video": False, "size": 0,
        "video_codec": "", "audio_codec": "", "rotation": 0,
    }
    try:
        result["size"] = int((info.get("format") or {}).get("size") or 0)
    except Exception:
        pass
    for stream in info.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video" and not result["has_video"]:
            result["has_video"] = True
            result["width"] = int(stream.get("width") or 0)
            result["height"] = int(stream.get("height") or 0)
            result["video_codec"] = stream.get("codec_name") or ""
            rate = (stream.get("avg_frame_rate") or "0/1").split("/")
            try:
                result["fps"] = float(rate[0]) / float(rate[1]) if float(rate[1]) else 0.0
            except Exception:
                result["fps"] = 0.0
            try:
                for side in stream.get("side_data_list") or []:
                    if "rotation" in side:
                        result["rotation"] = int(float(side["rotation"]))
            except Exception:
                pass
        elif codec_type == "audio" and not result["has_audio"]:
            result["has_audio"] = True
            result["audio_codec"] = stream.get("codec_name") or ""
    if not result["duration"]:
        result["duration"] = get_media_duration(path)
    return result


def has_audio_stream(path: str | os.PathLike) -> bool:
    return get_video_info(path)["has_audio"]


def generate_test_media(path: str | os.PathLike, seconds: float = 12.0,
                        resolution: str = "1280x720", with_audio: bool = True) -> None:
    """Synthetic clip used by the offline smoke test."""
    args = [
        "-y", "-f", "lavfi", "-i", f"testsrc2=size={resolution}:rate=30:duration={seconds}",
    ]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}"]
        args += ["-map", "0:v", "-map", "1:a", "-shortest"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p"]
    if with_audio:
        args += ["-c:a", "aac", "-b:a", "128k"]
    args += [str(path)]
    run_ffmpeg(args, check=True)


def make_silence_wav(duration_sec: float, out_path: str | os.PathLike) -> str:
    """Exact length silent stereo WAV (used for timeline gaps).

    Note: ``anullsrc=...:d=`` is only supported by ffmpeg >= 5.0; the
    length is therefore set with ``-t`` which works on every version.
    """
    duration_sec = max(0.02, float(duration_sec))
    run_ffmpeg([
        "-y", "-threads", "1", "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=stereo",
        "-t", f"{duration_sec:.4f}",
        "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(out_path),
    ], check=True)
    return str(out_path)


def encode_mp3(src: str | os.PathLike, out_path: str | os.PathLike,
               bitrate: str = "192k", progress: ProgressFn = None) -> str:
    duration = get_media_duration(src)
    run_ffmpeg(["-y", "-i", str(src), "-c:a", "libmp3lame", "-b:a", bitrate, str(out_path)],
               total_duration=duration, label="MP3", progress=progress, check=True)
    return str(out_path)
