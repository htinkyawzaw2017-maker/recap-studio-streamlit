"""Thread-safe task store: progress, stages, ETA, logs, cancellation.

The old implementation kept a plain dict plus a json file and rewrote it on
every progress tick from whichever thread happened to be running. Two
problems: (1) dictionaries were mutated concurrently (``RuntimeError:
dictionary changed size during iteration`` was possible while the UI polled)
and (2) the UI had no way to know *which stage* was running, so a long render
looked stuck. Here every mutation goes through a lock, and progress is broken
into named stages with per-stage start/end timestamps (used to compute ETA).
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import config
from .util import CancelledError, atomic_write_json, get_logger, human_time, read_json

log = get_logger("recap.jobs")

STAGE_DEFS: list[tuple[str, str]] = [
    ("prepare", "ဗီဒီယို စစ်ဆေးခြင်း"),
    ("analyze", "AI ဖြင့် စကားပြောခန်း ခွဲခြမ်းစိတ်ဖြာခြင်း"),
    ("coverage", "အစအဆုံး လွတ်နေသော နေရာများ ဖြည့်စွက်ခြင်း"),
    ("voice", "အသံ သွင်းခြင်း (TTS)"),
    ("mix", "Timeline အသံ ပေါင်းစပ်ခြင်း"),
    ("subtitles", "စာတန်းထိုး ဖန်တီးခြင်း"),
    ("render", "Final Video Render လုပ်ခြင်း"),
    ("finalize", "ဖိုင် သိမ်းဆည်းခြင်း"),
]


@dataclass
class Job:
    id: str
    kind: str = "recap"
    #: owning account (multi-user mode). "shared" / "" = pre-Phase-2 install.
    user_id: str = ""
    username: str = ""
    status: str = "queued"          # queued|running|completed|failed|cancelled
    progress: float = 0.0
    message: str = "အလုပ် စတင်နေပါသည်..."
    stage: str = "prepare"
    stage_label: str = STAGE_DEFS[0][1]
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    input_video: str = ""
    output_video: str = ""
    download_url: str = ""
    preview_url: str = ""
    srt_url: str = ""
    ass_url: str = ""
    audio_url: str = ""
    duration: float = 0.0
    dialogues: list[dict[str, Any]] = field(default_factory=list)
    hook_line1: str = ""
    hook_line2: str = ""
    coverage: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    logs: list[str] = field(default_factory=list)
    stages: dict[str, str] = field(default_factory=lambda: {k: "pending" for k, _ in STAGE_DEFS})
    cancel_requested: bool = False
    request: dict[str, Any] = field(default_factory=dict)

    #: request fields that must never be written to disk or sent to a browser
    SECRET_REQUEST_FIELDS = ("api_key", "api_keys", "password", "access_key")

    def to_dict(self, include_logs: bool = True) -> dict[str, Any]:
        data = asdict(self)
        request = data.get("request")
        if isinstance(request, dict):
            data["request"] = {k: v for k, v in request.items()
                               if k not in Job.SECRET_REQUEST_FIELDS}
        data["eta_seconds"] = self.eta_seconds()
        data["elapsed_seconds"] = self.elapsed_seconds()
        # keep cancel_requested: the UI needs it to show "ရပ်နေသည်…" instead of
        # a progress bar that looks stuck while ffmpeg winds down
        data["cancelling"] = bool(self.cancel_requested) and self.status in {"queued", "running"}
        if not include_logs:
            data.pop("logs", None)
        return data

    def elapsed_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return round(end - self.started_at, 1)

    def eta_seconds(self) -> float:
        if self.status not in {"running", "queued"} or self.progress < 3:
            return 0.0
        elapsed = time.time() - (self.started_at or time.time())
        if elapsed <= 1:
            return 0.0
        total = elapsed / max(0.01, self.progress / 100.0)
        return round(max(0.0, total - elapsed), 1)


class JobStore:
    def __init__(self, directory: Path | str = config.TASK_DIR, max_logs: int = 250,
                 persist_interval: float = 1.5):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_logs = max_logs
        #: Progress ticks arrive ~3x/second; rewriting a 100 kB task file that
        #: often wasted disk I/O on EBS and slowed the whole app down.
        self.persist_interval = max(0.0, float(persist_interval))
        self._jobs: dict[str, Job] = {}
        self._last_persist: dict[str, float] = {}
        self._lock = threading.RLock()
        self._load_from_disk()

    # ── persistence ────────────────────────────────────────────────────
    def _path(self, job_id: str) -> Path:
        return self.directory / f"{job_id}.json"

    def _load_from_disk(self) -> None:
        for entry in sorted(self.directory.glob("task_*.json")):
            data = read_json(entry)
            if not isinstance(data, dict):
                continue
            try:
                job = Job(**{k: v for k, v in data.items() if k in Job.__dataclass_fields__})
                if job.status in {"running", "queued"}:
                    # The process that owned it is gone (container restart).
                    job.status = "failed"
                    job.error = job.error or "Server restart ဖြစ်သွားသဖြင့် task ရပ်သွားပါသည်။ ပြန်လည် စတင်ပေးပါ။"
                    job.message = job.error
                self._jobs[job.id] = job
            except Exception as exc:
                log.warning("skipping unreadable task file %s: %s", entry, exc)

    def _persist(self, job: Job, force: bool = False) -> None:
        now = time.time()
        last = self._last_persist.get(job.id, 0.0)
        if not force and self.persist_interval and (now - last) < self.persist_interval:
            return
        self._last_persist[job.id] = now
        atomic_write_json(self._path(job.id), job.to_dict(include_logs=False))

    # ── CRUD ───────────────────────────────────────────────────────────
    def create(self, kind: str = "recap", request: Optional[dict] = None,
               duration: float = 0.0, user_id: str = "", username: str = "") -> Job:
        job_id = f"task_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        job = Job(id=job_id, kind=kind, duration=duration, request=request or {},
                  user_id=user_id or "", username=username or "")
        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                return job
            data = read_json(self._path(job_id))
            if isinstance(data, dict):
                try:
                    job = Job(**{k: v for k, v in data.items() if k in Job.__dataclass_fields__})
                    self._jobs[job.id] = job
                    return job
                except Exception:
                    return None
        return None

    def list(self, limit: int = 40, user_id: Optional[str] = None) -> list[Job]:
        """All known jobs, newest first.

        Also picks up task files written by *another* container (an AWS ECS
        deployment behind an ALB has no sticky sessions, so a status poll can
        land on a different task - reading the shared task folder keeps the
        history and the live status consistent).
        """
        try:
            for entry in self.directory.glob("task_*.json"):
                job_id = entry.stem
                if job_id not in self._jobs:
                    self.get(job_id)
        except Exception as exc:
            log.debug("scanning task dir failed: %s", exc)
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        if user_id is not None:
            # Multi-user mode: a caller may only ever see their own jobs.
            # Legacy jobs (no owner recorded) belong to the "shared" scope.
            wanted = user_id or "shared"
            jobs = [j for j in jobs if (j.user_id or "shared") == wanted]
        return jobs[:limit]

    def owner_of(self, job_id: str) -> str:
        job = self.get(job_id)
        return (job.user_id or "shared") if job else ""

    def is_owned_by(self, job_id: str, user_id: str) -> bool:
        owner = self.owner_of(job_id)
        return bool(owner) and owner == (user_id or "shared")

    def count_active(self, user_id: Optional[str] = None) -> int:
        with self._lock:
            jobs = list(self._jobs.values())
        if user_id is not None:
            wanted = user_id or "shared"
            jobs = [j for j in jobs if (j.user_id or "shared") == wanted]
        return sum(1 for j in jobs if j.status in {"queued", "running"})

    def delete(self, job_id: str) -> bool:
        with self._lock:
            self._jobs.pop(job_id, None)
            try:
                self._path(job_id).unlink(missing_ok=True)
                return True
            except Exception:
                return False

    # ── mutation helpers ───────────────────────────────────────────────
    def update(self, job_id: str, *, progress: Optional[float] = None,
               message: Optional[str] = None, stage: Optional[str] = None,
               status: Optional[str] = None, persist: bool = True, **fields: Any) -> Optional[Job]:
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return None
            if progress is not None:
                job.progress = round(max(0.0, min(100.0, float(progress))), 1)
            if message is not None:
                job.message = message
            if stage is not None and stage in dict(STAGE_DEFS):
                job.stage = stage
                job.stage_label = dict(STAGE_DEFS)[stage]
                ordered = [key for key, _ in STAGE_DEFS]
                for key in ordered[: ordered.index(stage)]:
                    if job.stages.get(key) not in {"failed", "skipped"}:
                        job.stages[key] = "done"
                job.stages[stage] = "running"
                if message is None:
                    job.message = job.stage_label
            if status is not None:
                job.status = status
                if status == "running" and not job.started_at:
                    job.started_at = time.time()
                if status in {"completed", "failed", "cancelled"}:
                    job.finished_at = time.time()
                    if status == "completed":
                        job.stages = {k: "done" for k, _ in STAGE_DEFS}
                    elif stage in dict(STAGE_DEFS):
                        job.stages[stage] = "failed" if status == "failed" else "skipped"
            for key, value in fields.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            if persist:
                # status transitions are important enough to always hit the
                # disk (a restarted container must not resurrect a dead job)
                self._persist(job, force=status is not None or progress is None)
            return job

    def log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return
            stamp = time.strftime("%H:%M:%S")
            job.logs.append(f"[{stamp}] {message}")
            if len(job.logs) > self.max_logs:
                del job.logs[: len(job.logs) - self.max_logs]
            log.info("[%s] %s", job_id, message)

    def request_cancel(self, job_id: str, kill_processes: bool = True) -> bool:
        with self._lock:
            job = self.get(job_id)
            if job is None or job.status in {"completed", "failed", "cancelled"}:
                return False
            job.cancel_requested = True
            job.message = "⏹️ ရပ်တန့်ရန် တောင်းဆိုထားပါသည် — ffmpeg ကို ရပ်နေပါသည်..."
            self._persist(job, force=True)
        if kill_processes:
            # Cancel must be instant: an encode can run for 30+ minutes and
            # only checking a flag between stages felt like "cancel မရဘူး".
            try:
                from .media import process_registry
                killed = process_registry.kill_owner(job_id)
                if killed:
                    self.log(job_id, f"⏹️ ffmpeg process {killed} ခု ရပ်လိုက်ပါသည်")
            except Exception as exc:  # never break the API on a kill error
                log.debug("kill on cancel failed: %s", exc)
        return True

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            job = self.get(job_id)
            return bool(job and job.cancel_requested)

    def raise_if_cancelled(self, job_id: str) -> None:
        if self.is_cancelled(job_id):
            raise JobCancelled("အလုပ်ကို ရပ်တန့်လိုက်ပါပြီ။")


class JobCancelled(CancelledError):
    """Raised inside the pipeline when the user cancels a job.

    Subclasses :class:`recapstudio.util.CancelledError` so a cancellation that
    bubbles up from ffmpeg (media layer) is recognised as a *user cancel*
    rather than reported as a crash.
    """


class StageProgress:
    """Maps a 0-100 progress value of one stage onto the global bar."""

    def __init__(self, store: JobStore, job_id: str, stage: str,
                 start: float, end: float, weight_message: str = ""):
        self.store = store
        self.job_id = job_id
        self.stage = stage
        self.start = start
        self.end = end
        self.weight_message = weight_message

    def __call__(self, pct: float, message: str = "") -> None:
        self.store.raise_if_cancelled(self.job_id)
        span = self.end - self.start
        overall = self.start + span * (max(0.0, min(100.0, pct)) / 100.0)
        text = message or self.weight_message
        self.store.update(self.job_id, progress=overall, message=text or None, stage=self.stage)


def format_duration(seconds: float) -> str:
    return human_time(seconds)
