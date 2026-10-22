"""End-to-end orchestration: analyst → voice → subtitles → render.

Progress is reported through :class:`JobStore` on fixed stage boundaries so a
long render shows "which step, how much is left" instead of a frozen bar.
"""
from __future__ import annotations

import math
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from . import config
from .ai import extract_timeline, normalise as normalise_dialogues, repair_timeline
from .jobs import JobCancelled, JobStore, StageProgress
from .keys import key_ring
from .media import process_registry
from .media import get_media_duration, get_video_info, run_ffmpeg
from .render import export_srt, render_master
from .subtitles import build_ass
from .tts import (VOICE_CATALOG, fit_lines_to_windows, master_audio, tts_engine)
from .util import (CancelledError, ensure_disk_space, get_logger, human_bytes,
                   human_time, safe_rmtree)

log = get_logger("recap.pipeline")

STAGE_BOUNDS = {
    "prepare": (0.0, 4.0),
    "analyze": (4.0, 46.0),
    "coverage": (46.0, 58.0),
    "voice": (58.0, 82.0),
    "mix": (82.0, 86.0),
    "subtitles": (86.0, 89.0),
    "render": (89.0, 98.0),
    "finalize": (98.0, 100.0),
}

#: stages used by the "shorts splitter" jobs (progress bar labels)
SPLIT_STAGES = ["prepare", "split", "finalize"]


def _voice_config(language: str, voice_key: str) -> dict[str, str]:
    catalog = VOICE_CATALOG.get(language) or VOICE_CATALOG["my"]
    return catalog.get(voice_key) or next(iter(catalog.values()))


def _fallback_voice(language: str, voice_key: str) -> dict[str, str] | None:
    """A different voice of the same language (used when TTS fails a line)."""
    catalog = VOICE_CATALOG.get(language) or VOICE_CATALOG["my"]
    current = _voice_config(language, voice_key)
    for key, cfg in catalog.items():
        if cfg.get("voice") != current.get("voice"):
            return cfg
    return None


def _job_workdir(job_id: str) -> Path:
    path = config.TMP_DIR / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


class RecapPipeline:
    def __init__(self, store: JobStore):
        self.store = store

    # ── helpers ────────────────────────────────────────────────────────
    def _stage(self, job_id: str, stage: str, message: str = "") -> StageProgress:
        start, end = STAGE_BOUNDS.get(stage, (0.0, 100.0))
        return StageProgress(self.store, job_id, stage, start, end, message)

    def _cancel(self, job_id: str):
        return lambda: self.store.is_cancelled(job_id)

    # ── main entry ─────────────────────────────────────────────────────
    def run_recap(self, job_id: str, payload: dict[str, Any]) -> None:
        store = self.store
        job = store.get(job_id)
        if job is None:
            return
        if store.is_cancelled(job_id):
            # stopped while it was still queued (concurrency limit)
            store.update(job_id, status="cancelled", message="⏹️ အလုပ်ကို ရပ်လိုက်ပါပြီ (မစတင်မီ)")
            return
        work_dir = _job_workdir(job_id)
        try:
            store.update(job_id, status="running", stage="prepare", progress=1,
                         message="ဗီဒီယို ဖိုင် စစ်ဆေးနေပါသည်...")
            store.log(job_id, f"Job started • payload keys: {sorted(payload.keys())}")

            input_video = str(config.resolve(payload.get("input_video", "")))
            if not Path(input_video).exists():
                raise FileNotFoundError(
                    "တင်ထားသော ဗီဒီယိုဖိုင် ရှာမတွေ့ပါ (server restart ဖြစ်သွားနိုင်ပါသည်)။ "
                    "ဗီဒီယိုကို ပြန်တင်ပေးပါ။"
                )
            info = get_video_info(input_video)
            duration = info["duration"]
            if duration <= 0.5:
                raise ValueError("ဗီဒီယို ကြာချိန် ဖတ်၍ မရပါ - ဖိုင် ပျက်နေနိုင်ပါသည်။")

            quality = payload.get("quality", "balanced")
            ensure_disk_space(int(info["size"] * 2.5) + 200 * 1024 * 1024)
            store.update(job_id, duration=duration, input_video=config.rel(input_video),
                         stats={"input": {"duration": duration, "width": info["width"],
                                          "height": info["height"], "size": info["size"],
                                          "has_audio": info["has_audio"]}},
                         message=f"✅ ဗီဒီယို အဆင်သင့် ({human_time(duration)} • "
                                 f"{info['width']}x{info['height']} • {human_bytes(info['size'])})")
            store.log(job_id, f"Input OK: {duration:.2f}s {info['width']}x{info['height']} "
                              f"{human_bytes(info['size'])} audio={info['has_audio']}")

            # ── 1. AI timeline ────────────────────────────────────────
            api_key = (payload.get("api_key") or "").strip()
            fill_mode = payload.get("fill_mode", "continuous")
            analyze_prog = self._stage(job_id, "analyze")
            # A job created in the UI always uses the server-side key ring
            # (slot #1 active, #2/#3 for automatic failover).
            payload_key_ring = key_ring if not api_key else None
            if payload_key_ring and not payload_key_ring.has_key() and not config.settings.demo_mode:
                raise ValueError(
                    "Gemini API Key မထည့်ရသေးပါ။ ⚙️ Settings → API Keys တွင် Key #1 ထည့်ပါ "
                    "(Key #2/#3 ထည့်ထားပါက quota ပြည့်ချိန် အလိုအလျောက် ကူးပေးပါမည်)။"
                )
            result = extract_timeline(
                api_key=api_key,
                video_path=input_video,
                duration=duration,
                language=payload.get("lang", "my"),
                mode=payload.get("mode", "auto"),
                fill_mode=fill_mode,
                model=payload.get("model"),
                progress=analyze_prog,
                cancel=self._cancel(job_id),
                log_fn=lambda msg: store.log(job_id, msg),
                key_ring=payload_key_ring,
            )
            if fill_mode == "dialogue":
                # Hooks are recap marketing copy, not part of a faithful dub.
                result["hook_line1"] = ""
                result["hook_line2"] = ""
            dialogues = result["dialogues"]
            if not dialogues:
                raise ValueError(
                    "AI မှ စကားပြောခန်း ရှာမတွေ့ပါ။ ဗီဒီယိုတွင် စကားပြောခန်း မပါဝင်နိုင်ပါ "
                    "သို့မဟုတ် AI model ကို ပြောင်း၍ ပြန်စမ်းပါ။"
                )
            coverage = result.get("coverage", {})
            store.update(job_id, stage="coverage", progress=STAGE_BOUNDS["coverage"][1],
                         dialogues=dialogues,
                         hook_line1=result.get("hook_line1", ""),
                         hook_line2=result.get("hook_line2", ""),
                         coverage=coverage,
                         message=f"🧠 စကားပြောခန်း {len(dialogues)} ခု တွေ့ပါသည် "
                                 f"(coverage {coverage.get('coverage_percent', 0)}%)")
            store.log(job_id, f"AI timeline: {len(dialogues)} lines, "
                              f"coverage={coverage.get('coverage_percent')}%")
            self.store.raise_if_cancelled(job_id)

            # ── 1b. narration length budget ───────────────────────────
            # A line that needs 9 seconds to read but only owns a 4 second
            # window used to be rescued by speeding the voice up, which is
            # what made the narration sound rushed (#2). Trim it to what can
            # be spoken calmly — the subtitle then shows exactly what is said.
            dialogues, condensed = fit_lines_to_windows(
                dialogues, duration, payload.get("lang", "my"),
                respect_end=(fill_mode == "dialogue"))
            if condensed:
                store.log(job_id, f"✂️ စာကြောင်း {condensed} ကြောင်းကို အချိန်ကိုက် တိုအောင် ချုံ့လိုက်သည် "
                                  f"(အသံ မြန်လွန်းခြင်း မဖြစ်စေရန်)")
                store.update(job_id, dialogues=dialogues,
                             stats={**store.get(job_id).stats,
                                    "script": {"condensed_lines": condensed,
                                               "total_lines": len(dialogues)}})

            # ── 2. voice ──────────────────────────────────────────────
            voice_cfg = _voice_config(payload.get("lang", "my"), payload.get("voice", "thiha"))
            alt_voice_cfg = _fallback_voice(payload.get("lang", "my"), payload.get("voice", "thiha"))
            store.update(job_id, stage="voice", message="🎙️ အသံ သွင်းနေပါသည်...")
            voice_prog = self._stage(job_id, "voice")

            def voice_cb(pct: float, message: str) -> None:
                voice_prog(pct, message)

            mix = tts_engine.build_narration(
                dialogues=dialogues,
                voice_cfg=voice_cfg,
                duration=duration,
                work_dir=work_dir,
                tag=job_id,
                progress=voice_cb,
                cancel=self._cancel(job_id),
                fallback_voice_cfg=alt_voice_cfg,
                strict_timing=(fill_mode == "dialogue"),
            )
            # ── silence repair: the video must be voiced from start to end ─
            dialogues, mix = self._repair_silences(
                job_id=job_id, dialogues=dialogues, mix=mix, duration=duration,
                work_dir=work_dir, voice_cfg=voice_cfg, alt_voice_cfg=alt_voice_cfg,
                fill_mode=fill_mode, payload=payload, voice_prog=voice_prog,
            )
            voice_stats = mix.stats()
            store.update(job_id, dialogues=dialogues,
                         stats={**store.get(job_id).stats, "voice": voice_stats},
                         message=f"🎧 အသံ {voice_stats['lines']} လိုင်း ပြီးပါပြီ "
                                 f"(တိတ်ဆိတ်ချိန် {voice_stats['silent_seconds']}s)")
            for warning in mix.warnings:
                store.log(job_id, f"⚠️ {warning}")
            store.update(job_id, stats={**store.get(job_id).stats,
                                        "voice": mix.stats()})
            self.store.raise_if_cancelled(job_id)

            # ── 3. mix/master audio ───────────────────────────────────
            store.update(job_id, stage="mix", message="🧵 အသံ timeline တိကျစွာ ချိန်နေပါသည်...")
            narration_mp3 = work_dir / f"narration_{job_id}.mp3"
            master_audio(mix.audio_path, narration_mp3, duration,
                         loudnorm=config.settings.loudness_normalize,
                         bitrate=config.settings.audio_bitrate)
            narration_seconds = get_media_duration(narration_mp3)
            store.log(job_id, f"Narration mastered: {narration_seconds:.2f}s of {duration:.2f}s")
            store.update(job_id, progress=STAGE_BOUNDS["mix"][1])

            # ── 4. subtitles ──────────────────────────────────────────
            ass_path = None
            srt_path = None
            if payload.get("enable_subtitles", True):
                store.update(job_id, stage="subtitles", message="📝 စာတန်းထိုး ဖန်တီးနေပါသည်...")
                ass_path = str(work_dir / f"{job_id}.ass")
                target_w, target_h = self._target_size(payload, info)
                build_ass(
                    dialogues=dialogues, duration=duration, ass_path=ass_path,
                    hook_line1=payload.get("hook_line1", store.get(job_id).hook_line1) or "",
                    hook_line2=payload.get("hook_line2", store.get(job_id).hook_line2) or "",
                    font_size=int(payload.get("sub_font_size", 42)),
                    v_margin=int(target_h * (float(payload.get("sub_v_pos_percent", 22)) / 100.0)),
                    hex_color=payload.get("sub_color_hex", "#00F2FE"),
                    bg_style=payload.get("sub_bg_style", "Solid Box"),
                    width_percent=int(payload.get("sub_width_percent", 90)),
                    play_res=(target_w, target_h),
                    hook_seconds=float(payload.get("hook_seconds", 0)),
                    subtitle_alpha=int(payload.get("sub_alpha", 0)),
                    uppercase_hook=bool(payload.get("hook_uppercase", False)),
                )
                srt_path = str(work_dir / f"{job_id}.srt")
                export_srt(dialogues, srt_path)
                store.update(job_id, progress=STAGE_BOUNDS["subtitles"][1])

            # ── 5. render ─────────────────────────────────────────────
            store.update(job_id, stage="render", message="🎬 Final Video Render စတင်နေပါသည်...")
            render_prog = self._stage(job_id, "render")
            output_name = f"recap_{job_id}.mp4"
            # multi-user: artefacts live in the owner's output sandbox
            output_path = str(config.user_output_dir(store.owner_of(job_id)) / output_name)
            logo_path = payload.get("logo_path")
            logo_pos = None
            if logo_path and payload.get("logo_pos_x") is not None and payload.get("logo_pos_y") is not None:
                logo_pos = (float(payload["logo_pos_x"]), float(payload["logo_pos_y"]))
            render_result = render_master(
                input_video=input_video,
                narration_wav=str(narration_mp3),
                output_video=output_path,
                ass_path=ass_path,
                logo_path=str(config.resolve(logo_path)) if logo_path else None,
                logo_pos=logo_pos,
                reframe=payload.get("reframe_mode", "Smart Blur Background"),
                mute_original=bool(payload.get("mute_original", True)),
                keep_original_level=float(payload.get("original_level", 0.12)),
                narration_gain=float(payload.get("narration_gain", 1.3)),
                quality=quality,
                target_size=self._target_size(payload, info),
                duration=duration,
                progress=render_prog,
                cancel=self._cancel(job_id),
                owner=job_id,
            )

            # ── 6. finalize ───────────────────────────────────────────
            store.update(job_id, stage="finalize", message="💾 ဖိုင် သိမ်းဆည်းနေပါသည်...")
            final_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            published = self._publish_extras(job_id, output_path, ass_path, srt_path, narration_mp3)
            stats = {
                **(store.get(job_id).stats or {}),
                "output": {
                    "size": final_size,
                    "duration": render_result.get("duration", duration),
                    "burned_subtitles": render_result.get("burned_subtitles", False),
                    "fast_path": render_result.get("fast_path", False),
                },
            }
            warnings = list(render_result.get("warnings", [])) + list(mix.warnings)
            if warnings:
                stats["warnings"] = warnings
            if voice_stats.get("silent_windows"):
                stats["silent_windows"] = voice_stats["silent_windows"]
            store.update(job_id,
                         status="completed", progress=100, stage="finalize",
                         output_video=config.rel(output_path),
                         download_url=f"/api/download/{output_name}",
                         preview_url=f"/api/asset?path={config.rel(output_path)}",
                         stats=stats, **published,
                         message=f"🎉 ဗီဒီယို အောင်မြင်စွာ ပြီးပါပြီ ({human_bytes(final_size)})")
            store.log(job_id, f"DONE → {output_name} ({human_bytes(final_size)}) in "
                              f"{human_time(store.get(job_id).elapsed_seconds())}")
            safe_rmtree(work_dir)

        except (JobCancelled, CancelledError):
            store.update(job_id, status="cancelled", message="⏹️ အလုပ်ကို ရပ်တန့်လိုက်ပါပြီ")
            store.log(job_id, "cancelled by user")
            safe_rmtree(work_dir)
        except Exception as exc:  # noqa: BLE001 - surface everything to the UI
            if store.is_cancelled(job_id):
                # the user pressed stop while an ffmpeg/ffprobe call was dying
                store.update(job_id, status="cancelled", message="⏹️ အလုပ်ကို ရပ်တန့်လိုက်ပါပြီ")
                store.log(job_id, "cancelled by user")
                safe_rmtree(work_dir)
                return
            log.exception("recap job %s failed", job_id)
            message = str(exc)
            if len(message) > 900:
                message = message[:400] + " … " + message[-400:]
            store.update(job_id, status="failed", error=message,
                         message=f"❌ {message}")
            store.log(job_id, f"FAILED: {message}")
            safe_rmtree(work_dir)
        finally:
            # make sure a cancelled job never leaves an orphan encoder running
            process_registry.kill_owner(job_id)

    # ── silence repair (full coverage guarantee) ───────────────────────
    def _repair_silences(self, *, job_id: str, dialogues: list[dict], mix,
                         duration: float, work_dir: Path, voice_cfg: dict,
                         alt_voice_cfg, fill_mode: str, payload: dict,
                         voice_prog) -> tuple[list[dict], Any]:
        """Find windows where the narration ended up silent and voice them.

        TTS can drop a line (voice throttling) and the AI can leave a long
        scene unexplained - both used to leave minutes of silence in the
        middle of a "continuous" recap. For every silent window we ask the AI
        for a short line, synthesise it and re-assemble (the TTS cache makes
        the second pass cheap: only the new lines are generated).
        """
        store = self.store
        threshold = max(3.0, float(config.settings.max_narration_gap))
        if fill_mode != "continuous":
            return dialogues, mix
        rounds = max(0, int(config.settings.coverage_repair_rounds))
        for round_index in range(rounds):
            big = [g for g in mix.gaps
                   if (g["end"] - g["start"]) >= threshold and g["start"] < duration - 1.0]
            if not big:
                break
            self.store.raise_if_cancelled(job_id)
            longest = max(g["end"] - g["start"] for g in big)
            store.log(job_id, f"🔎 တိတ်ဆိတ်နေသော ကွက် {len(big)} ခု (အရှည်ဆုံး {longest:.1f}s) "
                              f"— အသံ ပြန်ဖြည့်နေပါသည် (round {round_index + 1})")
            store.update(job_id, stage="voice",
                         message=f"🔁 အသံ မပါသော ကွက် {len(big)} ခုကို ပြန်ဖြည့်နေပါသည်...")
            extra: list[dict] = []
            try:
                extra = repair_timeline(
                    video_path=str(config.resolve(payload.get("input_video", ""))),
                    duration=duration,
                    language=payload.get("lang", "my"),
                    mode=payload.get("mode", "auto"),
                    fill_mode=fill_mode,
                    model=payload.get("model") or "",
                    gaps=big,
                    progress=lambda pct, msg: voice_prog(min(40.0, pct * 0.4), msg),
                    cancel=self._cancel(job_id),
                    log_fn=lambda msg: store.log(job_id, msg),
                    key_ring=key_ring if not payload.get("api_key") else None,
                )
            except CancelledError:
                raise
            except Exception as exc:  # AI unavailable → deterministic filler
                store.log(job_id, f"⚠️ silence repair AI failed: {exc}")
            merged = normalise_dialogues(list(dialogues) + extra, duration) if extra else dialogues
            if not extra:
                # no AI: still fill with neutral connector narration so the
                # video is never silent for a minute
                filler = self._filler_lines(big, duration, payload.get("lang", "my"))
                if not filler:
                    break
                merged = normalise_dialogues(list(dialogues) + filler, duration)
            mix = tts_engine.build_narration(
                dialogues=merged, voice_cfg=voice_cfg, duration=duration,
                work_dir=work_dir, tag=f"{job_id}_r{round_index + 1}",
                progress=voice_prog, cancel=self._cancel(job_id),
                fallback_voice_cfg=alt_voice_cfg,
            )
            dialogues = merged
            stats = mix.stats()
            store.update(job_id, dialogues=dialogues,
                         stats={**store.get(job_id).stats, "voice": stats})
            if stats["max_silence_seconds"] >= threshold:
                continue
            break
        remaining = [g for g in mix.gaps if (g["end"] - g["start"]) >= threshold]
        if remaining:
            total = sum(g["end"] - g["start"] for g in remaining)
            mix.warnings.append(
                f"အသံ မပါသော ကွက် {len(remaining)} ခု ({total:.0f}s) ကျန်နေပါသည် — "
                "Timeline Editor မှ လိုင်းထည့်ပြီး ပြန် Render လုပ်နိုင်ပါသည်။"
            )
            store.log(job_id, f"⚠️ silent windows remaining: {remaining[:6]}")
        return dialogues, mix

    @staticmethod
    def _filler_lines(gaps: list[dict], duration: float, lang: str) -> list[dict]:
        from .ai import TimelineExtractor, split_gaps
        out: list[dict] = []
        # long holes are split into ~8s slots, one short line each, otherwise a
        # single sentence would only cover the first few seconds
        for idx, gap in enumerate(split_gaps(gaps)[:40]):
            text = TimelineExtractor._FALLBACK_LINES[idx % len(TimelineExtractor._FALLBACK_LINES)]
            if lang == "en":
                text = ("Meanwhile the story keeps moving forward, so stay with us.",
                        "Let's see what happens next in this scene.",
                        "Something important is about to change here.",
                        "Keep watching to see how they handle this moment.")[idx % 4]
            out.append({"start": gap["start"], "end": gap["end"], "speaker": "Recap",
                        "text": text})
        return out

    # ── shorts splitter as a background job ────────────────────────────
    def run_split(self, job_id: str, payload: dict[str, Any]) -> None:
        """Split a long video into Shorts parts without blocking the UI.

        The endpoint used to run synchronously: a 2 hour video kept the HTTP
        request open for minutes, the browser (and any proxy in front of it)
        gave up and the page looked frozen. Now it is a normal job with a
        progress bar and a working cancel button.
        """
        store = self.store
        job = store.get(job_id)
        if job is None:
            return
        try:
            store.update(job_id, status="running", stage="prepare", progress=2,
                         message="✂️ ဗီဒီယို စစ်ဆေးနေပါသည်...")
            try:
                video_path = str(config.resolve(payload.get("video_path", "")))
            except ValueError as exc:
                raise ValueError(f"ဗီဒီယိုဖိုင် လမ်းကြောင်း မမှန်ကန်ပါ: {exc}") from exc
            if not Path(video_path).exists():
                raise FileNotFoundError("ဗီဒီယိုဖိုင် ရှာမတွေ့ပါ — ပြန်တင်ပေးပါ။")
            slice_sec = max(5, int(payload.get("slice_sec", 60) or 60))
            aspect = str(payload.get("aspect", "9:16"))
            info = get_video_info(video_path)
            store.update(job_id, duration=info["duration"], input_video=config.rel(video_path),
                         message=f"✂️ {human_time(info['duration'])} ကို အပိုင်း "
                                 f"{max(1, int(-(-info['duration'] // slice_sec)))} ခု ခွဲနေပါသည်...")
            self.store.raise_if_cancelled(job_id)

            def _progress(pct: float, message: str) -> None:
                store.update(job_id, progress=5 + pct * 0.9, message=message, stage="split")

            parts = self.split_video(video_path=video_path, slice_sec=slice_sec,
                                     aspect=aspect, progress=_progress,
                                     cancel=self._cancel(job_id), owner=job_id,
                                     out_dir=config.user_output_dir(store.owner_of(job_id)))
            if not parts:
                raise RuntimeError("အပိုင်း မထွက်ပါ — ဗီဒီယိုဖိုင် ပျက်နိုင်ပါသည်။")
            store.update(job_id, status="completed", progress=100, stage="finalize",
                         stats={"split": {"parts": parts, "slice_sec": slice_sec,
                                          "aspect": aspect, "count": len(parts)}},
                         message=f"✅ အပိုင်း {len(parts)} ခု ခွဲပြီးပါပြီ")
            store.log(job_id, f"split done: {len(parts)} parts of {slice_sec}s")
        except (JobCancelled, CancelledError):
            store.update(job_id, status="cancelled", message="⏹️ ခွဲထုတ်ခြင်းကို ရပ်လိုက်ပါပြီ")
            store.log(job_id, "split cancelled by user")
        except Exception as exc:  # noqa: BLE001
            log.exception("split job %s failed", job_id)
            store.update(job_id, status="failed", error=str(exc)[:800],
                         message=f"❌ {str(exc)[:700]}")
            store.log(job_id, f"FAILED: {exc}")
        finally:
            process_registry.kill_owner(job_id)

    # ── re-render with edited timeline ─────────────────────────────────
    def run_rerender(self, job_id: str, payload: dict[str, Any]) -> None:
        store = self.store
        job = store.get(job_id)
        if job is None:
            return
        try:
            dialogues = payload.get("dialogues") or job.dialogues or []
            dialogues = [
                {"start": float(d.get("start", 0)), "end": float(d.get("end", 0)),
                 "text": str(d.get("text", "")), "speaker": str(d.get("speaker", ""))}
                for d in dialogues if str(d.get("text", "")).strip()
            ]
            if not dialogues:
                raise ValueError("Timeline တွင် စာသား မရှိပါ - အနည်းဆုံး လိုင်းတစ်ခု ထည့်ပါ။")
            fill_mode = payload.get("fill_mode", (job.request or {}).get("fill_mode", "continuous"))
            input_video = str(config.resolve(job.input_video))
            duration = get_media_duration(input_video) or job.duration
            if fill_mode == "dialogue":
                dialogues, _ = fit_lines_to_windows(
                    dialogues, duration, payload.get("lang", "my"), respect_end=True)
            info = get_video_info(input_video)
            work_dir = _job_workdir(job_id)
            ensure_disk_space(300 * 1024 * 1024)

            store.update(job_id, status="running", stage="voice", progress=STAGE_BOUNDS["voice"][0],
                         dialogues=dialogues, message="✏️ ပြင်ဆင်ထားသော Timeline ဖြင့် အသံ ပြန်သွင်းနေပါသည်...")
            voice_cfg = _voice_config(payload.get("lang", "my"), payload.get("voice", "thiha"))
            alt_voice_cfg = _fallback_voice(payload.get("lang", "my"), payload.get("voice", "thiha"))
            mix = tts_engine.build_narration(
                dialogues=dialogues, voice_cfg=voice_cfg, duration=duration,
                work_dir=work_dir, tag=f"{job_id}_v2",
                progress=self._stage(job_id, "voice"), cancel=self._cancel(job_id),
                fallback_voice_cfg=alt_voice_cfg,
                strict_timing=(fill_mode == "dialogue"),
            )
            store.update(job_id, stage="mix")
            narration_mp3 = work_dir / f"narration_{job_id}_v2.mp3"
            master_audio(mix.audio_path, narration_mp3, duration,
                         loudnorm=config.settings.loudness_normalize,
                         bitrate=config.settings.audio_bitrate)

            ass_path = None
            srt_path = None
            target_w, target_h = self._target_size(payload, info)
            if payload.get("enable_subtitles", True):
                store.update(job_id, stage="subtitles")
                ass_path = str(work_dir / f"{job_id}_v2.ass")
                build_ass(
                    dialogues=dialogues, duration=duration, ass_path=ass_path,
                    hook_line1=(payload.get("hook_line1", job.hook_line1) if fill_mode != "dialogue" else ""),
                    hook_line2=(payload.get("hook_line2", job.hook_line2) if fill_mode != "dialogue" else ""),
                    font_size=int(payload.get("sub_font_size", 42)),
                    v_margin=int(target_h * (float(payload.get("sub_v_pos_percent", 22)) / 100.0)),
                    hex_color=payload.get("sub_color_hex", "#00F2FE"),
                    bg_style=payload.get("sub_bg_style", "Solid Box"),
                    width_percent=int(payload.get("sub_width_percent", 90)),
                    play_res=(target_w, target_h),
                    hook_seconds=float(payload.get("hook_seconds", 0)),
                    uppercase_hook=bool(payload.get("hook_uppercase", False)),
                )
                srt_path = str(work_dir / f"{job_id}_v2.srt")
                export_srt(dialogues, srt_path)

            store.update(job_id, stage="render", message="🎬 ဗီဒီယို ပြန်လည် ထုတ်လုပ်နေပါသည်...")
            output_name = f"recap_{job_id}_v{int(time.time())}.mp4"
            output_path = str(config.user_output_dir(store.owner_of(job_id)) / output_name)
            logo_path = payload.get("logo_path", job.request.get("logo_path"))
            logo_pos = None
            if logo_path and payload.get("logo_pos_x") is not None and payload.get("logo_pos_y") is not None:
                logo_pos = (float(payload["logo_pos_x"]), float(payload["logo_pos_y"]))
            render_master(
                input_video=input_video, narration_wav=str(narration_mp3),
                output_video=output_path, ass_path=ass_path,
                logo_path=str(config.resolve(logo_path)) if logo_path else None,
                logo_pos=logo_pos,
                reframe=payload.get("reframe_mode", job.request.get("reframe_mode", "Smart Blur Background")),
                mute_original=bool(payload.get("mute_original", True)),
                keep_original_level=float(payload.get("original_level", 0.12)),
                narration_gain=float(payload.get("narration_gain", 1.3)),
                quality=payload.get("quality", job.request.get("quality", "balanced")),
                target_size=(target_w, target_h), duration=duration,
                progress=self._stage(job_id, "render"), cancel=self._cancel(job_id),
                owner=job_id,
            )
            published = self._publish_extras(job_id, output_path, ass_path, srt_path, narration_mp3)
            stats = {**(job.stats or {}), "voice": mix.stats(),
                     "output": {"size": os.path.getsize(output_path)}}
            store.update(job_id, status="completed", progress=100, stage="finalize",
                         output_video=config.rel(output_path),
                         download_url=f"/api/download/{output_name}",
                         preview_url=f"/api/asset?path={config.rel(output_path)}",
                         stats=stats, **published,
                         message="✨ ပြင်ဆင်ချက်များဖြင့် ပြန်ထုတ်ပြီးပါပြီ!")
            safe_rmtree(work_dir)
        except (JobCancelled, CancelledError):
            store.update(job_id, status="cancelled", message="⏹️ ရပ်တန့်လိုက်ပါပြီ")
        except Exception as exc:  # noqa: BLE001
            if store.is_cancelled(job_id):
                store.update(job_id, status="cancelled", message="⏹️ ရပ်တန့်လိုက်ပါပြီ")
                return
            log.exception("rerender %s failed", job_id)
            store.update(job_id, status="failed", error=str(exc)[:900],
                         message=f"❌ {str(exc)[:800]}")
        finally:
            process_registry.kill_owner(job_id)

    # ── assets published next to the master ────────────────────────────
    def _publish_extras(self, job_id: str, output_path: str, ass_path: Optional[str],
                        srt_path: Optional[str], narration_mp3: str) -> dict[str, Any]:
        extras: dict[str, Any] = {}
        base = Path(output_path).stem
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if srt_path and Path(srt_path).exists():
                target = out_dir / f"{base}.srt"
                shutil.copyfile(srt_path, target)
                extras["srt_url"] = f"/api/download/{target.name}"
            if ass_path and Path(ass_path).exists():
                target = out_dir / f"{base}.ass"
                shutil.copyfile(ass_path, target)
                extras["ass_url"] = f"/api/download/{target.name}"
        except Exception as exc:
            log.warning("publishing subtitle files failed: %s", exc)
        try:
            if narration_mp3 and Path(narration_mp3).exists():
                target = out_dir / f"{base}_narration.mp3"
                shutil.copyfile(narration_mp3, target)
                extras["audio_url"] = f"/api/download/{target.name}"
        except Exception as exc:
            log.warning("publishing narration mp3 failed: %s", exc)
        return extras

    @staticmethod
    def _target_size(payload: dict[str, Any], info: dict[str, Any]) -> tuple[int, int]:
        aspect = payload.get("output_aspect", "9:16")
        if aspect == "original":
            w = info.get("width") or 1280
            h = info.get("height") or 720
            # keep even dimensions and cap at 1920 for sane encode times
            scale = min(1.0, 1920 / max(w, h, 1))
            return max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
        if aspect == "1:1":
            return 1080, 1080
        if aspect == "16:9":
            return 1280, 720
        return 720, 1280

    # ── thumbnail ──────────────────────────────────────────────────────
    def generate_thumbnail(self, video_path: str, timestamp: float, hook1: str, hook2: str,
                           output_path: str, style: str = "bold", aspect: str = "16:9",
                           text_position: str = "left") -> dict:
        """Extract a real video frame and compose a high-contrast thumbnail."""
        from PIL import Image, ImageDraw
        from .fonts import get_mm_pil_font

        dimensions = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (1080, 1080)}
        width, height = dimensions.get(aspect, dimensions["16:9"])
        text_position = "right" if text_position == "right" else "left"
        frame_path = config.TMP_DIR / f"frame_{uuid.uuid4().hex[:8]}.png"
        try:
            run_ffmpeg([
                "-y", "-ss", f"{max(0.0, timestamp):.3f}", "-i", video_path,
                "-frames:v", "1",
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                       f"crop={width}:{height}",
                str(frame_path),
            ], check=True)
            if not frame_path.exists():
                raise RuntimeError("Frame ဖမ်းယူ၍ မရပါ — AI အကြံပြုထားသော timestamp ကို ပြောင်းစမ်းပါ။")

            base = Image.open(frame_path).convert("RGBA")
            # Fade a dark gradient into the text-safe side; keep the video
            # subject visible instead of covering the whole frame with a box.
            shade = Image.new("RGBA", base.size, (0, 0, 0, 0))
            shade_draw = ImageDraw.Draw(shade)
            fade_width = int(width * 0.72)
            for step in range(fade_width):
                progress = step / max(1, fade_width - 1)
                alpha = int(218 * ((1.0 - progress) ** 1.6))
                x = step if text_position == "left" else width - 1 - step
                shade_draw.line((x, 0, x, height), fill=(3, 8, 18, alpha))
            # Subtle bottom fade protects copy on busy / bright footage.
            for step in range(int(height * 0.27)):
                progress = step / max(1, int(height * 0.27) - 1)
                alpha = int(92 * progress)
                y = height - 1 - step
                shade_draw.line((0, y, width, y), fill=(3, 8, 18, alpha))
            base = Image.alpha_composite(base, shade)

            draw = ImageDraw.Draw(base)
            headline = (hook1 or "စိတ်လှုပ်ရှားစရာ ဇာတ်လမ်း").strip()
            subheadline = (hook2 or "အဆုံးထိ ကြည့်ပါ").strip()
            font_big = get_mm_pil_font(max(42, int(width * 0.076)), bold=True)
            font_small = get_mm_pil_font(max(24, int(width * 0.040)), bold=True)
            yellow = (255, 220, 55, 255)
            white = (255, 255, 255, 255)
            cyan = (0, 226, 245, 255)
            max_text_width = int(width * (0.52 if aspect == "16:9" else 0.78))

            def wrap(text: str, font, max_width: int, max_lines: int = 3) -> list[str]:
                words = (text or "").split()
                if not words:
                    return []
                lines: list[str] = []
                current = words[0]
                for word in words[1:]:
                    probe = f"{current} {word}"
                    if draw.textlength(probe, font=font) <= max_width:
                        current = probe
                    else:
                        lines.append(current)
                        current = word
                lines.append(current)
                if len(lines) > max_lines:
                    lines = lines[:max_lines]
                    tail = lines[-1]
                    while tail and draw.textlength(tail + "…", font=font) > max_width:
                        tail = tail[:-1]
                    lines[-1] = tail.rstrip() + "…"
                return lines

            lines = wrap(headline, font_big, max_text_width, 3)
            secondary_lines = wrap(subheadline, font_small, max_text_width, 2)
            margin_x = int(width * 0.065)
            anchor_x = margin_x if text_position == "left" else width - margin_x
            anchor = "la" if text_position == "left" else "ra"
            y = int(height * (0.27 if aspect == "16:9" else 0.24))
            line_height = int(font_big.size * 1.12)
            for line in lines:
                draw.text((anchor_x, y), line, font=font_big, fill=yellow, anchor=anchor,
                          stroke_width=max(3, int(width * 0.004)), stroke_fill=(0, 0, 0, 255))
                y += line_height
            y += int(height * 0.035)
            for line in secondary_lines:
                draw.text((anchor_x, y), line, font=font_small, fill=white, anchor=anchor,
                          stroke_width=max(2, int(width * 0.0025)), stroke_fill=(0, 0, 0, 255))
                y += int(font_small.size * 1.18)

            # A single accent rule creates a consistent branded hierarchy.
            bar_y = max(int(height * 0.14), y - line_height - int(height * 0.035))
            bar_x = anchor_x if text_position == "left" else anchor_x - int(width * 0.09)
            draw.rounded_rectangle(
                (bar_x, bar_y, bar_x + int(width * 0.075), bar_y + max(6, int(height * 0.012))),
                radius=max(3, int(height * 0.008)), fill=cyan)

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            base.convert("RGB").save(output_path, "JPEG", quality=95, optimize=True)
            return {"path": output_path, "size": os.path.getsize(output_path),
                    "width": width, "height": height, "aspect": aspect}
        finally:
            frame_path.unlink(missing_ok=True)

    # ── shorts splitter ────────────────────────────────────────────────
    def split_video(self, video_path: str, slice_sec: int, aspect: str,
                    progress: Optional[callable] = None,
                    cancel: Optional[callable] = None,
                    owner: str = "", out_dir: Optional[Path] = None) -> list[dict]:
        info = get_video_info(video_path)
        duration = info["duration"]
        if duration <= 0:
            raise ValueError("ဗီဒီယို ဖိုင် မမှန်ကန်ပါ")
        slice_sec = max(5, int(slice_sec))
        total = max(1, math.ceil(duration / slice_sec))
        tag = uuid.uuid4().hex[:6]
        # Part files live in the output folder so they can be downloaded
        # immediately and are cleaned up by the janitor.
        jobs: list[tuple[int, float, float]] = []
        for index in range(total):
            start = index * slice_sec
            length = min(slice_sec, duration - start)
            if length <= 0.2:
                continue
            jobs.append((index, start, length))

        results: dict[int, dict] = {}
        workers = max(1, min(4, (os.cpu_count() or 2)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._split_one, video_path, index, start, length,
                            aspect, info, tag, cancel, owner, out_dir): index
                for index, start, length in jobs
            }
            done = 0
            for future in as_completed(futures):
                index = futures[future]
                done += 1
                if cancel and cancel():
                    for pending in futures:
                        pending.cancel()
                    raise CancelledError("ခွဲထုတ်ခြင်းကို ရပ်လိုက်ပါပြီ")
                try:
                    part = future.result()
                    if part:
                        results[index] = part
                except Exception as exc:
                    log.warning("split part %s failed: %s", index, exc)
                if progress:
                    progress(done / max(1, len(jobs)) * 100,
                             f"✂️ အပိုင်း {done}/{len(jobs)} ခွဲထုတ်ပြီးပါပြီ")
        return [results[key] for key in sorted(results)]

    def _split_one(self, video_path: str, index: int, start: float, length: float,
                   aspect: str, info: dict, tag: str,
                   cancel: Optional[callable] = None,
                   owner: str = "", out_dir: Optional[Path] = None) -> Optional[dict]:
        if cancel and cancel():
            raise CancelledError("ခွဲထုတ်ခြင်းကို ရပ်လိုက်ပါပြီ")
        name = f"part_{index + 1:02d}_{tag}.mp4"
        target_dir = Path(out_dir) if out_dir else config.OUTPUT_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / name
        same_aspect = False
        if aspect == "original":
            same_aspect = True
        elif aspect == "9:16" and info["height"] > info["width"]:
            same_aspect = True
        elif aspect == "16:9" and info["width"] >= info["height"]:
            same_aspect = True

        if same_aspect and not config.settings.force_reencode:
            # stream copy: near instant, no quality loss
            run_ffmpeg([
                "-y", "-ss", f"{start:.3f}", "-i", video_path, "-t", f"{length:.3f}",
                "-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
                str(out_path),
            ], check=False, cancel=cancel, owner=owner)
        if not out_path.exists() or out_path.stat().st_size < 2048:
            if aspect == "9:16":
                vf = "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280"
            elif aspect == "1:1":
                vf = "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080"
            elif aspect == "16:9":
                vf = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2"
            else:
                vf = "null"
            run_ffmpeg([
                "-y", "-ss", f"{start:.3f}", "-i", video_path, "-t", f"{length:.3f}",
                "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart", str(out_path),
            ], check=True, cancel=cancel, owner=owner)
        if not out_path.exists() or out_path.stat().st_size < 2048:
            return None
        # width/height + the workspace-relative path let the SPA hand a part
        # straight back to the Studio ("✂ Splitter → 🎬 Studio → recap").
        try:
            part_info = get_video_info(str(out_path))
        except Exception:  # noqa: BLE001 - probing a part must never fail the split
            part_info = {}
        return {
            "part": index + 1,
            "filename": name,
            "url": f"/api/download/{name}",
            "preview_url": f"/api/asset?path={config.rel(out_path)}",
            "path": config.rel(out_path),
            "width": part_info.get("width"),
            "height": part_info.get("height"),
            "duration": round(float(part_info.get("duration")
                                    or get_media_duration(out_path)), 2),
            "size": out_path.stat().st_size,
        }
