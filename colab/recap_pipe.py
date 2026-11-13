"""Shared pipeline for the Recap Dubbing Colab notebook (v2).

One logic core, two front-ends: the notebook's step-by-step cells and the
optional Gradio UI cell both call these functions, so a fix here fixes both.

Design notes
- Heavy dependencies (faster-whisper, google-genai, edge-tts, torch) are
  imported lazily inside their functions, so importing this module never
  requires a GPU, a key, or those packages.
- No secrets live here: the Gemini key is passed in by the caller (Colab
  Secrets via ``userdata.get("GEMINI_API_KEY")`` in the notebook).
- Sync is start-anchored: each dub clip starts at its Whisper timestamp.
  Exact retime (slowing/freezing video to fit the voice) lives in the
  Streamlit app, not here.
- Flip/color reduce automated matches; they do NOT guarantee copyright
  safety.
"""

import json
import os
import shutil
import subprocess

__all__ = [
    "probe_duration", "download_youtube", "extract_audio", "transcribe",
    "translate", "synthesize", "mix_and_srt", "pick_encoder",
    "render_video", "make_shorts", "run_all",
]


# ── helpers ──────────────────────────────────────────────────────────
def probe_duration(path):
    # v2.4: fail loudly instead of a cryptic KeyError when ffprobe cannot
    # read the file (missing path, corrupt/unsupported upload).
    if not os.path.exists(path):
        raise FileNotFoundError(f"video file not found: {path}")
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True).stdout
    data = json.loads(out) if out.strip() else {}
    if "format" not in data or "duration" not in data["format"]:
        raise RuntimeError(
            f"ffprobe could not read this video: {path} "
            "(file may be corrupt or in an unsupported format)")
    return float(data["format"]["duration"])


def srt_ts(sec):
    h, m = divmod(int(sec), 3600)
    m, s = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{int(round((sec - int(sec)) * 1000)):03d}"


def pick_encoder():
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True).stdout
    return "h264_nvenc" if "h264_nvenc" in enc else "libx264"


# ── 1. ingest ────────────────────────────────────────────────────────
def download_youtube(url, work, retries=3):
    """Download to work/source.<ext>, returning the real file path.

    v2.8: this used to hardcode -o source.mp4, but yt-dlp writes the merged
    container it actually got (sometimes .mkv/.webm when no mp4 is offered),
    so probe_duration then hit a missing file. Now we detect the produced
    file whatever its extension, and retry on transient yt-dlp failures.
    """
    import glob
    import time

    # clear stale outputs so we detect this run's real file
    for old in glob.glob(os.path.join(work, "source.*")):
        try:
            os.remove(old)
        except OSError:
            pass
    template = os.path.join(work, "source.%(ext)s")
    last_err = None
    for attempt in range(retries):
        try:
            subprocess.run(
                ["yt-dlp", "-f",
                 "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
                 "--merge-output-format", "mp4",
                 "-o", template, url], check=True)
            produced = [p for p in glob.glob(os.path.join(work, "source.*"))
                        if not p.endswith(".part")]
            if produced:
                return produced[0]
            last_err = RuntimeError("yt-dlp finished but produced no source.* file")
        except subprocess.CalledProcessError as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"yt-dlp download failed after {retries} tries: {last_err}")


def extract_audio(src, work):
    audio = os.path.join(work, "audio_16k.wav")
    # v2.8: a source with NO audio stream makes ffmpeg exit non-zero with a
    # cryptic message; probe for a stream first and say so plainly.
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "a", src],
        capture_output=True, text=True).stdout
    if '"codec_type": "audio"' not in probe and '"codec_type":"audio"' not in probe:
        raise RuntimeError(
            "this video has no audio track to transcribe — "
            "there is nothing to dub. Check the source.")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-vn", "-ac", "1", "-ar", "16000", audio], check=True)
    return audio


# ── 2. Whisper transcription ─────────────────────────────────────────
def transcribe(audio, model_name="large-v3-turbo", language=None):
    # v2.6 one-go fix: PyAV 19 removed the metadata_errors argument that
    # faster-whisper 1.2.1 still passes, so transcribe dies on av 19 no
    # matter what setup installed. Instead of depending on the installed
    # av version, strip the kwarg at runtime: PyAV 19 reads metadata as
    # UTF-8/surrogateescape unconditionally, so dropping it loses nothing.
    # This makes transcribe work on av 18 AND av 19, in any session, with
    # no restart and no pip change.
    try:
        import av as _av
        if int(str(_av.__version__).split(".")[0]) >= 19 and not getattr(_av, "_rp_open_patched", False):
            _orig_open = _av.open
            def _open_compat(*args, **kwargs):
                kwargs.pop("metadata_errors", None)
                return _orig_open(*args, **kwargs)
            _av.open = _open_compat
            _av._rp_open_patched = True
    except Exception:
        pass
    from faster_whisper import WhisperModel
    try:
        import torch
        want_cuda = torch.cuda.is_available()
    except Exception:
        want_cuda = False

    # v2.8 fix: a Colab runtime can report cuda.is_available()=True yet have
    # ctranslate2 fail to load libcublas.so.12 (driver/cuBLAS mismatch). When
    # that happens the whole run died at transcribe. Try CUDA first, then
    # retry once on CPU/int8 so the pipeline keeps working on any runtime.
    attempts = [("cuda", "float16")] if want_cuda else []
    attempts.append(("cpu", "int8"))

    last_err = None
    for device, compute in attempts:
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute)
            segments, info = model.transcribe(
                audio, language=language, vad_filter=True)
            transcript = [{"start": round(s.start, 2), "end": round(s.end, 2),
                           "text": s.text.strip()}
                          for s in segments if s.text.strip()]
            return transcript, getattr(info, "language", "?"), device
        except (RuntimeError, OSError) as e:
            last_err = e
            print(f"[transcribe] {device}/{compute} failed: {e} — "
                  + ("trying cpu/int8" if device == "cuda" else "no fallback left"))
    raise last_err


def _seg_key(seg):
    return round(float(seg.get("start", 0.0)), 1)


def _missing_segments(segments, clean):
    """Source segment indexes that got NO dub line back (truncation/drops)."""
    got = {_seg_key(d) for d in clean}
    return [i for i, s in enumerate(segments) if _seg_key(s) not in got]


def _target_lang_name(target_lang):
    return {
        "my": "MYANMAR", "en": "ENGLISH", "zh": "CHINESE (Mandarin)",
        "ja": "JAPANESE", "ko": "KOREAN", "es": "SPANISH", "fr": "FRENCH",
        "de": "GERMAN", "pt": "PORTUGUESE", "ru": "RUSSIAN", "hi": "HINDI",
        "ar": "ARABIC", "id": "INDONESIAN", "th": "THAI",
    }.get((target_lang or "my").lower()[:2], str(target_lang or "MYANMAR").upper())


def _translate_subset(subset, api_key, model_name, target_lang="my"):
    """Translate just the given source segments (used to re-ask the missing
    tail after a truncated response). Returns the new clean rows."""
    if not subset:
        return []
    import json as _json
    from google import genai as _g
    client = _g.Client(api_key=api_key)
    tgt = _target_lang_name(target_lang)
    prompt = (
        f"You are a professional dubbing scriptwriter. Translate the transcript below into NATURAL SPOKEN {tgt} for dubbing.\n"
        "Rules: one output line per input segment, same start/end; each line must be ONE COMPLETE sentence "
        "(never a dangling fragment); natural retelling, never word-for-word literal translation; "
        "never invent new meaning; never merge two segments into one line; keep names as spoken. "
        "Render the FULL meaning — never shorten, summarise or drop any sentence. "
        "Return ONLY a JSON array of {start, end, text} — EVERY segment gets one line.\n\n"
        + _json.dumps(subset, ensure_ascii=False))
    for _ in range(3):
        try:
            resp = client.models.generate_content(
                model=model_name, contents=prompt,
                config={"response_mime_type": "application/json"})
            raw = (getattr(resp, "text", None) or "").strip()
            if not raw:
                continue
            if raw.startswith("```"):
                raw = "\n".join(raw.splitlines()[1:-1])
            data = _json.loads(raw) if raw.startswith("[") else _json.loads(raw)
            if isinstance(data, list):
                return [d for d in data
                        if isinstance(d, dict)
                        and isinstance(d.get("start"), (int, float))
                        and str(d.get("text", "")).strip()]
        except Exception:
            continue
    return []


# ── 3. Gemini text-only translation ──────────────────────────────────
def translate(segments, api_key, model_name="gemini-2.5-flash-lite", target_lang="my"):
    from google import genai
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing — add it in Colab Secrets.")
    if not segments:
        return []
    client = genai.Client(api_key=api_key)
    tgt = _target_lang_name(target_lang)
    prompt = (
        f"You are a professional dubbing scriptwriter. Translate the transcript below into NATURAL SPOKEN {tgt} for dubbing.\n"
        "Rules: one output line per input segment, same start/end; each line must be ONE COMPLETE sentence "
        "(never a dangling fragment); natural retelling, never word-for-word literal translation; "
        "never invent new meaning; never merge two segments into one line; keep names as spoken. "
        "Render the FULL meaning — never shorten, summarise or drop any sentence or clause. "
        "Return ONLY a JSON array of {start, end, text}.\n\n"
        + json.dumps(segments, ensure_ascii=False))

    # v2.8: Gemini can return None/empty (safety block, quota) or malformed /
    # truncated JSON (max_output_tokens cut on long transcripts, stray code
    # fences) — both used to crash with an opaque AttributeError/JSONDecodeError.
    # Retry a few times, salvage a truncated trailing array element, and only
    # raise a clear error if it still can't be parsed.
    import time

    last_err = None
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=model_name, contents=prompt,
                config={"response_mime_type": "application/json"})
            raw = (getattr(resp, "text", None) or "").strip()
            if not raw:
                raise RuntimeError("empty Gemini response (safety block or quota)")
            if raw.startswith("```"):  # defensive: strip code fences if added
                raw = "\n".join(raw.splitlines()[1:-1])
            try:
                dub = json.loads(raw)
            except json.JSONDecodeError:
                # salvage: drop the last, possibly truncated object in the array
                cut = raw.rfind("},")
                if cut > 0:
                    dub = json.loads(raw[:cut + 1] + "]")
                else:
                    raise
            if not isinstance(dub, list) or not dub:
                raise RuntimeError("Gemini returned a non-list / empty dub script")
            # keep only well-formed entries so downstream never hits a bad row
            clean = [d for d in dub
                     if isinstance(d, dict)
                     and isinstance(d.get("start"), (int, float))
                     and isinstance(d.get("end"), (int, float))
                     and str(d.get("text", "")).strip()]
            if not clean:
                raise RuntimeError("Gemini dub script had no usable {start,end,text} rows")
            # v2.10: completeness check — a truncated/salvaged response can
            # silently DROP segments (usually the tail). Re-ask ONLY for the
            # missing segment indexes so every source line gets a dub line.
            missing = _missing_segments(segments, clean)
            for _retry in range(2):
                if not missing:
                    break
                subset = [segments[i] for i in missing]
                patch = _translate_subset(subset, api_key, model_name, target_lang)
                if not patch:
                    break
                clean.extend(patch)
                missing = _missing_segments(segments, clean)
            if missing:
                raise RuntimeError(
                    f"Gemini dub script is INCOMPLETE: {len(missing)} of "
                    f"{len(segments)} source segments have no translation "
                    f"(indexes {missing[:8]}...). Not rendering a partial dub.")
            return clean
        except Exception as e:  # JSONDecodeError, empty response, AND API/quota errors
            last_err = e
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Gemini translation failed after retries: {last_err}")


# ── 4. Edge-TTS ──────────────────────────────────────────────────────
def synthesize(dub, voice, clips_dir, retries=4):
    """Synthesize every dub line to an mp3 clip.

    v2.7: this used to swallow every failure into a `failed` list, so a
    voice error / rate-limit / empty clip produced NO mp3s, the mix became
    a silent WAV, and the final video had a muted audio track — the "video
    fine but Burmese voice is mute" bug.

    Root cause of the mute in the Gradio UI: run_all() runs inside
    Gradio's event loop, and asyncio.run() cannot be called from a running
    loop — it raised "RuntimeError: asyncio.run() cannot be called from a
    running event loop" for EVERY clip, so nothing was ever synthesized.
    Fixed by running the coroutine on a fresh loop in a worker thread,
    which works both in plain notebook cells and inside Gradio.

    Each line also retries with a backoff, and remaining failures are
    reported so the caller can stop loudly instead of returning a muted
    video.
    """
    import asyncio
    import threading
    import time

    import edge_tts

    os.makedirs(clips_dir, exist_ok=True)

    def run_coro(coro):
        # asyncio.run() fails inside Gradio's running loop; a private
        # loop in its own thread always works.
        result = {}

        def target():
            loop = asyncio.new_event_loop()
            try:
                result["value"] = loop.run_until_complete(coro)
            except BaseException as e:
                result["error"] = e
            finally:
                loop.close()

        t = threading.Thread(target=target)
        t.start()
        t.join()
        if "error" in result:
            raise result["error"]
        return result["value"]

    async def synth(text, out):
        await edge_tts.Communicate(text, voice).save(out)

    failed = []
    for i, d in enumerate(dub):
        text = str(d.get("text", "")).strip()
        if not text:  # edge-tts always fails on empty text — skip, don't retry
            continue
        out = os.path.join(clips_dir, f"{i:04d}.mp3")
        ok = False
        last_err = ""
        for attempt in range(retries):
            try:
                run_coro(synth(text, out))
                if os.path.exists(out) and os.path.getsize(out) >= 500:
                    ok = True
                    break
                last_err = "empty/too-small audio"
            except Exception as e:
                last_err = str(e)[:120]
            time.sleep(1.5 * (attempt + 1))  # backoff: 1.5s, 3s, 4.5s
        if not ok:
            failed.append((i, text[:40], last_err))
    return failed


# ── 5. Pydub mix + SRT ───────────────────────────────────────────────
def mix_and_srt(dub, clips_dir, duration, src, mute_original, work):
    from pydub import AudioSegment

    # v2.9 fix — OVERLAP (အသံထပ် / double-voice) bug: Myanmar lines often run
    # longer than their Whisper window, so clip N bled past clip N+1's start
    # and the two voices overlaid → muddy "double audio". Now each clip is
    # trimmed to end just before the next segment's start (with a short fade)
    # so only ONE voice ever sounds at a time → clear single narration.
    mix = AudioSegment.silent(duration=int(duration * 1000), frame_rate=24000)
    srt = []
    added = 0
    dur_ms = int(duration * 1000)
    GAP_MS = 60  # small breath between lines
    for i, d in enumerate(dub):
        p = os.path.join(clips_dir, f"{i:04d}.mp3")
        if not os.path.exists(p):
            continue
        try:
            clip = AudioSegment.from_file(p)
        except Exception as e:
            # v2.8: a corrupt/partial clip used to kill the whole mix; skip it
            print(f"[mix] clip {i} unreadable ({e}) — skipping")
            continue
        start_ms = max(0, int(d["start"] * 1000))
        # window ends right before the NEXT segment's start (or the video end)
        next_start = None
        for j in range(i + 1, len(dub)):
            if os.path.exists(os.path.join(clips_dir, f"{j:04d}.mp3")):
                next_start = int(dub[j]["start"] * 1000)
                break
        window_end = next_start if next_start is not None else dur_ms
        max_len = max(200, window_end - start_ms - GAP_MS)
        # v2.10: never cut a translated sentence unless a real overlap forces
        # it. window_end is the next line's start (or video end), so a clip
        # longer than the window would bleed INTO the next speaker's turn.
        if next_start is not None and len(clip) > max_len:
            # genuine overlap: end just before the next turn (soft fade) —
            # the only cut that keeps one voice at a time. The Streamlit app
            # handles the same case by retiming the picture instead.
            clip = clip[:max_len].fade_out(120)
            print(f"[mix] clip {i} would overlap next turn — tail faded at "
                  f"{max_len/1000:.2f}s")
        elif next_start is None and len(clip) > max_len:
            # last line overrunning the video end: keep the full sentence
            # (the muxer's -t trims at the container level, not mid-word —
            # but keep the clip whole so the SRT time matches the speech)
            print(f"[mix] clip {i} runs past video end — kept in full")
        mix = mix.overlay(clip, position=start_ms)
        srt.append(f"{i + 1}\n{srt_ts(d['start'])} --> {srt_ts(d['end'])}\n{d['text']}\n")
        added += 1
    # v2.7 safety net: never export a silent dub track (the mute-video bug).
    if added == 0 and mute_original:
        raise RuntimeError(
            "no TTS clips were mixed — the Burmese voiceover would be MUTE. "
            "Check the synthesize() output above.")
    if not mute_original:
        orig = AudioSegment.from_file(src) - 20  # original kept low under the dub
        mix = orig.overlay(mix)
    dub_track = os.path.join(work, "dub_track.wav")
    mix.export(dub_track, format="wav")
    srt_path = os.path.join(work, "dub.srt")
    with open(srt_path, "w") as fh:
        fh.write("\n".join(srt))
    return dub_track, srt_path, len(srt)


# ── 6. Render ────────────────────────────────────────────────────────
def _run_ffmpeg(cmd, venc):
    """Run ffmpeg; if NVENC fails (encoder present but no NVIDIA GPU),
    retry once with libx264 instead of erroring out."""
    try:
        subprocess.run(cmd(venc), check=True)
        return venc
    except subprocess.CalledProcessError:
        if venc == "h264_nvenc":
            subprocess.run(cmd("libx264"), check=True)
            return "libx264 (nvenc fallback)"
        raise


def render_video(src, dub_track, duration, out, flip=True, color=True):
    venc = pick_encoder()
    vf = []
    if flip:
        vf.append("hflip")
    if color:
        vf.append("eq=contrast=1.05:saturation=1.10")
    vf_str = ",".join(vf) if vf else "null"

    def cmd(v):
        return ["ffmpeg", "-y", "-i", src, "-i", dub_track,
                "-map", "0:v:0", "-map", "1:a:0",
                "-vf", vf_str, "-c:v", v, "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-t", str(duration), out]

    return out, _run_ffmpeg(cmd, venc)


def make_shorts(src, dub_track, duration, out):
    fc = ("[0:v]split[a][b];[a]crop=ih*9/16:ih,scale=1080:1920,setsar=1[fg];"
          "[b]scale=1080:1920,gblur=sigma=20[bg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v]")

    def cmd(v):
        return ["ffmpeg", "-y", "-i", src, "-i", dub_track,
                "-filter_complex", fc, "-map", "[v]", "-map", "1:a:0",
                "-c:v", v, "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-t", str(duration), out]

    _run_ffmpeg(cmd, pick_encoder())
    return out


# ── full run (used by the Gradio UI cell) ────────────────────────────
def run_all(video_file=None, youtube_url="", voice="my-MM-NilarNeural",
            make_shorts_flag=True, api_key=None,
            gemini_model="gemini-2.5-flash-lite",
            whisper_model="large-v3-turbo", language=None,
            target_lang="my",
            flip=True, color=True, mute_original=True,
            output_name="recap_dubbed", work="/content/work_ui"):
    """Run the whole pipeline. Returns (main_mp4|None, shorts_mp4|None, log).

    ``language`` is the SOURCE speech language hint for Whisper (None = auto
    detect); ``target_lang`` is the dubbing OUTPUT language (v2.10)."""
    os.makedirs(work, exist_ok=True)
    log = []
    try:
        if video_file:
            ext = os.path.splitext(video_file)[1] or ".mp4"
            src = os.path.join(work, "source" + ext)
            shutil.copy(video_file, src)
            log.append(f"video file: {video_file}")
        elif youtube_url and youtube_url.strip():
            src = download_youtube(youtube_url.strip(), work)
            log.append(f"youtube: {youtube_url.strip()}")
        else:
            return None, None, "ERROR: give a video file OR a YouTube URL."
        duration = probe_duration(src)
        log.append(f"duration: {duration:.1f}s")
        audio = extract_audio(src, work)
        transcript, detected, device = transcribe(audio, whisper_model, language)
        log.append(f"whisper ({device}, detected {detected}): {len(transcript)} segments")
        json.dump(transcript, open(os.path.join(work, "transcript.json"), "w"),
                  ensure_ascii=False, indent=1)
        if not transcript:
            raise RuntimeError(
                "Whisper found no speech in this video — there is nothing to "
                "dub. Is this a music/silent clip, or is LANGUAGE set wrong?")
        dub = translate(transcript, api_key, gemini_model, target_lang)
        json.dump(dub, open(os.path.join(work, "dub_script.json"), "w"),
                  ensure_ascii=False, indent=1)
        log.append(f"gemini dub lines: {len(dub)} ({_target_lang_name(target_lang)})")
        if not dub:
            raise RuntimeError("Gemini returned no dub lines — cannot continue.")
        # v2.10: the TTS voice must speak the OUTPUT language. A voice left
        # on the my-MM default while dubbing into zh/ja/en produced garbled
        # or silent takes — pick the language's default voice unless the
        # caller named a voice that actually matches the target language.
        def _voice_for(lang):
            defaults = {
                "my": "my-MM-NilarNeural", "en": "en-US-JennyNeural",
                "zh": "zh-CN-XiaoxiaoNeural", "ja": "ja-JP-NanamiNeural",
                "ko": "ko-KR-SunHiNeural", "es": "es-ES-ElviraNeural",
                "fr": "fr-FR-DeniseNeural", "de": "de-DE-KatjaNeural",
                "pt": "pt-BR-FranciscaNeural", "ru": "ru-RU-SvetlanaNeural",
                "hi": "hi-IN-SwaraNeural", "ar": "ar-SA-ZariyahNeural",
                "id": "id-ID-GadisNeural", "th": "th-TH-PremwadeeNeural",
            }
            return defaults.get((lang or "my").lower()[:2], defaults["my"])
        lang_prefix = (target_lang or "my").lower()[:2]
        tts_voice = voice if (voice or "").lower().startswith(lang_prefix) \
            else _voice_for(target_lang)
        if tts_voice != voice:
            log.append(f"voice {voice} does not speak "
                       f"{_target_lang_name(target_lang)} — using {tts_voice}")
        failed = synthesize(dub, tts_voice, os.path.join(work, "clips"))
        log.append(f"tts ok: {len(dub) - len(failed)} / {len(dub)}"
                   + (f" (failed: {len(failed)})" if failed else ""))
        if failed:
            for f in failed:
                log.append(f"  FAILED clip {f[0]}: {f[2]}")
            raise RuntimeError(
                f"{len(failed)}/{len(dub)} TTS clips failed for voice "
                f"{tts_voice} — the voiceover would be MUTE. "
                "Try the other voice or run again.")
        dub_track, _, n_srt = mix_and_srt(
            dub, os.path.join(work, "clips"), duration, src, mute_original, work)
        log.append(f"mix + srt: {n_srt} lines")
        main, venc = render_video(src, dub_track, duration,
                                  os.path.join(work, output_name + ".mp4"),
                                  flip, color)
        log.append(f"render ({venc}): {main} ({os.path.getsize(main) / 1e6:.1f} MB)")
        shorts = None
        if make_shorts_flag:
            shorts = make_shorts(src, dub_track, duration,
                                 os.path.join(work, output_name + "_9x16.mp4"))
            log.append(f"shorts: {shorts} ({os.path.getsize(shorts) / 1e6:.1f} MB)")
        log.append("DONE — download the files above.")
        return main, shorts, "\n".join(log)
    except Exception as e:
        log.append(f"ERROR: {type(e).__name__}: {e}")
        return None, None, "\n".join(log)
