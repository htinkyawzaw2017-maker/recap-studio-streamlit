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
def download_youtube(url, work):
    src = os.path.join(work, "source.mp4")
    subprocess.run(["yt-dlp", "-f",
                    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
                    "-o", src, url], check=True)
    return src


def extract_audio(src, work):
    audio = os.path.join(work, "audio_16k.wav")
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute)
    segments, info = model.transcribe(audio, language=language, vad_filter=True)
    transcript = [{"start": round(s.start, 2), "end": round(s.end, 2),
                   "text": s.text.strip()} for s in segments if s.text.strip()]
    return transcript, getattr(info, "language", "?"), device


# ── 3. Gemini text-only translation ──────────────────────────────────
def translate(segments, api_key, model_name="gemini-2.5-flash-lite"):
    from google import genai
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing — add it in Colab Secrets.")
    client = genai.Client(api_key=api_key)
    prompt = (
        "You are a Myanmar dubbing scriptwriter. Translate the transcript below into NATURAL SPOKEN MYANMAR for dubbing.\n"
        "Rules: one output line per input segment, same start/end; each line must be ONE COMPLETE sentence "
        "(never a dangling fragment); natural retelling, never word-for-word literal translation; "
        "never invent new meaning; never merge two segments into one line; keep names as spoken. "
        "Return ONLY a JSON array of {start, end, text}.\n\n"
        + json.dumps(segments, ensure_ascii=False))
    resp = client.models.generate_content(
        model=model_name, contents=prompt,
        config={"response_mime_type": "application/json"})
    raw = resp.text.strip()
    if raw.startswith("```"):  # defensive: strip code fences if added
        raw = "\n".join(raw.splitlines()[1:-1])
    return json.loads(raw)


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
        out = os.path.join(clips_dir, f"{i:04d}.mp3")
        ok = False
        last_err = ""
        for attempt in range(retries):
            try:
                run_coro(synth(d["text"], out))
                if os.path.exists(out) and os.path.getsize(out) >= 500:
                    ok = True
                    break
                last_err = "empty/too-small audio"
            except Exception as e:
                last_err = str(e)[:120]
            time.sleep(1.5 * (attempt + 1))  # backoff: 1.5s, 3s, 4.5s
        if not ok:
            failed.append((i, d["text"][:40], last_err))
    return failed


# ── 5. Pydub mix + SRT ───────────────────────────────────────────────
def mix_and_srt(dub, clips_dir, duration, src, mute_original, work):
    from pydub import AudioSegment

    # v2.7: match Edge-TTS output (24 kHz mono). The default 11025 Hz
    # silent bed made pydub resample every clip down to 11 kHz (muffled,
    # not mute, but poor). 24 kHz keeps the voice clean.
    mix = AudioSegment.silent(duration=int(duration * 1000), frame_rate=24000)
    srt = []
    added = 0
    for i, d in enumerate(dub):
        p = os.path.join(clips_dir, f"{i:04d}.mp3")
        if not os.path.exists(p):
            continue
        clip = AudioSegment.from_file(p)
        mix = mix.overlay(clip, position=int(d["start"] * 1000))
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
            flip=True, color=True, mute_original=True,
            output_name="recap_dubbed", work="/content/work_ui"):
    """Run the whole pipeline. Returns (main_mp4|None, shorts_mp4|None, log)."""
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
        dub = translate(transcript, api_key, gemini_model)
        json.dump(dub, open(os.path.join(work, "dub_script.json"), "w"),
                  ensure_ascii=False, indent=1)
        log.append(f"gemini dub lines: {len(dub)}")
        failed = synthesize(dub, voice, os.path.join(work, "clips"))
        log.append(f"tts ok: {len(dub) - len(failed)} / {len(dub)}"
                   + (f" (failed: {len(failed)})" if failed else ""))
        if failed:
            for f in failed:
                log.append(f"  FAILED clip {f[0]}: {f[2]}")
            raise RuntimeError(
                f"{len(failed)}/{len(dub)} TTS clips failed for voice "
                f"{voice} — the Burmese voiceover would be MUTE. "
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
