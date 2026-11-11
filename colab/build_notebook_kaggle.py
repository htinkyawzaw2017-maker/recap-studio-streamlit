"""Generate recap-dubbing-kaggle.ipynb — Kaggle twin of the Colab notebook.

Same pipeline (colab/recap_pipe.py, downloaded from GitHub at setup), only the
host-specific glue differs from Colab:

- GPU/Internet are toggles in Kaggle's Settings panel (right side), not a
  runtime-type menu.
- Secrets: Add-ons → Secrets + UserSecretsClient (Colab uses userdata).
- Upload: "+ Add Input → Upload → New Dataset" — files land under
  /kaggle/input; Colab's files.upload() does not exist on Kaggle.
- Outputs: /kaggle/working is shown in the notebook's Output tab after save —
  Colab's files.download() does not exist on Kaggle.
- No #@param forms on Kaggle — the config cell is plain variables.

Run: python build_notebook_kaggle.py -> writes recap-dubbing-kaggle.ipynb.
Secrets (GEMINI_API_KEY, TG_BOT_TOKEN, TG_CHAT_ID) come from Kaggle Secrets
only — never hardcode keys, never paste them in chat.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PIPE_URL = ("https://raw.githubusercontent.com/htinkyawzaw2017-maker/"
            "recap-studio-streamlit/main/colab/recap_pipe.py")


def md(text):
    return {"cell_type": "markdown", "id": f"md-{abs(hash(text)) % 10**8:08d}",
            "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "id": f"code-{abs(hash(text)) % 10**8:08d}",
            "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md('''# 🎬 Recap Dubbing — Kaggle version (v1.1)

**Flow:** Video ingest → Whisper Turbo GPU transcription → Gemini translates the *text only* into natural spoken Myanmar → Edge-TTS → Pydub timeline mix → GPU render → 9:16 Shorts + Telegram send.

Same engine as the Colab notebook, tuned for **Kaggle Notebooks** (30 GPU-hours/week, T4 x2 or P100).

## How to use (no code typing needed)
1. **Settings (right panel) → Accelerator → GPU T4 x2** (or P100), and **Internet → On** (needs a phone-verified account).
2. **Add-ons → Secrets → Add a new secret**: label `GEMINI_API_KEY` (from Google AI Studio), then **Attach** it to this notebook. Optional: `TG_BOT_TOKEN` + `TG_CHAT_ID`.
3. **Video:** either paste a YouTube link in the config cell below, or click **+ Add Input → Upload → New Dataset** and upload your video file.
4. Edit the **config cell** (2nd code cell), then **Run All**.
5. When it finishes: **Save Version** — the MP4s appear in the notebook's **Output** tab for download. Kaggle keeps the Output even after the session ends (until you delete the notebook).
6. **Phone-friendly UI (optional):** run the Gradio cell at the very end — it prints a public link you can open from your phone.
'''))

cells.append(code('''# ── CONFIG — edit these lines ──────────────────────────────────────
VIDEO_SOURCE = "youtube"          # "youtube" or "upload" (dataset input)
YOUTUBE_URL = ""                  # e.g. https://www.youtube.com/watch?v=...
LANGUAGE = "auto"                 # "auto", "zh", "en", "my"
WHISPER_MODEL = "large-v3-turbo"  # "large-v3-turbo" (best), "medium", "small"
GEMINI_MODEL = "gemini-2.5-flash-lite"
VOICE = "my-MM-NilarNeural"       # or "my-MM-ThihaNeural"
FLIP = True                       # mirror the video
COLOR_TWEAK = True                # slight contrast/saturation boost
MUTE_ORIGINAL = True              # pure dub voice (False = original low under dub)
MAKE_SHORTS_9x16 = True
SEND_TELEGRAM = False
OUTPUT_NAME = "recap_dubbed"
# "auto" means Whisper detects the language itself.
LANGUAGE = None if LANGUAGE == "auto" else LANGUAGE
print("source:", VIDEO_SOURCE, "| voice:", VOICE, "| shorts:", MAKE_SHORTS_9x16)
'''))

cells.append(code('''import subprocess, sys

print("=== install ===")
subprocess.run(["pip", "install", "-q", "faster-whisper", "google-genai",
                "edge-tts", "pydub", "yt-dlp", "requests", "gradio"], check=True)
# PyAV 19 removed the metadata_errors kwarg that faster-whisper 1.2.1 passes;
# pin av<19 so transcribe never hits that TypeError. (recap_pipe.py also
# strips the kwarg at runtime as a second line of defence.)
subprocess.run(["pip", "install", "-q", "av<19", "faster-whisper"], check=True)
import av
print("av version:", av.__version__)
print("=== pipeline module ===")
subprocess.run(["wget", "-q", PIPE_URL_PLACEHOLDER,
                "-O", "/kaggle/working/recap_pipe.py"], check=True)
sys.path.insert(0, "/kaggle/working")
import recap_pipe as rp
print("=== GPU ===")
# No nvidia-smi on CPU accelerators — never let this check crash the setup.
import shutil
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi", "-L"])
else:
    print("no nvidia-smi — Settings → Accelerator is still Off/CPU?")
import torch
print("torch cuda available:", torch.cuda.is_available())
print("video encoder:", rp.pick_encoder(),
      "(NVENC falls back to libx264 automatically if no GPU)")
'''.replace("PIPE_URL_PLACEHOLDER", PIPE_URL)))

cells.append(code('''import glob, json, os

WORK = "/kaggle/working/work"
os.makedirs(WORK, exist_ok=True)
SRC = None
if VIDEO_SOURCE == "upload":
    # Kaggle uploads arrive as an input dataset: + Add Input → Upload →
    # New Dataset, files land somewhere under /kaggle/input.
    exts = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v")
    cands = [p for p in glob.glob("/kaggle/input/**/*", recursive=True)
             if p.lower().endswith(exts)]
    assert cands, (
        "No video in /kaggle/input — click **+ Add Input → Upload → "
        "New Dataset**, upload your video, then re-run this cell. "
        "(Or set VIDEO_SOURCE='youtube' and paste a URL.)")
    SRC = cands[0]
    if len(cands) > 1:
        print(f"{len(cands)} videos found — using the first: {SRC}")
else:
    assert YOUTUBE_URL.startswith("http"), (
        "Paste a real YouTube link in the config cell, or switch "
        "VIDEO_SOURCE to 'upload' after adding your video as an input.")
    SRC = rp.download_youtube(YOUTUBE_URL, WORK)
DURATION = rp.probe_duration(SRC)
print(f"video: {SRC}  duration: {DURATION:.1f}s")
'''))

cells.append(code('''AUDIO = rp.extract_audio(SRC, WORK)
print("audio ready:", AUDIO)
'''))

cells.append(code('''TRANSCRIPT, detected, device = rp.transcribe(AUDIO, WHISPER_MODEL, LANGUAGE)
json.dump(TRANSCRIPT, open(os.path.join(WORK, "transcript.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"whisper ({device}, detected {detected}): {len(TRANSCRIPT)} segments")
for s in TRANSCRIPT[:5]:
    print(f"{s['start']:>7.2f}  {s['text']}")
'''))

cells.append(code('''from kaggle_secrets import UserSecretsClient

sec = UserSecretsClient()
try:
    GEMINI_API_KEY = sec.get_value("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None
assert GEMINI_API_KEY, (
    "Add GEMINI_API_KEY in Add-ons → Secrets and ATTACH it to this "
    "notebook — never paste keys in code or chat.")
DUB = rp.translate(TRANSCRIPT, GEMINI_API_KEY, GEMINI_MODEL)
json.dump(DUB, open(os.path.join(WORK, "dub_script.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"dub lines: {len(DUB)}")
for d in DUB[:5]:
    print(f"{d['start']:>7.2f}  {d['text']}")
'''))

cells.append(code('''CLIPS = os.path.join(WORK, "clips")
failed = rp.synthesize(DUB, VOICE, CLIPS)
print(f"clips ok: {len(DUB) - len(failed)} / {len(DUB)}")
for f in failed:
    print("FAILED:", f)
# A muted output almost always means every TTS clip failed (voice error /
# rate-limit) and the mix went silent. Stop here instead of rendering a
# muted video.
assert not failed, \\
    f"{len(failed)}/{len(DUB)} TTS clips failed for voice {VOICE} — " \\
    "Burmese voiceover would be MUTE. Try the other voice or re-run this cell."
'''))

cells.append(code('''DUB_TRACK, SRT_PATH, n_srt = rp.mix_and_srt(
    DUB, CLIPS, DURATION, SRC, MUTE_ORIGINAL, WORK)
print("dub track:", DUB_TRACK, "| srt lines:", n_srt)
'''))

cells.append(code('''OUT, venc = rp.render_video(SRC, DUB_TRACK, DURATION,
                              os.path.join(WORK, OUTPUT_NAME + ".mp4"),
                              FLIP, COLOR_TWEAK)
print("done:", OUT, f"({os.path.getsize(OUT) / 1e6:.1f} MB, encoder {venc})")
'''))

cells.append(code('''if MAKE_SHORTS_9x16:
    SHORTS = rp.make_shorts(SRC, DUB_TRACK, DURATION,
                            os.path.join(WORK, OUTPUT_NAME + "_9x16.mp4"))
    print("shorts:", SHORTS, f"({os.path.getsize(SHORTS) / 1e6:.1f} MB)")
else:
    SHORTS = None
    print("9:16 skipped")
'''))

cells.append(code('''# Kaggle has no files.download(): /kaggle/working is the notebook's
# permanent Output. Save a Version — then use the Output tab to download.
# On Colab this cell used files.download(); here we just list the files.
files_out = [OUT] + ([SHORTS] if SHORTS else []) + \\
            [os.path.join(WORK, "dub_script.json"), os.path.join(WORK, "dub.srt")]
print("=== outputs (Output tab after Save Version) ===")
for f in files_out:
    if f and os.path.exists(f):
        print(f"  {f}  ({os.path.getsize(f) / 1e6:.1f} MB)")
'''))

cells.append(code('''if SEND_TELEGRAM:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient()
    try:
        tok = sec.get_value("TG_BOT_TOKEN")
        chat = sec.get_value("TG_CHAT_ID")
    except Exception:
        tok = chat = None
    assert tok and chat, "Add TG_BOT_TOKEN and TG_CHAT_ID in Kaggle Secrets first."
    import requests
    for f in [OUT] + ([SHORTS] if SHORTS else []):
        if f and os.path.exists(f) and os.path.getsize(f) < 50 * 1024 * 1024:
            with open(f, "rb") as fh:
                r = requests.post(f"https://api.telegram.org/bot{tok}/sendVideo",
                                  data={"chat_id": chat}, files={"video": fh},
                                  timeout=300)
            print(f, r.status_code, r.text[:200])
        else:
            print("skip (>50MB Bot API limit or missing):", f)
else:
    print("Telegram send off (SEND_TELEGRAM=False in the config)")
'''))

cells.append(md('''## 📱 Phone-friendly UI (optional — run this cell only if you want a link)
This cell starts a small web UI and prints a public link. Open that link from your phone: upload a video (or paste a YouTube URL), pick a voice, press **Run dubbing**, and download the finished videos from the page. Notes:
- The cell **keeps running** while the UI is up — that is normal. Stop the cell when you are done.
- The link dies when the session ends — run the cell again for a fresh link.
- Kaggle sessions are capped (~12 h, GPU quota 30 h/week) — download outputs before that.
'''))

cells.append(code('''import gradio as gr
import os, subprocess, sys
# Self-heal: fresh sessions wipe /kaggle/working? No — it persists, but this
# cell may run before setup in a fresh copy; fetch the module if missing.
if not os.path.exists("/kaggle/working/recap_pipe.py"):
    print("recap_pipe.py missing — downloading it now...")
    subprocess.run(["wget", "-q", PIPE_URL_PLACEHOLDER,
                    "-O", "/kaggle/working/recap_pipe.py"], check=True)
sys.path.insert(0, "/kaggle/working")
import recap_pipe as rp
from kaggle_secrets import UserSecretsClient

def ui_run(video_file, youtube_url, voice, wmodel, make_shorts):
    try:
        api_key = UserSecretsClient().get_value("GEMINI_API_KEY")
    except Exception:
        api_key = None
    main, shorts, log = rp.run_all(
        video_file=video_file, youtube_url=youtube_url, voice=voice,
        whisper_model=wmodel,
        make_shorts_flag=make_shorts, api_key=api_key,
        work="/kaggle/working/work_ui")
    return main, shorts, log

with gr.Blocks(title="Recap Dubbing") as demo:
    gr.Markdown("Upload a video OR paste a YouTube URL, pick a voice + Whisper model, press Run dubbing.")
    video_file = gr.File(label="Video file (leave empty if using a URL)")
    youtube_url = gr.Textbox(label="YouTube URL (optional)")
    voice = gr.Dropdown(["my-MM-NilarNeural", "my-MM-ThihaNeural"],
                        value="my-MM-NilarNeural", label="Voice")
    wmodel = gr.Dropdown(["large-v3-turbo", "medium", "small"],
                         value="large-v3-turbo", label="Whisper model")
    make_shorts = gr.Checkbox(value=True, label="Also make 9:16 Shorts")
    btn = gr.Button("Run dubbing")
    out_main = gr.Video(label="Dubbed video")
    out_shorts = gr.Video(label="9:16 Shorts")
    log = gr.Textbox(label="Status")
    btn.click(ui_run, [video_file, youtube_url, voice, wmodel, make_shorts],
              [out_main, out_shorts, log])
demo.launch(share=True)
'''.replace("PIPE_URL_PLACEHOLDER", PIPE_URL)))

cells.append(md('''## 🖥️ Use this notebook as a LOCAL backend (optional)
Don't want to run the pipeline inside this notebook? Use the GPU as a private API your Mac calls directly — upload a video, get the dubbed MP4 back, all from your local terminal.

1. Run the cell below — it prints a `https://*.trycloudflare.com` URL and a token.
2. On your Mac, grab [`local_dub.py`](https://raw.githubusercontent.com/htinkyawzaw2017-maker/recap-studio-streamlit/main/colab/local_dub.py) from the repo, `pip3 install requests`, then:
   ```
   python3 local_dub.py myvideo.mp4 --server https://YOUR-URL --token YOUR_TOKEN
   ```
Keep this cell running while you send jobs. The URL is fresh each session.
'''))

cells.append(code('''import os, subprocess, sys
if not os.path.exists("/kaggle/working/gpu_server.py"):
    subprocess.run(["wget", "-q",
        "https://raw.githubusercontent.com/htinkyawzaw2017-maker/"
        "recap-studio-streamlit/main/colab/gpu_server.py",
        "-O", "/kaggle/working/gpu_server.py"], check=True)
# Change MY_TOKEN to your own secret — local_dub.py must send the same token.
MY_TOKEN = "recap-secret-123"
# GEMINI_API_KEY must be in env for the server's /dub endpoint.
from kaggle_secrets import UserSecretsClient
try:
    os.environ["GEMINI_API_KEY"] = UserSecretsClient().get_value("GEMINI_API_KEY")
except Exception:
    pass
!python /kaggle/working/gpu_server.py --token "$MY_TOKEN"
'''))

cells.append(md('''## ⚠️ Honest limits (Kaggle)
- **GPU quota: 30 hours/week.** A ~10-minute video ≈ 5–10 GPU-minutes, so this is plenty for personal use; watch the quota meter on your profile.
- **Session cap ~12 h** (9 h for some accelerators). Save Version keeps the Output files, but an interrupted run must be re-run.
- **Single voice.** True multi-voice dubbing needs speaker diarization (e.g. pyannote + a Hugging Face token) — a future upgrade, not faked here.
- **Sync is start-anchored.** Each dub clip starts at its Whisper timestamp; if a Myanmar line runs longer than its window it flows into the following gap. The Streamlit app's retime is the exact-sync solution.
- **Subtitles are an SRT sidecar** (`dub.srt`) — video players shape Myanmar correctly from it.
- **Flip/color** reduce automated matches but do **not** guarantee copyright safety.
- **Internet must be On** for yt-dlp/Edge-TTS/Gemini; your account needs phone verification to enable it.
'''))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                   "isInternetEnabled": True, "sourceType": "notebook"},
    },
    "cells": cells,
}

out = os.path.join(HERE, "recap-dubbing-kaggle.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, ensure_ascii=False, indent=1)
print(f"wrote {out} ({len(cells)} cells)")
