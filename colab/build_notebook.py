"""Generate recap-dubbing-colab.ipynb v2 — the easy-to-use version.

v2 changes vs v1:
- Config is a Colab FORM (dropdowns/checkboxes/textboxes) — no code typing,
  so typos like YOUTUBE_URL="uploade" cannot happen.
- All pipeline logic lives in recap_pipe.py (same repo folder); the cells
  are thin wrappers, and the optional Gradio cell reuses the same code.
- NVENC auto-falls back to libx264 when no NVIDIA GPU is present
  (verified: on a GPU-less machine NVENC fails with "Cannot load
  libcuda.so.1" and the retry succeeds).

Run: python build_notebook.py  ->  writes recap-dubbing-colab.ipynb next to it.
Secrets (GEMINI_API_KEY, TG_BOT_TOKEN, TG_CHAT_ID) come from Colab's
Secrets panel only — never hardcode keys, never paste them in chat.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return {"cell_type": "markdown", "id": f"md-{abs(hash(text)) % 10**8:08d}",
            "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "id": f"code-{abs(hash(text)) % 10**8:08d}",
            "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md('''# 🎬 Recap Dubbing — easy version (v2.6)

**Flow:** Video ingest → Whisper Turbo GPU transcription → Gemini translates the *text only* into natural spoken Myanmar → Edge-TTS → Pydub timeline mix → GPU render (NVENC + flip + color + pure voice) → 9:16 Shorts + Telegram send.

**Why Whisper-first?** Whisper hears the actual speech and gives exact words + timestamps, so there are no missed lines, no wrong timestamps and no invented dialogue. Gemini then translates from exact text instead of guessing from video frames.

## How to use (no code typing needed)
1. **Runtime → Change runtime type → T4 GPU**, then connect.
2. Left panel 🔑 **Secrets**: add `GEMINI_API_KEY` (from Google AI Studio). Optional: `TG_BOT_TOKEN` + `TG_CHAT_ID` for Telegram sending. Never paste keys in code or chat.
3. In the **form cell** below, pick from the dropdowns/checkboxes (YouTube link OR upload — not both).
4. **Runtime → Run all**, wait, then download the mp4s from the download cell.
5. **Phone-friendly UI (optional):** run the Gradio cell at the very end — it prints a public link you can open from your phone. Keep this Colab tab open while it runs; the link dies when the session ends.
6. **Download outputs before the session ends** — Colab runtimes are temporary (or set `SAVE_TO_DRIVE` on in the form).
'''))

cells.append(code('''import subprocess, sys

print("=== install ===")
subprocess.run(["pip", "install", "-q", "faster-whisper", "google-genai",
                "edge-tts", "pydub", "yt-dlp", "requests", "gradio"], check=True)
# v2.5 fix — CORRECTS v2.1, which had it backwards: PyAV 19 (released
# 2026-09-29) REMOVED the metadata_errors argument from av.open(), while
# faster-whisper 1.2.1 still passes it — so av 19 + faster-whisper ALWAYS
# fails at transcribe with TypeError. Upgrading av made it worse. Pin av<19
# (18.x works); pip downgrades an installed av 19 automatically. A runtime
# restart is still required after this, since the old av stays loaded in
# the kernel otherwise.
subprocess.run(["pip", "install", "-q", "av<19", "faster-whisper"], check=True)
import av
print("av version:", av.__version__)
print("=== pipeline module ===")
subprocess.run(["wget", "-q",
                "https://raw.githubusercontent.com/htinkyawzaw2017-maker/recap-studio-streamlit/main/colab/recap_pipe.py",
                "-O", "/content/recap_pipe.py"], check=True)
sys.path.insert(0, "/content")
import recap_pipe as rp
print("=== GPU ===")
# v2.3: nvidia-smi does not exist on CPU-only runtimes — never let this
# check crash the whole setup. torch.cuda below is the real GPU test.
import shutil
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi", "-L"])
else:
    print("no nvidia-smi — this runtime has no NVIDIA GPU (CPU runtime?).")
import torch
print("torch cuda available:", torch.cuda.is_available())
print("video encoder:", rp.pick_encoder(), "(NVENC falls back to libx264 automatically if no GPU)")
'''))

cells.append(code('''# ── PICK FROM THE FORM — no code typing needed ─────────────────────
VIDEO_SOURCE = "youtube"  #@param ["youtube", "upload"]
YOUTUBE_URL = ""  #@param {type:"string"}
LANGUAGE = "auto"  #@param ["auto", "zh", "en", "my"]
WHISPER_MODEL = "large-v3-turbo"  #@param ["large-v3-turbo", "medium", "small"]
GEMINI_MODEL = "gemini-2.5-flash-lite"  #@param {type:"string"}
VOICE = "my-MM-NilarNeural"  #@param ["my-MM-NilarNeural", "my-MM-ThihaNeural"]
FLIP = True  #@param {type:"boolean"}
COLOR_TWEAK = True  #@param {type:"boolean"}
MUTE_ORIGINAL = True  #@param {type:"boolean"}
MAKE_SHORTS_9x16 = True  #@param {type:"boolean"}
SEND_TELEGRAM = False  #@param {type:"boolean"}
SAVE_TO_DRIVE = False  #@param {type:"boolean"}
OUTPUT_NAME = "recap_dubbed"  #@param {type:"string"}
# "auto" means Whisper detects the language itself.
LANGUAGE = None if LANGUAGE == "auto" else LANGUAGE
print("source:", VIDEO_SOURCE, "| voice:", VOICE, "| shorts:", MAKE_SHORTS_9x16)
'''))

cells.append(code('''import json, os

WORK = "/content/work"
os.makedirs(WORK, exist_ok=True)
if VIDEO_SOURCE == "upload":
    from google.colab import files
    import shutil
    up = files.upload()
    # v2.4 fix: files.upload() saves into the notebook's cwd (/content),
    # NOT into WORK. Move it into WORK, or probe_duration hits a missing
    # file and ffprobe's empty JSON surfaces as a cryptic KeyError.
    name = list(up.keys())[0]
    uploaded = os.path.join("/content", name)
    SRC = os.path.join(WORK, name)
    if os.path.abspath(uploaded) != os.path.abspath(SRC):
        shutil.move(uploaded, SRC)
else:
    assert YOUTUBE_URL and YOUTUBE_URL.startswith("http"), "Paste a real YouTube link in the form, or switch VIDEO_SOURCE to upload."
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

cells.append(code('''from google.colab import userdata

try:
    GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None
assert GEMINI_API_KEY, "Add GEMINI_API_KEY in Colab Secrets (left panel key icon) — never paste keys in code or chat."
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

cells.append(code('''from google.colab import files

for f in [OUT] + ([SHORTS] if SHORTS else []) + \\
         [os.path.join(WORK, "dub_script.json"), os.path.join(WORK, "dub.srt")]:
    if f and os.path.exists(f):
        files.download(f)
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    import shutil
    for f in [OUT] + ([SHORTS] if SHORTS else []):
        if f and os.path.exists(f):
            shutil.copy(f, "/content/drive/MyDrive/")
    print("saved to Drive")
'''))

cells.append(code('''if SEND_TELEGRAM:
    try:
        tok = userdata.get("TG_BOT_TOKEN")
        chat = userdata.get("TG_CHAT_ID")
    except Exception:
        tok = chat = None
    assert tok and chat, "Add TG_BOT_TOKEN and TG_CHAT_ID in Colab Secrets first."
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
    print("Telegram send off (SEND_TELEGRAM=False in the form)")
'''))

cells.append(md('''## 📱 Phone-friendly UI (optional — run this cell only if you want a link)
This cell starts a small web UI and prints a public link. Open that link from your phone: upload a video (or paste a YouTube URL), pick a voice, press **Run dubbing**, and download the finished videos from the page. Notes:
- The cell **keeps running** while the UI is up — that is normal. Stop the cell when you are done.
- Keep this Colab tab open while a job runs (Pro has no background execution).
- The link dies when the session ends — run the cell again for a fresh link.
'''))

cells.append(code('''import gradio as gr
from google.colab import userdata
# v2.2 self-heal: the setup cell downloads recap_pipe.py into /content, but a
# fresh/restarted session wipes /content. If this cell runs before setup,
# fetch the module here so the UI still starts (heavy deps still need setup).
import os, subprocess, sys
if not os.path.exists("/content/recap_pipe.py"):
    print("recap_pipe.py missing (fresh session?) — downloading it now...")
    subprocess.run(["wget", "-q",
                    "https://raw.githubusercontent.com/htinkyawzaw2017-maker/recap-studio-streamlit/main/colab/recap_pipe.py",
                    "-O", "/content/recap_pipe.py"], check=True)
sys.path.insert(0, "/content")
import recap_pipe as rp

def ui_run(video_file, youtube_url, voice, make_shorts):
    try:
        api_key = userdata.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
    main, shorts, log = rp.run_all(
        video_file=video_file, youtube_url=youtube_url, voice=voice,
        make_shorts_flag=make_shorts, api_key=api_key)
    return main, shorts, log

with gr.Blocks(title="Recap Dubbing") as demo:
    gr.Markdown("Upload a video OR paste a YouTube URL, pick a voice, press Run dubbing.")
    video_file = gr.File(label="Video file (leave empty if using a URL)")
    youtube_url = gr.Textbox(label="YouTube URL (optional)")
    voice = gr.Dropdown(["my-MM-NilarNeural", "my-MM-ThihaNeural"],
                        value="my-MM-NilarNeural", label="Voice")
    make_shorts = gr.Checkbox(value=True, label="Also make 9:16 Shorts")
    btn = gr.Button("Run dubbing")
    out_main = gr.Video(label="Dubbed video")
    out_shorts = gr.Video(label="9:16 Shorts")
    log = gr.Textbox(label="Status")
    btn.click(ui_run, [video_file, youtube_url, voice, make_shorts],
              [out_main, out_shorts, log])
demo.launch(share=True)
'''))

cells.append(md('''## ⚠️ Honest limits (v2)
- **Session is temporary.** Download outputs (or use Drive) before disconnect — Colab does not host anything 24/7.
- **Single voice.** True multi-voice dubbing needs speaker diarization (e.g. pyannote + a Hugging Face token) — a future upgrade, not faked here.
- **Sync is start-anchored.** Each dub clip starts at its Whisper timestamp; if a Myanmar line runs longer than its window it flows into the following gap. The Streamlit app's retime (slowing/freezing video to fit the voice) is the exact-sync solution.
- **Subtitles are an SRT sidecar** (`dub.srt`) — video players shape Myanmar correctly from it. Burned-in PNG subtitles with proper Myanmar shaping (Pillow + Raqm) live in the Streamlit app.
- **Flip/color** reduce automated matches but do **not** guarantee copyright safety.
- **Units:** the run bills the account signed into Colab. AI Pro ≈ 200 compute units/month; if one account runs low, switch accounts (units do not pool). On a machine without an NVIDIA GPU, rendering falls back to libx264 automatically (slower, same output).
'''))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "cells": cells,
}

out = os.path.join(HERE, "recap-dubbing-colab.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, ensure_ascii=False, indent=1)
print(f"wrote {out} ({len(cells)} cells)")
