"""Generate recap-dubbing-colab.ipynb — Whisper-first Myanmar dubbing pipeline.

Run: python build_notebook.py  ->  writes recap-dubbing-colab.ipynb next to it.
The notebook is standalone (does not import the Streamlit app) so it runs
on a fresh Colab runtime: YouTube/upload -> Whisper Turbo GPU -> Gemini
text translation -> Edge-TTS -> Pydub mix -> NVENC render -> 9:16 + TG Bot.
Secrets (GEMINI_API_KEY, TG_BOT_TOKEN, TG_CHAT_ID) are read from Colab's
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

cells.append(md('''# 🎬 Recap Dubbing — Whisper-first pipeline (Colab GPU)

**Flow:** Video ingest → Whisper Turbo GPU transcription → Gemini translates the *text only* into natural spoken Myanmar → Edge-TTS → Pydub timeline mix → GPU render (NVENC + flip + color + pure voice) → 9:16 Shorts + Telegram send.

**Why Whisper-first?** Whisper hears the actual speech and gives exact words + timestamps, so there are no missed lines, no wrong timestamps and no invented dialogue. Gemini then translates from exact text instead of guessing from video frames. This is the script-first approach: full transcript → translation → timing fit.

## 🔑 Setup first (do this once)
1. Open the **Secrets panel** (🔑 icon, left sidebar) and add:
   - `GEMINI_API_KEY` — required (from Google AI Studio; one key per Google Cloud project)
   - `TG_BOT_TOKEN` and `TG_CHAT_ID` — only if you want Telegram sending
   - Never paste keys into code cells or chat.
2. **Runtime → Change runtime type → T4 GPU** (a free GPU is enough for Whisper Turbo + NVENC).
3. Run the cells top to bottom. Use the **same Google account** as your AI Pro plan if you want its 200 compute units/month — units are per account and do not pool across accounts.
4. **Download your outputs before the session ends** — Colab runtimes are temporary and files disappear on disconnect (or set `SAVE_TO_DRIVE = True`).
'''))

cells.append(code('''import subprocess

print("=== GPU ===")
subprocess.run(["nvidia-smi", "-L"])
print("=== install ===")
subprocess.run(["pip", "install", "-q", "faster-whisper", "google-genai",
                "edge-tts", "pydub", "yt-dlp", "requests"], check=True)
print("=== ffmpeg encoders (want h264_nvenc; libx264 is the fallback) ===")
out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                     capture_output=True, text=True).stdout
print("\\n".join(l for l in out.splitlines() if "h264_nvenc" in l or "libx264" in l))
import torch
print("torch cuda available:", torch.cuda.is_available())
'''))

cells.append(code('''# ── EDIT THESE ──────────────────────────────────────────────────────
VIDEO_SOURCE = "youtube"   # "youtube" or "upload"
YOUTUBE_URL  = ""          # paste a YouTube link here when VIDEO_SOURCE="youtube"
LANGUAGE     = None        # None = auto-detect; or "zh" (Chinese), "en", "my"
WHISPER_MODEL = "large-v3-turbo"  # Colab GPU: turbo is fast; CPU fallback: "small"
GEMINI_MODEL  = "gemini-2.5-flash-lite"  # text-only translation; model names drift — use a current Flash-Lite
VOICE         = "my-MM-NilarNeural"     # or "my-MM-ThihaNeural"
FLIP          = True       # mirror (reduces automated matches; NOT a copyright guarantee)
COLOR_TWEAK   = True       # slight contrast/saturation lift
MUTE_ORIGINAL = True       # True = pure dub voice only; False = original kept low under the dub
MAKE_SHORTS_9x16 = True
SEND_TELEGRAM = False
SAVE_TO_DRIVE = False
OUTPUT_NAME   = "recap_dubbed"
'''))

cells.append(code('''import json, os, subprocess

WORK = "/content/work"
os.makedirs(WORK, exist_ok=True)

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                          "-show_format", path], capture_output=True, text=True).stdout
    return float(json.loads(out)["format"]["duration"])

if VIDEO_SOURCE == "upload":
    from google.colab import files
    up = files.upload()
    SRC = os.path.join(WORK, list(up.keys())[0])
else:
    assert YOUTUBE_URL, "YOUTUBE_URL is empty — paste a link or set VIDEO_SOURCE='upload'."
    SRC = os.path.join(WORK, "source.mp4")
    subprocess.run(["yt-dlp", "-f",
                    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
                    "-o", SRC, YOUTUBE_URL], check=True)
DURATION = probe(SRC)
print(f"video: {SRC}  duration: {DURATION:.1f}s")
'''))

cells.append(code('''AUDIO = os.path.join(WORK, "audio_16k.wav")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", SRC,
                "-vn", "-ac", "1", "-ar", "16000", AUDIO], check=True)
print("audio ready:", AUDIO)
'''))

cells.append(code('''from faster_whisper import WhisperModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
compute = "float16" if device == "cuda" else "int8"
if device == "cpu":
    print("WARNING: no GPU — Whisper will be slow. Switch runtime to T4 GPU.")
model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
segments, info = model.transcribe(AUDIO, language=LANGUAGE, vad_filter=True)
TRANSCRIPT = [{"start": round(s.start, 2), "end": round(s.end, 2),
               "text": s.text.strip()} for s in segments if s.text.strip()]
json.dump(TRANSCRIPT, open(os.path.join(WORK, "transcript.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"detected: {info.language} ({info.language_probability:.2f})  segments: {len(TRANSCRIPT)}")
for s in TRANSCRIPT[:5]:
    print(f"{s['start']:>7.2f}  {s['text']}")
'''))

cells.append(code('''from google.colab import userdata
from google import genai

try:
    GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None
assert GEMINI_API_KEY, "Add GEMINI_API_KEY in Colab Secrets (left panel key icon) — never paste keys in code or chat."
client = genai.Client(api_key=GEMINI_API_KEY)
prompt = (
    "You are a Myanmar dubbing scriptwriter. Translate the transcript below into NATURAL SPOKEN MYANMAR for dubbing.\\n"
    "Rules: one output line per input segment, same start/end; each line must be ONE COMPLETE sentence "
    "(never a dangling fragment); natural retelling, never word-for-word literal translation; "
    "never invent new meaning; never merge two segments into one line; keep names as spoken. "
    "Return ONLY a JSON array of {start, end, text}.\\n\\n"
    + json.dumps(TRANSCRIPT, ensure_ascii=False))
resp = client.models.generate_content(
    model=GEMINI_MODEL, contents=prompt,
    config={"response_mime_type": "application/json"})
raw = resp.text.strip()
if raw.startswith("```"):  # defensive: strip code fences if the model adds them
    raw = "\\n".join(raw.splitlines()[1:-1])
DUB = json.loads(raw)
json.dump(DUB, open(os.path.join(WORK, "dub_script.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"dub lines: {len(DUB)}")
for d in DUB[:5]:
    print(f"{d['start']:>7.2f}  {d['text']}")
'''))

cells.append(code('''import asyncio
import edge_tts

CLIPS = os.path.join(WORK, "clips")
os.makedirs(CLIPS, exist_ok=True)

async def synth(text, out):
    await edge_tts.Communicate(text, VOICE).save(out)

failed = []
for i, d in enumerate(DUB):
    out = os.path.join(CLIPS, f"{i:04d}.mp3")
    try:
        asyncio.run(synth(d["text"], out))
        if os.path.getsize(out) < 500:
            raise RuntimeError("empty audio")
    except Exception as e:
        failed.append((i, d["text"][:40], str(e)[:80]))
print(f"clips ok: {len(DUB) - len(failed)} / {len(DUB)}")
for f in failed:
    print("FAILED:", f)
'''))

cells.append(code('''from pydub import AudioSegment

def ts(sec):
    h, m = divmod(int(sec), 3600)
    m, s = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{int(round((sec - int(sec)) * 1000)):03d}"

mix = AudioSegment.silent(duration=int(DURATION * 1000))
srt = []
for i, d in enumerate(DUB):
    p = os.path.join(CLIPS, f"{i:04d}.mp3")
    if not os.path.exists(p):
        continue
    clip = AudioSegment.from_file(p)
    mix = mix.overlay(clip, position=int(d["start"] * 1000))
    srt.append(f"{i + 1}\\n{ts(d['start'])} --> {ts(d['end'])}\\n{d['text']}\\n")
if not MUTE_ORIGINAL:
    orig = AudioSegment.from_file(SRC) - 20  # original kept low under the dub
    mix = orig.overlay(mix)
DUB_TRACK = os.path.join(WORK, "dub_track.wav")
mix.export(DUB_TRACK, format="wav")
open(os.path.join(WORK, "dub.srt"), "w").write("\\n".join(srt))
print("dub track:", DUB_TRACK, f"({len(mix) / 1000:.1f}s)  srt lines:", len(srt))
'''))

cells.append(code('''enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                        capture_output=True, text=True).stdout
VENC = "h264_nvenc" if "h264_nvenc" in enc else "libx264"
print("video encoder:", VENC)
vf = []
if FLIP:
    vf.append("hflip")
if COLOR_TWEAK:
    vf.append("eq=contrast=1.05:saturation=1.10")
vf_str = ",".join(vf) if vf else "null"
OUT = os.path.join(WORK, OUTPUT_NAME + ".mp4")
subprocess.run(["ffmpeg", "-y", "-i", SRC, "-i", DUB_TRACK,
                "-map", "0:v:0", "-map", "1:a:0",
                "-vf", vf_str, "-c:v", VENC, "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-t", str(DURATION), OUT], check=True)
print("done:", OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB")
'''))

cells.append(code('''if MAKE_SHORTS_9x16:
    SHORTS = os.path.join(WORK, OUTPUT_NAME + "_9x16.mp4")
    fc = ("[0:v]split[a][b];[a]crop=ih*9/16:ih,scale=1080:1920,setsar=1[fg];"
          "[b]scale=1080:1920,gblur=sigma=20[bg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v]")
    subprocess.run(["ffmpeg", "-y", "-i", SRC, "-i", DUB_TRACK,
                    "-filter_complex", fc, "-map", "[v]", "-map", "1:a:0",
                    "-c:v", VENC, "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-t", str(DURATION), SHORTS], check=True)
    print("shorts:", SHORTS, f"{os.path.getsize(SHORTS) / 1e6:.1f} MB")
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
    print("Telegram send off (SEND_TELEGRAM=False)")
'''))

cells.append(md('''## ⚠️ Honest limits of this notebook (v1)
- **Session is temporary.** Download outputs (or use Drive) before disconnect — Colab does not host anything 24/7.
- **Single voice.** True multi-voice dubbing needs speaker diarization (e.g. pyannote + a Hugging Face token) — a future upgrade, not faked here by assigning voices to segments.
- **Sync is start-anchored.** Each dub clip starts at its Whisper timestamp; if a Myanmar line runs longer than its window it flows into the following gap. The Streamlit app's retime (slowing/freezing video to fit the voice) is the exact-sync solution.
- **Subtitles are an SRT sidecar** (`dub.srt`) — video players shape Myanmar correctly from it. Burned-in PNG subtitles with proper Myanmar shaping (Pillow + Raqm) live in the Streamlit app.
- **Flip/color** reduce automated matches but do **not** guarantee copyright safety.
- **Units:** the run bills the account signed into Colab. AI Pro ≈ 200 compute units/month; if one account runs low, switch accounts (units do not pool).
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
