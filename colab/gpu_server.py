"""Recap GPU backend server — run this inside a Colab/Kaggle notebook cell.

Turns the notebook's free GPU into a private API your local machine calls:

    Local Mac                    Colab / Kaggle GPU
    ─────────                    ──────────────────
    local_dub.py  ── video ──►   FastAPI + recap_pipe
                  ◄── MP4 ──     Whisper + translate + TTS + render

How to run (in a notebook cell on Colab or Kaggle):

    import subprocess
    subprocess.run(["wget", "-q",
        "https://raw.githubusercontent.com/htinkyawzaw2017-maker/"
        "recap-studio-streamlit/main/colab/recap_pipe.py",
        "-O", "/content/recap_pipe.py"], check=True)  # /kaggle/working on Kaggle
    subprocess.run(["wget", "-q",
        "https://raw.githubusercontent.com/htinkyawzaw2017-maker/"
        "recap-studio-streamlit/main/colab/gpu_server.py",
        "-O", "/content/gpu_server.py"], check=True)
    !python /content/gpu_server.py --token MY_SECRET_TOKEN

It installs deps, starts FastAPI on :8000, opens a cloudflared tunnel and
prints a https://*.trycloudflare.com URL. Copy that URL into local_dub.py's
--server argument. The URL dies when the session ends — copy the new one
each session.

Endpoints:
    GET  /health            -> {"ok": true, "cuda": bool}
    POST /transcribe        -> multipart wav upload -> {transcript, language, device}
    POST /dub               -> multipart video upload + form fields -> .zip of outputs

All endpoints require header  X-Token: <your token>.
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time

WORK = os.environ.get("RECAP_WORK", "/content/gpu_work")


def install_deps():
    subprocess.run(
        ["pip", "install", "-q", "faster-whisper", "google-genai", "edge-tts",
         "pydub", "yt-dlp", "requests", "fastapi", "uvicorn[standard]",
         "python-multipart", "av<19"],
        check=True)


def start_cloudflared(port):
    """Download cloudflared if needed, tunnel localhost:port, return URL."""
    bin_path = "/content/cloudflared" if os.path.isdir("/content") else "./cloudflared"
    if not os.path.exists(bin_path):
        subprocess.run(
            ["wget", "-q",
             "https://github.com/cloudflare/cloudflared/releases/latest/"
             "download/cloudflared-linux-amd64",
             "-O", bin_path], check=True)
        os.chmod(bin_path, 0o755)
    proc = subprocess.Popen(
        [bin_path, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    url_box = {}

    def read():
        for line in proc.stdout:
            m = re.search(r"https://[\w.-]+\.trycloudflare\.com", line)
            if m:
                url_box["url"] = m.group(0)
                return

    threading.Thread(target=read, daemon=True).start()
    for _ in range(60):  # wait up to 60s for the URL
        if "url" in url_box:
            return url_box["url"], proc
        time.sleep(1)
    raise RuntimeError("cloudflared did not print a tunnel URL within 60s")


def build_app(token, gemini_model, whisper_model):
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, "/content")
    sys.path.insert(0, "/kaggle/working")
    import recap_pipe as rp

    app = FastAPI(title="Recap GPU backend")

    def auth(x_token: str | None):
        if token and x_token != token:
            raise HTTPException(status_code=401, detail="bad token")

    @app.get("/health")
    def health():
        try:
            import torch
            cuda = torch.cuda.is_available()
        except Exception:
            cuda = False
        return {"ok": True, "cuda": cuda}

    @app.post("/transcribe")
    def transcribe(file: UploadFile = File(...),
                   model: str = Form(whisper_model),
                   language: str = Form("auto"),
                   x_token: str | None = Header(None)):
        auth(x_token)
        os.makedirs(WORK, exist_ok=True)
        audio = os.path.join(WORK, "up_" + os.path.basename(file.filename or "a.wav"))
        with open(audio, "wb") as fh:
            fh.write(file.file.read())
        lang = None if language == "auto" else language
        try:
            tr, detected, device = rp.transcribe(audio, model, lang)
        except Exception as e:
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, 500)
        return {"transcript": tr, "language": detected, "device": device}

    @app.post("/dub")
    def dub(file: UploadFile = File(...),
            voice: str = Form("my-MM-NilarNeural"),
            language: str = Form("auto"),
            make_shorts: bool = Form(True),
            mute_original: bool = Form(True),
            flip: bool = Form(True),
            color: bool = Form(True),
            model: str = Form(whisper_model),
            x_token: str | None = Header(None)):
        auth(x_token)
        import glob
        import shutil
        import zipfile

        run_dir = os.path.join(WORK, "dub_" + str(int(time.time())))
        os.makedirs(run_dir, exist_ok=True)
        ext = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
        src = os.path.join(run_dir, "source" + ext)
        with open(src, "wb") as fh:
            fh.write(file.file.read())

        api_key = os.environ.get("GEMINI_API_KEY", "")
        lang = None if language == "auto" else language
        main, shorts, log = rp.run_all(
            video_file=src, voice=voice, make_shorts_flag=make_shorts,
            api_key=api_key, whisper_model=model, language=lang,
            flip=flip, color=color, mute_original=mute_original,
            work=os.path.join(run_dir, "out"))
        if not main:
            return JSONResponse({"error": log}, 500)

        zpath = os.path.join(run_dir, "recap_output.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in [main, shorts] + glob.glob(
                    os.path.join(run_dir, "out", "*.srt")) + glob.glob(
                    os.path.join(run_dir, "out", "*.json")):
                if f and os.path.exists(f):
                    z.write(f, os.path.basename(f))
            z.writestr("log.txt", log)
        return FileResponse(zpath, media_type="application/zip",
                            filename="recap_output.zip")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("RECAP_TOKEN", "change-me"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--whisper-model", default="large-v3-turbo")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash-lite")
    args = ap.parse_args()

    print("[1/3] installing deps...", flush=True)
    install_deps()
    print("[2/3] starting cloudflared tunnel...", flush=True)
    url, _proc = start_cloudflared(args.port)
    print("=" * 60, flush=True)
    print(f"  PUBLIC URL : {url}")
    print(f"  TOKEN      : {args.token}")
    print("  local command:")
    print(f"    python3 local_dub.py video.mp4 --server {url} --token {args.token}")
    print("=" * 60, flush=True)
    print("[3/3] serving (keep this cell running)...", flush=True)

    import uvicorn
    uvicorn.run(build_app(args.token, args.gemini_model, args.whisper_model),
                host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
