"""Local client for the Colab/Kaggle Recap GPU backend (gpu_server.py).

Usage — start the server cell in Colab/Kaggle first, copy the printed URL,
then on your Mac:

    python3 local_dub.py myvideo.mp4 \
        --server https://xxxx.trycloudflare.com --token MY_SECRET_TOKEN

It uploads the video, waits while the GPU dubs it, downloads the zip of
outputs (MP4 + shorts + SRT + script) next to your video file.

Requires only the stdlib + `requests` (pip install requests).
"""

import argparse
import os
import sys
import time
import zipfile

try:
    import requests
except ImportError:
    sys.exit("pip install requests first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="video file to dub")
    ap.add_argument("--server", required=True,
                    help="cloudflared URL printed by gpu_server.py")
    ap.add_argument("--token", required=True, help="same --token as the server")
    ap.add_argument("--voice", default="my-MM-NilarNeural",
                    choices=["my-MM-NilarNeural", "my-MM-ThihaNeural"])
    ap.add_argument("--language", default="auto",
                    help="auto / zh / en / my — auto = Whisper detects")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--no-shorts", action="store_true", help="skip 9:16 shorts")
    ap.add_argument("--keep-original-audio", action="store_true",
                    help="keep original audio low under the dub")
    ap.add_argument("--no-flip", action="store_true", help="no horizontal flip")
    ap.add_argument("--no-color", action="store_true", help="no color tweak")
    ap.add_argument("--out", default="", help="output zip path (default: <video>_dubbed.zip)")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"video not found: {args.video}")
    server = args.server.rstrip("/")
    headers = {"X-Token": args.token}

    # health check
    try:
        h = requests.get(server + "/health", headers=headers, timeout=30)
        h.raise_for_status()
        print("backend:", h.json())
    except Exception as e:
        sys.exit(f"cannot reach backend at {server}: {e}\n"
                 "Is the gpu_server.py cell still running?")

    print(f"uploading {args.video} "
          f"({os.path.getsize(args.video) / 1e6:.1f} MB)...", flush=True)
    t0 = time.time()
    with open(args.video, "rb") as fh:
        data = {
            "voice": args.voice,
            "language": args.language,
            "model": args.model,
            "make_shorts": str(not args.no_shorts),
            "mute_original": str(not args.keep_original_audio),
            "flip": str(not args.no_flip),
            "color": str(not args.no_color),
        }
        try:
            r = requests.post(server + "/dub", headers=headers,
                              files={"file": (os.path.basename(args.video), fh)},
                              data=data, timeout=args.timeout, stream=True)
        except requests.exceptions.ReadTimeout:
            sys.exit("timed out waiting for the GPU to finish — increase --timeout")
    if r.status_code == 401:
        sys.exit("401 bad token — pass the same --token the server was started with")
    if r.status_code != 200:
        sys.exit(f"server error {r.status_code}:\n{r.text[:2000]}")

    out = args.out or os.path.splitext(args.video)[0] + "_dubbed.zip"
    with open(out, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    print(f"done in {time.time() - t0:.0f}s -> {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")

    outdir = os.path.splitext(out)[0]
    os.makedirs(outdir, exist_ok=True)
    with zipfile.ZipFile(out) as z:
        z.extractall(outdir)
        for n in z.namelist():
            print("  ", n)
    if os.path.exists(os.path.join(outdir, "log.txt")):
        print("\n--- log ---")
        print(open(os.path.join(outdir, "log.txt")).read())


if __name__ == "__main__":
    main()
