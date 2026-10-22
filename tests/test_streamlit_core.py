"""Offline end-to-end test of the Streamlit edition's core pipeline.

Run with demo/fake env so no API key or network TTS is needed:
    RECAP_DATA_DIR=/tmp/recap_test RECAP_FAKE_TTS=1 RECAP_DEMO_MODE=1 \
        python tests/test_streamlit_core.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recapstudio import config
config.ensure_dirs()

from recapstudio.ai import demo_timeline
from recapstudio.keys import UnlimitedKeyPool
from recapstudio.media import generate_test_media, get_video_info
from recapstudio.pipeline import RecapPipeline, _voice_config
from recapstudio.render import export_srt, render_master
from recapstudio.subtitles import build_ass
from recapstudio.tts import TTSEngine, master_audio

work = Path(tempfile.mkdtemp(prefix="recap_core_test_"))
checks = 0

def ok(cond, label):
    global checks
    assert cond, f"FAILED: {label}"
    checks += 1
    print(f"  ✓ {label}")

# 1 ── unlimited key pool (25 keys, failover across all of them)
pool = UnlimitedKeyPool([f"AIza{i:036d}" for i in range(25)], cooldown_seconds=60)
ok(len(pool) == 25 and pool.has_key(), "pool holds 25 keys")
k1, s1 = pool.active_key()
nxt = pool.rotate_after_failure(s1, "429 quota")
ok(nxt and nxt != k1 and pool.active_index != s1 - 1, "failover moves to another key")

# 2 ── synthetic source video (12s)
src = work / "src.mp4"
generate_test_media(src, seconds=12)
info = get_video_info(str(src))
ok(info["duration"] > 11, f"test video created ({info['duration']:.1f}s)")

# 3 ── splitter → parts carry absolute paths for the transfer button
pipe = RecapPipeline(None)
parts = pipe.split_video(str(src), 5, "original", out_dir=work / "parts")
ok(len(parts) == 3, f"splitter produced {len(parts)} parts")

# 4 ── demo timeline (stands in for a Gemini timeline in this offline test)
data = demo_timeline(info["duration"], "my")
dialogues = data["dialogues"]
ok(len(dialogues) >= 2, f"timeline lines: {len(dialogues)}")

# 5 ── script-edit maths: every line fits (or is flagged) by the CPS budget
from recapstudio.tts import chars_per_second
cps = chars_per_second("my")
over = [d for d in dialogues if len(d["text"]) / cps > (d["end"] - d["start"]) * 1.12]
print(f"  · over-budget lines in demo script: {len(over)} (editor flags these)")

# 6 ── voice (fake TTS beeps offline) + master + subtitles + render
engine = TTSEngine()
mix = engine.build_narration(dialogues=dialogues, voice_cfg=_voice_config("my", "thiha"),
                             duration=info["duration"], work_dir=work, tag="test")
ok(Path(mix.audio_path).exists(), "narration track built")
narration = work / "narration.mp3"
master_audio(mix.audio_path, narration, info["duration"])
ok(narration.exists() and narration.stat().st_size > 1000, "narration mastered to MP3")
ass = work / "recap.ass"
build_ass(dialogues=dialogues, duration=info["duration"], ass_path=ass,
          hook_line1=data["hook_line1"], hook_line2=data["hook_line2"],
          font_size=42, v_margin=100, play_res=(int(info["width"]), int(info["height"])))
srt = work / "recap.srt"
export_srt(dialogues, srt)
ok(ass.exists() and srt.exists(), "ASS + SRT subtitles built")
out = work / "final.mp4"
res = render_master(str(src), str(narration), str(out), ass_path=str(ass),
                    reframe="Original (no reframe)", quality="fast",
                    target_size=(int(info["width"]), int(info["height"])),
                    duration=info["duration"])
ok(out.exists() and out.stat().st_size > 5000, f"final MP4 rendered ({out.stat().st_size} bytes)")
print(f"\nALL {checks} CHECKS PASSED → {out}")
