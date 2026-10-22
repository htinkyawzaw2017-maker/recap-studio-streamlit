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
# 7 ── retime: voice stays 1.0x, picture slows to at most 0.9x + freeze
from recapstudio.media import get_media_duration
from recapstudio.retime import MIN_VIDEO_SPEED as _MVS, retime_video
long_text = "ဒီဇာတ်လမ်းမှာ သူရဲကောင်းတစ်ယောက် ရွာကို ကယ်တင်ဖို့ တောင်ပေါ်ကို တက်သွားပါတယ် " * 3
rt_dialogues = [
    {"start": 0.0, "end": 6.0, "speaker": "Recap", "text": long_text},
    {"start": 6.0, "end": 12.0, "speaker": "Recap", "text": "နောက်ဆုံးမှာ အားလုံး ပျော်ရွှင်စွာ နေထိုင်ကြပါတယ်။"},
]
rt_mix = engine.build_narration(dialogues=rt_dialogues, voice_cfg=_voice_config("my", "thiha"),
                                duration=info["duration"], work_dir=work, tag="rt", retime=True)
ok(bool(rt_mix.spans) and all(_MVS - 0.001 <= sp.speed <= 1.0 for sp in rt_mix.spans),
   f"retime speeds stay within {_MVS}–1.0x")
ok(all(pl.tts_rate == "+0%" and pl.tempo == 1.0 for pl in rt_mix.planned),
   "voice is never speed-adjusted (always natural 1.0x)")
ok(rt_mix.duration > info["duration"], "long line kept in full — output runs longer")
rt_vid = work / "retimed.mp4"
retime_video(str(src), rt_mix.spans, rt_vid)
ok(abs(get_media_duration(str(rt_vid)) - rt_mix.duration) < 0.35,
   "retimed video and narration lengths match (A/V sync)")

# 8 ── tighten: Recap voice must run continuously (no dead picture tails),
#      while Dubbing keeps the source timing exactly
short_dialogues = [
    {"start": 0.0, "end": 6.0, "speaker": "Recap", "text": "ဇာတ်လမ်း စပါပြီ။"},
    {"start": 6.0, "end": 12.0, "speaker": "Recap", "text": "သူရဲကောင်း ရောက်လာပါတယ်။"},
]
tg_mix = engine.build_narration(dialogues=short_dialogues, voice_cfg=_voice_config("my", "thiha"),
                                duration=info["duration"], work_dir=work, tag="tg", retime=True)
ok(tg_mix.duration < info["duration"] - 4
   and all((g["end"] - g["start"]) <= 0.6 for g in tg_mix.gaps),
   "recap tighten: no silent stretch longer than 0.6s")
dub_mix = engine.build_narration(dialogues=short_dialogues, voice_cfg=_voice_config("my", "thiha"),
                                 duration=info["duration"], work_dir=work, tag="dub",
                                 retime=True, strict_timing=True)
ok(abs(dub_mix.duration - info["duration"]) < 0.4,
   "dubbing stays faithful to source timing")

# 9 ── bundled fonts travel as base64 (GitHub snapshot is text-only) and
#      decode back to the exact TTFs — this is what fixes Myanmar shaping
#      on Streamlit Cloud, where no .ttf ships in the repo
import base64 as _b64
from recapstudio import config as _cfg
for _name in ("NotoSansMyanmar-Regular", "NotoSansMyanmar-Bold"):
    _b64_path = _cfg.BUNDLED_FONT_DIR / f"{_name}.ttf.b64"
    _tmp_ttf = work / f"{_name}.ttf"
    _tmp_ttf.write_bytes(_b64.b64decode(_b64_path.read_text(encoding="ascii")))
    from recapstudio.fonts import _valid_font
    ok(_b64_path.exists() and _valid_font(_tmp_ttf), f"bundled font b64 decodes: {_name}")

# 10 ── mirror mode flips the picture (subtitles stay readable — checked
#       by construction: hflip is applied before the subtitle burn)
mir_out = work / "final_mirror.mp4"
render_master(str(src), str(narration), str(mir_out), ass_path=str(ass),
              reframe="Original (no reframe)", quality="fast", mirror=True,
              target_size=(int(info["width"]), int(info["height"])),
              duration=info["duration"])
import subprocess as _sp
for _src_f, _name in ((out, "frame_normal"), (mir_out, "frame_mirror")):
    _sp.run(["ffmpeg", "-y", "-ss", "5", "-i", str(_src_f), "-frames:v", "1",
             str(work / f"{_name}.png")], capture_output=True)
from PIL import Image as _Im
import numpy as _np
_a = _np.asarray(_Im.open(work / "frame_normal.png").convert("L"), dtype=float)
_b = _np.asarray(_Im.open(work / "frame_mirror.png").convert("L"), dtype=float)
ok(_a.shape == _b.shape and abs(_a - _b[:, ::-1]).mean() < abs(_a - _b).mean(),
   "mirror mode flips the picture horizontally")

# 11 ── TTS clips have their leading encoder silence stripped (sync)
_padded = work / "padded.mp3"
_sil = work / "sil03.wav"
_tone = work / "tone1.wav"
_sp.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
         "-t", "0.3", str(_sil)], capture_output=True)
_sp.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1",
         str(_tone)], capture_output=True)
_lst = work / "pad_list.txt"
_lst.write_text(f"file '{_sil}'\nfile '{_tone}'\n", encoding="utf-8")
_sp.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(_lst),
         "-c:a", "libmp3lame", str(_padded)], capture_output=True)
_conv = work / "padded.wav"
ok(engine._convert(_padded, _conv, 1.0, None)
   and get_media_duration(str(_conv)) < 1.15,
   "leading silence stripped from TTS clips")

# 12 ── v1.4: per-minute exact fit (ကွက်တိ) — an over-long line must be
#       shortened to its scene instead of overrunning into the next scene
from recapstudio.ai import TimelineExtractor
from recapstudio.util import estimate_speech_seconds
_ext = TimelineExtractor()
_bad = [
    {"start": 0.0, "end": 5.0, "speaker": "", "text": long_text},
    {"start": 5.0, "end": 12.0, "speaker": "",
     "text": "နောက်ဆုံးမှာ အားလုံး ပျော်ရွှင်စွာ နေထိုင်ကြပါတယ်။"},
]
_fitted, _rep = _ext._enforce_minute_fit([dict(d) for d in _bad], 12.0)
ok(_rep["fixed_ai"] + _rep["fixed_cut"] >= 1,
   "minute-fit shortened the over-long line")
ok(estimate_speech_seconds(_fitted[0]["text"], "my") <= 5.0 * 1.15,
   "fitted line now fits its own scene window")

# 13 ── v1.4: retime floor — picture slows further before any freeze
from recapstudio.retime import compute_spans
ok(_MVS <= 0.75, f"retime floor allows deeper slowdown ({_MVS}x)")
_sp1 = compute_spans([(0.0, 4.0, 5.2, 0)], 4.0)
ok(abs(_sp1[0].speed - 4.0 / 5.2) < 0.01 and _sp1[0].freeze == 0.0,
   "5.2s speech in a 4s scene: picture slows, no freeze needed")
_sp2 = compute_spans([(0.0, 4.0, 6.0, 0)], 4.0)
ok(abs(_sp2[0].speed - _MVS) < 0.01 and _sp2[0].freeze > 0.5,
   "extreme overflow: floor speed first, freeze only the remainder")

# 14 ── v1.4: PNG/Raqm subtitle burn (libass shapes Myanmar incorrectly in
#       our deploy environments; shaping itself is A/B-verified visually)
from recapstudio import subpng
ok(subpng.raqm_available(), "Raqm shaping engine available")
_subs_dir = work / "subs"
_track = subpng.build_subtitle_track(
    dialogues, width=int(info["width"]), height=int(info["height"]),
    duration=info["duration"], work_dir=_subs_dir)
ok(_track is not None and Path(_track).exists(), "PNG subtitle track built")
_th = int(get_video_info(str(_track))["height"])
_burned = work / "final_pngsubs.mp4"
subpng.burn_subtitles(str(out), str(_burned), track=_track, track_height=_th,
                      height=int(info["height"]), duration=info["duration"])
ok(_burned.exists()
   and abs(get_media_duration(str(_burned)) - info["duration"]) < 0.4,
   "PNG subtitles burned onto the video, duration kept")

# 15 ── v1.4.1 regression: analysis runs chunks in worker threads; the
#       app's log sink must be a plain list, never st.session_state
#       (touching it from a thread killed every chunk on Cloud with
#       "'st.session_state has no key logs'")
_app_src = (Path(__file__).resolve().parent.parent / "streamlit_app.py").read_text(encoding="utf-8")
ok('SS["logs"].append' not in _app_src,
   "app log_fn never writes session_state from a callback")
from concurrent.futures import ThreadPoolExecutor as _TPE
_shared: list[str] = []
_ext2 = TimelineExtractor(log_fn=lambda m: _shared.append(str(m)))
with _TPE(max_workers=4) as _pool:
    list(_pool.map(lambda i: _ext2.log(f"line {i}"), range(20)))
ok(len(_shared) == 20, "extractor logs fine from worker threads (plain list)")

# 16 ── v1.5: fragment merge — half-thought lines are joined back into
#       complete, listenable sentences (recap mode only)
_frags = [
    {"start": 0.0, "end": 3.0, "speaker": "", "text": "ဒါပေမဲ့ ကလေးပုံမြင်"},
    {"start": 3.2, "end": 7.0, "speaker": "", "text": "လိုက်တော့ သူမ အံ့ဩသွားပါတယ်။"},
    {"start": 8.0, "end": 11.0, "speaker": "", "text": "နောက်ထပ် ဝါကျ အပြည့် တစ်ခု ဖြစ်ပါတယ်။"},
]
_merged = TimelineExtractor()._merge_fragments([dict(d) for d in _frags])
ok(len(_merged) == 2 and _merged[0]["text"].endswith("ပါတယ်။"),
   "fragment lines merged into complete sentences")

# 17 ── v1.6: Burmese sentence-completeness detector on the owner's real
#       examples (fragments DO end with "။" — punctuation is not enough)
from recapstudio.ai import sentence_complete
for _frag in ("နှစ်ခြင်းနေတဲ့ တိုက်တန်းနှစ်ပေါ်က မခင်တစ်ဦးက။",
              "ပြင်သစ်လို ပြောပေမဲ့ နာမည်၊ နေရပ် ပြောဖို့ ငယ်လွန်း။",
              "ဒါပေမဲ့ ကလေးတွေ ဓာတ်ပုံ ပြင်သစ်သတင်းစာမှာ ပါလာတော့။",
              "အမေရိကမှာ ဘဝသစ်စဖို့။",
              "သူမကတော့ ဒီအကြောင်းတွေကို လုံးဝမသိခဲ့ဘဲ။",
              "သားလေးတွေနဲ့ ပြန်လည်ဆုံစည်းဖို့ အတ္တလတ္တိတ်သမုဒ္ဒရာကို။"):
    ok(not sentence_complete(_frag, "my"), f"fragment detected: {_frag[:18]}…")
for _full in ("သူက သားတွေနဲ့ မလိုက်ဘဲ သင်္ဘောနဲ့ အတူနစ်မြုပ်ခဲ့တယ်။",
              "ရက်သတ္တပတ်များစွာ ကြာသည်အထိ လာခေါ်သူ မရှိ။"):
    ok(sentence_complete(_full, "my"), f"complete sentence kept: {_full[:18]}…")
_pair = [
    {"start": 0.0, "end": 4.0, "speaker": "",
     "text": "နှစ်ခြင်းနေတဲ့ တိုက်တန်းနှစ်ပေါ်က မခင်တစ်ဦးက"},
    {"start": 4.0, "end": 9.0, "speaker": "",
     "text": "သူက သားတွေနဲ့ မလိုက်ဘဲ သင်္ဘောနဲ့ အတူနစ်မြုပ်ခဲ့တယ်။"},
]
_m2 = TimelineExtractor()._merge_fragments([dict(d) for d in _pair])
ok(len(_m2) == 1 and sentence_complete(_m2[0]["text"], "my"),
   "screenshot pair merged into one complete line")

print(f"\nALL {checks} CHECKS PASSED → {out}")
