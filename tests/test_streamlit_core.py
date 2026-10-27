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
              "သားလေးတွေနဲ့ ပြန်လည်ဆုံစည်းဖို့ အတ္တလတ္တိတ်သမုဒ္ဒရာကို။",
              # v1.10: verb-less endings from the owner's Ji Gong recap —
              # direction, dangling conjunction, bare noun phrase, participle
              "အာဖူက ဘဲကင်ဆိုင်သို့။",
              "ဂျီကုန်းက အာဖူကို ကလေးနှစ်ယောက်၊ ယောက်ျားလေးနှင့်။",
              "အာဖူမျက်လုံးများ။",
              "အာဖူက စားပွဲတွင် ထိုင်ကာ။",
              "သူက ငိုလျက်။"):
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

# 18 ── v1.8: caption-style terse lines (complete but empty) at the start
#       get merged into full lines, and the repair pass may EXPAND a
#       terse line up to its window budget (the old 1.45x cap blocked it)
from recapstudio.ai import line_is_terse
ok(line_is_terse("သူဟာ လူတိုင်း။", 6.0, "my"), "terse caption line detected")
ok(not line_is_terse("သူဟာ ရွာထဲက လူတိုင်း လေးစားရတဲ့ လူကြီး တစ်ဦး ဖြစ်ပါတယ်။", 6.0, "my"),
   "full line is not terse")
# v1.10: the old >=4s gate blinded this check for fast-cut drama (2-3s
# segments) — a scrap in a short window is terse, a full line using its
# whole short window is not
ok(line_is_terse("အာဖူမျက်လုံးများ။", 2.0, "my"), "terse scrap in short window detected")
ok(not line_is_terse("အာဖူက ထွက်ပြေးသွားသည်။", 2.0, "my"),
   "full line filling short window is not terse")
_captions = [
    {"start": 0.0, "end": 5.0, "speaker": "", "text": "သူဟာ ရွာသား တစ်ဦးပါ။"},
    {"start": 5.0, "end": 10.0, "speaker": "", "text": "သူ့မှာ သား နှစ်ယောက် ရှိပါတယ်။"},
    {"start": 10.0, "end": 17.0, "speaker": "",
     "text": "တစ်နေ့မှာတော့ မုန်တိုင်းကြီး တိုက်လာပြီး သူတို့ရဲ့ အိမ်လေး ပျက်စီးသွားခဲ့ပါတယ်။"},
]
_mc = TimelineExtractor()._merge_fragments([dict(d) for d in _captions])
ok(len(_mc) == 2 and "ရွာသား" in _mc[0]["text"] and "သား နှစ်ယောက်" in _mc[0]["text"],
   "opening caption lines merged into one full line")

import json as _json


class _FakeClient:
    def generate_json(self, model, parts, schema, **kw):
        return _json.dumps({"lines": [{"index": 0,
            "text": "သူဟာ ရွာထဲက လူတိုင်း လေးစားရတဲ့ လူကြီး တစ်ဦး ဖြစ်ပါတယ်။"}]})


_ex = TimelineExtractor()
_ex.client = _FakeClient()
_repaired = _ex._repair_fragments([
    {"start": 0.0, "end": 8.0, "speaker": "", "text": "သူဟာ လူတိုင်း။"},
    {"start": 8.0, "end": 15.0, "speaker": "",
     "text": "တစ်နေ့မှာတော့ မုန်တိုင်းကြီး တိုက်လာပြီး သူတို့ရဲ့ အိမ်လေး ပျက်စီးသွားခဲ့ပါတယ်။"},
])
ok("လေးစားရတဲ့" in _repaired[0]["text"],
   "repair pass expands a terse line up to its window budget")

# ── v1.12: bare-topic / trailing-also fragments + flowing merge ───────
for _frag in ("သူဟာ။", "သူမဟာ။", "ငါလည်း။"):
    ok(not sentence_complete(_frag, "my"), f"fragment detected: {_frag[:18]}…")
for _full in ("သူဟာ လူကြီး တစ်ဦး ဖြစ်ပါတယ်။", "ဒါဟာ ငါ့အိမ်ပါ။"):
    ok(sentence_complete(_full, "my"), f"complete sentence kept: {_full[:18]}…")
# the owner's Titanic chain: without the ဟာ suffix the merge chain broke
# at line 3 and nothing joined; with it + the strip-join, lines 1-4 flow
# as ONE sentence (single final "။") and line 5 stays separate (16s cap)
_titanic = [
    {"start": 0.0, "end": 3.9, "speaker": "",
     "text": "တိုက်တန်းနစ်သင်္ဘောကြီး နစ်မြုပ်နေချိန်မှာ။"},
    {"start": 4.0, "end": 6.7, "speaker": "", "text": "ဒါပေမဲ့ သူကိုယ်တိုင်ကတော့။"},
    {"start": 7.0, "end": 9.2, "speaker": "", "text": "အဲဒီနောက်မှာတော့ သူဟာ။"},
    {"start": 9.7, "end": 13.0, "speaker": "",
     "text": "ကလေးနှစ်ယောက်ကို ကယ်ဆယ်ရေးလှေပေါ်။"},
    {"start": 13.6, "end": 19.3, "speaker": "",
     "text": "ပြင်သစ်လိုပြောပေမဲ့ ငယ်လွန်း၍ နာမည်၊ လိပ်စာ မပြောနိုင်ခဲ့။"},
]
_mt = TimelineExtractor()._merge_fragments([dict(d) for d in _titanic])
ok(len(_mt) == 2 and _mt[0]["text"].count("။") == 1
   and _mt[0]["end"] == 13.0 and "သူဟာ ကလေးနှစ်ယောက်" in _mt[0]["text"],
   "titanic fragments merge into one flowing sentence")

# ── v1.13: dubbing lines get completed, not left dangling ───────────
for _frag in ("အဘွားကြီးရဲ့။", "သူ၏။",
              "ဒီနှစ်မှာတော့ ကျွန်တော် အသက်နှစ်ဆယ်...။"):
    ok(not sentence_complete(_frag, "my"), f"fragment detected: {_frag[:18]}…")
ok(sentence_complete("ဒါဆို... ယန်ဝမ်ရဲ့ မျက်လုံးလား။", "my"),
   "mid-line ellipsis with finished ending stays complete")
ok(sentence_complete("အဘွားကြီးရဲ့ အိမ်ကို သွားပါတယ်။", "my"),
   "possessive mid-sentence stays complete")
# v1.16 update: dubbing merge stays off ACROSS speakers (same-speaker
# turns now merge — see the v1.16 Narrator test below)
_dub = [
    {"start": 21.8, "end": 25.7, "speaker": "Jigong",
     "text": "ဒါကြောင့် ကျွန်တော်ကတော့ ပျင်းရိတဲ့သူတွေကို။"},
    {"start": 27.2, "end": 28.1, "speaker": "Old Woman", "text": "ဖိနပ်က ပြဲနေပြီ။"},
]
ok(len(TimelineExtractor(fill_mode="dialogue")._merge_fragments(
    [dict(d) for d in _dub])) == 2, "dubbing merge stays off across speakers")


class _FakeDubClient:
    def generate_json(self, model, parts, schema, **kw):
        return _json.dumps({"lines": [{"index": 0,
            "text": "ဒါပေမဲ့ ဆရာတော်ပြောတဲ့ မျက်စိကွယ်တယ်ဆိုတာက အမှန်ပါ။"}]})


_exd = TimelineExtractor(fill_mode="dialogue")
_exd.client = _FakeDubClient()
_repd = _exd._repair_fragments([
    {"start": 17.5, "end": 21.4, "speaker": "Jigong",
     "text": "ဒါပေမဲ့ ဆရာတော်ပြောတဲ့ မျက်စိကွယ်တယ်ဆိုတာ။"},
    {"start": 59.4, "end": 60.9, "speaker": "Old Woman",
     "text": "ရဟန်းက အသားစားလို့ရလို့လား။"},
])
ok("အမှန်ပါ" in _repd[0]["text"],
   "dubbing repair completes a dangling turn in its own window")
ok(_repd[1]["text"] == "ရဟန်းက အသားစားလို့ရလို့လား။",
   "complete dubbing turn untouched by repair")

# ── v1.14: same-timestamp pile-ups get spread into sequence ─────────
from recapstudio.ai import normalise
# the owner's dubbing tail: six turns all stamped at the video duration
_pile = [{"start": 138.5, "end": 142.0, "speaker": "Man", "text": "သွား၊ နင်နဲ့ ယမမင်းရဲ့ ပတ်သက်မှုတွေကို ဆေးကြောလိုက်။"}]
for _t in ("အာဖူ၊ မင်း ကိုယ့်စိတ်ကို လိုက်နာရမယ်။", "ဒီလူတွေရဲ့။", "ပြုံးမျက်နှ။",
           "နွားခေါင်း။", "ငရဲပြည်ရဲ့။", "ဟောင်းနွမ်း။"):
    _pile.append({"start": 180.070312, "end": 180.070312, "speaker": "Man", "text": _t})
_sp = normalise(_pile, 180.070312)
_tail = _sp[1:]
ok(len(_tail) == 6
   and all(_tail[k]["start"] < _tail[k + 1]["start"] for k in range(5))
   and all(d["end"] <= 180.070312 for d in _tail)
   and _tail[0]["start"] >= 142.05
   and _tail[0]["text"].startswith("အာဖူ") and _tail[-1]["text"].startswith("ဟောင်း"),
   "duration pile-up spreads into ordered lines inside the video")
# pile-up mid-video lays forward from its stamp, capped at the next line
_mid = [{"start": 10.0, "end": 10.0, "speaker": "", "text": "စာတစ်ကြောင်း။"},
        {"start": 10.0, "end": 10.0, "speaker": "", "text": "စာနှစ်ကြောင်း။"},
        {"start": 10.0, "end": 10.0, "speaker": "", "text": "စာသုံးကြောင်း။"},
        {"start": 20.0, "end": 23.0, "speaker": "", "text": "နောက်လိုင်း။"}]
_sm = normalise(_mid, 60.0)
ok(_sm[0]["start"] == 10.0
   and _sm[0]["start"] < _sm[1]["start"] < _sm[2]["start"]
   and _sm[2]["end"] <= 20.0, "mid-video pile-up lays forward in order")
# a lone line stamped exactly at the duration is pulled back inside
_one = [{"start": 100.0, "end": 105.0, "speaker": "", "text": "အရင်လိုင်း။"},
        {"start": 180.070312, "end": 180.070312, "speaker": "", "text": "နောက်ဆုံးလိုင်း။"}]
_so = normalise(_one, 180.070312)
ok(_so[1]["start"] < 180.070312 and _so[1]["end"] == 180.070312,
   "line stamped at duration is pulled back inside")

# ── v1.15: 100x-compressed timestamps are rescaled, never trusted ───
# the owner's Titanic paste: model wrote seconds/100 (0.47 for 47s) and
# minute-fit then amputated every line mid-word to a ~10-char budget
_compressed = [
    {"start": 0.0, "end": 0.02, "speaker": "Speaker", "text": "သူက သူ့သားတို့။"},
    {"start": 0.02, "end": 0.04, "speaker": "Speaker", "text": "တိုက်တန်းနစ်သင်္ဘောကြီး။"},
    {"start": 0.05, "end": 0.06, "speaker": "Speaker", "text": "သူတို့နဲ့ လိုက်မသွားဘူး။"},
    {"start": 0.43, "end": 0.45, "speaker": "Speaker", "text": "သူမ အတ္တလန္တိတ်ကို ဖြတ်ခဲ့တယ်။"},
    {"start": 0.45, "end": 0.47, "speaker": "Speaker", "text": "သားတွေနဲ့ အတူနေခဲ့တယ်။"},
]
_sc = normalise(_compressed, 47.07)
ok([round(d["start"], 2) for d in _sc] == [0.0, 2.0, 5.0, 43.0, 45.0]
   and round(_sc[-1]["end"], 2) == 47.0
   and "တိုက်တန်းနစ်သင်္ဘောကြီး" in _sc[1]["text"],
   "seconds/100 timestamps rescaled x100, text kept whole")
# 0-1 fractions rescale by duration instead
_frac = [
    {"start": 0.0, "end": 0.1, "speaker": "", "text": "သူဟာ ရွာထဲက လူကြီး တစ်ဦး ဖြစ်ပါတယ်။"},
    {"start": 0.25, "end": 0.35, "speaker": "", "text": "တစ်နေ့မှာတော့ မုန်တိုင်းကြီး တိုက်လာခဲ့ပါတယ်။"},
    {"start": 0.5, "end": 0.6, "speaker": "", "text": "သူတို့ရဲ့ အိမ်လေး ပျက်စီးသွားခဲ့ပါတယ်။"},
    {"start": 1.0, "end": 1.1, "speaker": "", "text": "နောက်ဆုံးမှာတော့ အားလုံး ပြန်လည်ထူထောင်ခဲ့ပါတယ်။"},
]
_sf = normalise(_frac, 47.0)
ok(round(_sf[1]["start"], 2) == 11.75 and _sf[-1]["end"] == 47.0
   and all(_sf[k]["start"] < _sf[k + 1]["start"] for k in range(3)),
   "0-1 fraction timestamps rescaled by duration")
# sane timestamps and short videos are never touched
_sane = [
    {"start": 0.0, "end": 3.9, "speaker": "", "text": "တိုက်တန်းနစ်သင်္ဘောကြီး နစ်မြုပ်နေချိန်မှာ။"},
    {"start": 4.0, "end": 6.7, "speaker": "", "text": "ဒါပေမဲ့ သူကိုယ်တိုင်ကတော့။"},
    {"start": 7.0, "end": 9.2, "speaker": "", "text": "အဲဒီနောက်မှာတော့ သူဟာ။"},
]
_ss = normalise(_sane, 47.07)
ok([d["start"] for d in _ss] == [0.0, 4.0, 7.0], "sane timestamps untouched")
_short = [
    {"start": 0.0, "end": 0.02, "speaker": "", "text": "စာတစ်ကြောင်း။"},
    {"start": 0.02, "end": 0.04, "speaker": "", "text": "စာနှစ်ကြောင်း။"},
    {"start": 0.05, "end": 0.06, "speaker": "", "text": "စာသုံးကြောင်း။"},
]
_sh = normalise(_short, 5.0)
ok([d["start"] for d in _sh] == [0.0, 0.02, 0.05], "short video guard: no rescale")
# minute-fit never butchers a sub-second window line
_exf = TimelineExtractor()
_fit, _rep = _exf._enforce_minute_fit(
    [{"start": 0.0, "end": 0.4, "speaker": "",
      "text": "သူဟာ ရွာထဲက လူတိုင်း လေးစားရတဲ့ လူကြီး တစ်ဦး ဖြစ်ပါတယ်။"}],
    60.0)
ok(_fit[0]["text"] == "သူဟာ ရွာထဲက လူတိုင်း လေးစားရတဲ့ လူကြီး တစ်ဦး ဖြစ်ပါတယ်။",
   "sub-second window line left whole by minute-fit")

# ── v1.16: dubbing same-speaker merge + Burmese "." → "။" ───────────
from recapstudio.ai import normalise_my_punctuation
ok(normalise_my_punctuation("ခေါ်ခဲ့. ဘယ်သူမှ မသိဘူး.") ==
   "ခေါ်ခဲ့။ ဘယ်သူမှ မသိဘူး။", "latin periods become Burmese full stops")
ok(normalise_my_punctuation("ဒေါ်လာ ၄.၅ သန်း။") == "ဒေါ်လာ ၄.၅ သန်း။",
   "decimals untouched by punctuation normalisation")
# the owner's new Narrator paste: one narrator's sentence split across
# three turns joins back into one flowing line (no words invented)
_narr = [
    {"start": 2.0, "end": 4.0, "speaker": "Narrator",
     "text": "တိုက်တန်းနစ် နစ်မြုပ်နေချိန်"},
    {"start": 4.0, "end": 6.0, "speaker": "Narrator",
     "text": "ဒါပေမဲ့ သူတို့နဲ့ လိုက်မယ့်အစား"},
    {"start": 6.0, "end": 9.0, "speaker": "Narrator",
     "text": "သူတို့ကို နှုတ်ဆက်နမ်းရှုံ့ပြီး သင်္ဘောနဲ့ နစ်မြုပ်ခဲ့တယ်"},
]
_mn = TimelineExtractor(fill_mode="dialogue")._merge_fragments(
    [dict(d) for d in _narr])
ok(len(_mn) == 1 and _mn[0]["end"] == 9.0 and "နစ်မြုပ်ခဲ့တယ်" in _mn[0]["text"],
   "same-speaker dubbing turns merge into one sentence")
# unknown speakers never merge in dubbing
_noname = [
    {"start": 0.0, "end": 2.0, "speaker": "", "text": "ဒါပေမဲ့ သူတို့နဲ့။"},
    {"start": 2.0, "end": 4.0, "speaker": "", "text": "လိုက်မသွားဘူး။"},
]
ok(len(TimelineExtractor(fill_mode="dialogue")._merge_fragments(
    [dict(d) for d in _noname])) == 2, "unknown-speaker dubbing turns stay separate")

# ── v1.17: long silences surface in the report + clean timestamps ───
_cov_lines = [
    {"start": 0.001, "end": 0.961, "speaker": "Woman", "text": "တကယ်ကို။"},
    {"start": 3.033, "end": 9.093, "speaker": "Man 1", "text": "အာဖူ၊ လူသက်တမ်းကြည့်တိုင်း။"},
    {"start": 9.693, "end": 13.993, "speaker": "Man 2", "text": "ကျွန်တော် မသိပါဘူး။"},
    {"start": 15.063, "end": 22.023, "speaker": "Man 2", "text": "နောက်ပိုင်း လူကြည့်တိုင်း။"},
    {"start": 27.273, "end": 31.063, "speaker": "Man 1", "text": "သက်တမ်းဆိုတာမျိုးက။"},
    {"start": 32.193, "end": 35.593, "speaker": "Man 1", "text": "အာဖူ မင်းဒီလိုလုပ်တာက။"},
    {"start": 37.493, "end": 42.433, "speaker": "Man 2", "text": "ကျွန်တော် ဒါတွေကို။"},
    {"start": 107.153, "end": 110.403, "speaker": "Bull Head", "text": "အာဖူ မင်းက။"},
    {"start": 114.283, "end": 117.653, "speaker": "Bull Head", "text": "ကံကြမ္မာကို။"},
    {"start": 124.083, "end": 126.063, "speaker": "Man 1", "text": "အာဖူ မင်းဒုက္ခရောက်ပြီ။"},
    {"start": 144.383, "end": 145.023, "speaker": "Man 2", "text": "ဟယ်!"},
    {"start": 146.333, "end": 180.08, "speaker": "Man 2", "text": "ဟာ ဒုက္ခပဲ။"},
]
_rep = TimelineExtractor()._coverage(_cov_lines, 180.08, 0)
ok(_rep["quiet_gaps"] == [{"start": 42.43, "end": 107.15, "seconds": 64.72}],
   "65s dubbing hole surfaces in coverage report (18s hole stays silent)")
_float = normalise([{"start": 147.65300000000002, "end": 154.13766040000002,
                     "speaker": "Man 2", "text": "စကားပြောရအောင်။"}], 180.08)
ok(_float[0]["start"] == 147.65 and _float[0]["end"] == 154.14,
   "computed timestamps round to 0.01s")

# ── v1.11: project persistence — refresh must not reset to step 1 ─────
import tempfile as _tf
from pathlib import Path as _Pl
from recapstudio.progress import load_progress, save_progress
_tmpd = _tf.mkdtemp(prefix="recap_progress_")
_state = {"video_path": "/tmp/v.mp4", "video_info": {"duration": 180.0},
          "parts": [], "current_path": "/tmp/v.mp4", "current_label": "full",
          "dialogues": [{"start": 0.0, "end": 5.0, "speaker": "", "text": "Test line."}],
          "hook1": "Hook", "hook2": "", "coverage": {"coverage_percent": 95},
          "fit": {}, "script_version": 1, "script_confirmed": True,
          "edited_dialogues": [{"start": 0.0, "end": 5.0, "speaker": "", "text": "Edited."}],
          "draft_dialogues": [], "outputs": {"mp4": "/tmp/f.mp4", "size": 123},
          "render_cfg": {"lang": "my"}, "keys": ["SECRET-MUST-NOT-PERSIST"],
          "busy": True}
ok(save_progress(_tmpd, "abc123", _state), "progress saved to disk")
_back = load_progress(_tmpd, "abc123")
ok(_back is not None and _back["script_confirmed"] is True
   and _back["edited_dialogues"][0]["text"] == "Edited."
   and _back["outputs"]["mp4"] == "/tmp/f.mp4",
   "progress round-trips (script + outputs)")
ok("keys" not in _back and "busy" not in _back,
   "secrets/transient state never persisted")
ok(load_progress(_tmpd, "nope") is None, "missing project loads as None")
(_Pl(_tmpd) / "project_bad" / "progress.json").parent.mkdir(exist_ok=True)
(_Pl(_tmpd) / "project_bad" / "progress.json").write_text("{not json")
ok(load_progress(_tmpd, "bad") is None, "corrupt save loads as None, never crashes")
ok(not save_progress(_tmpd, "", _state), "empty project id refuses to save")

print(f"\nALL {checks} CHECKS PASSED → {out}")
