# Recap Studio MM — Free Edition (Streamlit)
# ─────────────────────────────────────────────────────────────────────────
# All-free movie/video recap studio:
#   • Unlimited Gemini API key pool (AI Studio free keys, auto-failover)
#   • Any source language → natural 🇲🇲 Myanmar or 🇬🇧 English recap
#   • Long-video splitter → one click sends a part into Recap mode
#   • Script editor gate — review/edit every line BEFORE rendering
#   • Edge TTS voices (free) + FFmpeg render, zero-drift timeline
#
# Run locally:   streamlit run streamlit_app.py
# Deploy:        Streamlit Community Cloud (see README.md)
from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

# ── data dir MUST be set before recapstudio is imported ──────────────────
DATA_DIR = Path(os.getenv("RECAP_DATA_DIR",
                          os.path.join(tempfile.gettempdir(), "recap_studio_data")))
os.environ.setdefault("RECAP_DATA_DIR", str(DATA_DIR))
DATA_DIR.mkdir(parents=True, exist_ok=True)

import streamlit as st  # noqa: E402

# ── bridge Streamlit Secrets → env (before importing the engine) ─────────
def _secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

if _secret("RECAP_DEMO_MODE"):
    os.environ["RECAP_DEMO_MODE"] = str(_secret("RECAP_DEMO_MODE"))
if _secret("RECAP_FAKE_TTS"):
    os.environ["RECAP_FAKE_TTS"] = str(_secret("RECAP_FAKE_TTS"))

st.set_page_config(page_title="Recap Studio MM — Free Edition",
                   page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

# ── engine imports (after env is ready) ──────────────────────────────────
from recapstudio import config as recap_config  # noqa: E402
from recapstudio.ai import demo_timeline, extract_timeline  # noqa: E402
from recapstudio.keys import UnlimitedKeyPool, mask, validate_key  # noqa: E402
from recapstudio.media import get_video_info  # noqa: E402
from recapstudio.pipeline import RecapPipeline, _fallback_voice, _voice_config  # noqa: E402
from recapstudio.render import export_srt, render_master  # noqa: E402
from recapstudio.subtitles import build_ass  # noqa: E402
from recapstudio.tts import (  # noqa: E402
    TTSEngine,
    VOICE_CATALOG,
    chars_per_second,
    fit_lines_to_windows,
    master_audio,
)
from recapstudio.util import human_bytes, human_time  # noqa: E402

recap_config.ensure_dirs()
WORK_DIR = DATA_DIR / "streamlit_jobs"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# 🎨 Theme
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg, #0b1020 0%, #101736 220px, #0e1220 100%); }
  .hero {
    background: linear-gradient(120deg, #6d28d9 0%, #db2777 55%, #f59e0b 130%);
    border-radius: 20px; padding: 22px 28px; color: #fff;
    box-shadow: 0 12px 32px rgba(0,0,0,.45); margin-bottom: 10px;
  }
  .hero h1 { margin: 0; font-size: 1.9rem; line-height: 1.2; }
  .hero p  { margin: 6px 0 0 0; opacity: .92; font-size: .98rem; }
  .stepbar { display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 0 4px 0; }
  .pill {
    padding: 7px 14px; border-radius: 999px; font-size: .86rem; font-weight: 600;
    background: rgba(255,255,255,.08); color: #cbd5e1; border: 1px solid rgba(255,255,255,.14);
  }
  .pill.done   { background: #14532d; color: #bbf7d0; border-color: #22c55e55; }
  .pill.active { background: #7c3aed; color: #fff; border-color: #a78bfa; }
  .card {
    background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.10);
    border-radius: 16px; padding: 6px 18px 14px 18px; margin: 14px 0;
  }
  .card h3 { margin-top: 12px; }
  div[data-testid="stMetric"] {
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.09);
    border-radius: 12px; padding: 10px 12px;
  }
  .stButton > button { border-radius: 12px; font-weight: 700; }
  .bigbtn > button {
    background: linear-gradient(120deg, #7c3aed, #db2777) !important; color: #fff !important;
    border: none !important; padding: .65rem 1rem !important; font-size: 1.02rem !important;
  }
  .okbtn > button { background: #16a34a !important; color: #fff !important; border: none !important; }
  .warnline { color: #fbbf24; font-size: .88rem; }
  .footer { text-align: center; color: #94a3b8; font-size: .82rem; margin: 26px 0 8px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────
SS = st.session_state
SS.setdefault("keys", [])            # list[str] — Gemini keys (memory only)
SS.setdefault("video_path", "")      # original uploaded/linked video
SS.setdefault("video_info", None)
SS.setdefault("parts", [])           # splitter results
SS.setdefault("current_path", "")    # video (or part) sent to Recap mode
SS.setdefault("current_label", "")
SS.setdefault("dialogues", [])       # AI timeline (raw)
SS.setdefault("hook1", "")
SS.setdefault("hook2", "")
SS.setdefault("coverage", None)
SS.setdefault("script_version", 0)   # bump → editor resets from dialogues
SS.setdefault("script_confirmed", False)
SS.setdefault("edited_dialogues", [])
SS.setdefault("outputs", None)       # render results
SS.setdefault("logs", [])


def secret_keys() -> list[str]:
    """Keys supplied through Streamlit Secrets (GEMINI_API_KEYS)."""
    raw = _secret("GEMINI_API_KEYS")
    out: list[str] = []
    if isinstance(raw, (list, tuple)):
        out = [str(k).strip() for k in raw if str(k).strip()]
    elif isinstance(raw, str):
        out = [k.strip() for k in raw.replace(",", "\n").splitlines() if k.strip()]
    single = _secret("GEMINI_API_KEY")
    if single and str(single).strip() not in out:
        out.insert(0, str(single).strip())
    return out


def all_keys() -> list[str]:
    merged: list[str] = []
    for k in secret_keys() + list(SS["keys"]):
        if k and k not in merged:
            merged.append(k)
    return merged


def make_pool() -> UnlimitedKeyPool | None:
    keys = all_keys()
    return UnlimitedKeyPool(keys) if keys else None


def set_current(path: str, label: str) -> None:
    """Send a video/part into Recap mode (Splitter → Studio transfer)."""
    SS["current_path"] = path
    SS["current_label"] = label
    SS["dialogues"] = []
    SS["edited_dialogues"] = []
    SS["script_confirmed"] = False
    SS["outputs"] = None
    SS["script_version"] += 1


def current_info() -> dict | None:
    if SS["current_path"] and Path(SS["current_path"]).exists():
        try:
            return get_video_info(SS["current_path"])
        except Exception:
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────
# Header + stepper
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎬 Recap Studio MM — Free Edition</h1>
  <p>ဘာသာစကားမရွေး ဗီဒီယို → သဘာဝကျတဲ့ 🇲🇲 မြန်မာ / 🇬🇧 English Recap —
     Gemini Free Keys အများကြီး သုံးနိုင်ပြီး၊ Script ကို အရင်ပြင်ပြီးမှ Render လုပ်ပါတယ်။</p>
</div>
""", unsafe_allow_html=True)

_steps = [
    ("1️⃣ ဗီဒီယို", bool(SS["video_path"])),
    ("2️⃣ အပိုင်းခွဲ", bool(SS["parts"])),
    ("3️⃣ Recap Mode", bool(SS["current_path"])),
    ("4️⃣ Script ထုတ်", bool(SS["dialogues"])),
    ("5️⃣ Script ပြင်", bool(SS["script_confirmed"])),
    ("6️⃣ Render", bool(SS["outputs"])),
]
_first_open = next((i for i, (_, d) in enumerate(_steps) if not d), len(_steps) - 1)
_pills = "".join(
    f'<span class="pill {"done" if d else ("active" if i == _first_open else "")}">{name}</span>'
    for i, (name, d) in enumerate(_steps))
st.markdown(f'<div class="stepbar">{_pills}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Sidebar — 🔑 API Key Pool
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 Gemini API Keys")
    st.caption("AI Studio (aistudio.google.com) က Free key တွေ တစ်ကြောင်းကို တစ်ခုစီ ထည့်ပါ။ "
               "အကန့်အသတ် မရှိပါ — key တစ်ခုက quota ပြည့်ရင် နောက်တစ်ခုကို အလိုအလျောက် ပြောင်းပေးပါမယ်။ "
               "⚠️ Project မတူတဲ့ key တွေမှ quota တကယ် တိုးပါတယ်။")
    _txt = st.text_area("Keys (တစ်ကြောင်း = key တစ်ခု)", height=150,
                        value="\n".join(SS["keys"]),
                        placeholder="AIza...\nAIza...\nAIza...",
                        label_visibility="collapsed")
    c1, c2 = st.columns(2)
    if c1.button("💾 Keys သိမ်းမည်", use_container_width=True):
        SS["keys"] = [k.strip() for k in _txt.splitlines() if k.strip()]
        st.success(f"✅ Key {len(SS['keys'])} ခု သိမ်းပြီးပါပြီ")
    if c2.button("🧪 အားလုံး စမ်းမည်", use_container_width=True):
        _test_keys = [k.strip() for k in _txt.splitlines() if k.strip()]
        if not _test_keys:
            st.warning("အရင် key ထည့်ပါ")
        else:
            with st.spinner("Keys စမ်းနေပါသည်…"):
                for i, k in enumerate(_test_keys, 1):
                    ok, msg = validate_key(k)
                    st.write(f"Key #{i} `{mask(k)}` — {msg}")
    _n = len(all_keys())
    if _secret("GEMINI_API_KEYS") or _secret("GEMINI_API_KEY"):
        st.caption(f"🔐 Secrets မှ key {len(secret_keys())} ခု ပါဝင်ပြီး — စုစုပေါင်း **{_n} ခု** အသင့်ရှိပါသည်")
    elif _n:
        st.caption(f"စုစုပေါင်း **{_n} ခု** အသင့်ရှိပါသည် (ဒီ browser session အတွင်းသာ သိမ်းထားပါသည်)")
    else:
        st.warning("Key မထည့်ရသေးပါ — Demo Script Mode နဲ့တော့ စမ်းနိုင်ပါတယ်")

    st.divider()
    st.markdown("## ⚙️ Engine")
    model = st.selectbox("Gemini Model",
                         ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
                         help="Flash = အရည်အသွေး ကောင်း • Flash-Lite = quota ချွေတာ၊ မြန်")
    demo_script = st.checkbox("🎭 Demo Script Mode (Key မလိုဘဲ စမ်းရန်)", value=False)
    st.caption("Free stack: Gemini Free Tier • Edge TTS • FFmpeg • yt-dlp — "
               "ပိုက်ဆံပေးရတာ တစ်ခုမှ မပါဝင်ပါ။")


# ─────────────────────────────────────────────────────────────────────────
# 1️⃣ Video input
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 1️⃣ ဗီဒီယို တင်ပါ / လင့်ခ် ထည့်ပါ")
_tab_up, _tab_link = st.tabs(["📤 ဖိုင် တင်ရန်", "🔗 YouTube / TikTok လင့်ခ်"])

with _tab_up:
    _up = st.file_uploader("MP4 / MOV / MKV / WEBM", type=["mp4", "mov", "mkv", "webm"],
                           label_visibility="collapsed")
    if _up is not None:
        _sig = f"{_up.name}:{_up.size}"
        if SS.get("_up_sig") != _sig:
            _dest = WORK_DIR / f"upload_{uuid.uuid4().hex[:8]}_{Path(_up.name).name}"
            _dest.write_bytes(_up.getbuffer())
            SS["_up_sig"] = _sig
            SS["video_path"] = str(_dest)
            SS["parts"] = []
            try:
                SS["video_info"] = get_video_info(str(_dest))
            except Exception as exc:
                st.error(f"ဗီဒီယို ဖတ်၍ မရပါ: {exc}")
            st.rerun()

with _tab_link:
    _url = st.text_input("လင့်ခ်", placeholder="https://youtu.be/…  (သို့)  TikTok link",
                         label_visibility="collapsed")
    if st.button("⬇️ လင့်ခ်မှ ဒေါင်းလုဒ် လုပ်မည်", disabled=not _url.strip()):
        try:
            import yt_dlp
            _dest = WORK_DIR / f"link_{uuid.uuid4().hex[:8]}.mp4"
            with st.spinner("ဒေါင်းလုဒ် လုပ်နေပါသည်… (လင့်ခ် အရွယ်အစားပေါ် မူတည်ပြီး ကြာနိုင်ပါသည်)"):
                with yt_dlp.YoutubeDL({
                    "outtmpl": str(_dest.with_suffix(".%(ext)s")),
                    "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
                    "merge_output_format": "mp4", "quiet": True, "no_warnings": True,
                }) as ydl:
                    _info = ydl.extract_info(_url.strip(), download=True)
                    _real = Path(ydl.prepare_filename(_info)).with_suffix(".mp4")
                    if _real.exists() and _real != _dest:
                        _real.replace(_dest)
            SS["video_path"] = str(_dest)
            SS["parts"] = []
            SS["video_info"] = get_video_info(str(_dest))
            st.rerun()
        except Exception as exc:
            st.error(f"ဒေါင်းလုဒ် မအောင်မြင်ပါ: {exc}")

if SS["video_path"] and Path(SS["video_path"]).exists():
    _vi = SS["video_info"] or {}
    _cols = st.columns([2, 1, 1, 1])
    with _cols[0]:
        st.video(SS["video_path"])
    _cols[1].metric("⏱️ ကြာချိန်", human_time(float(_vi.get("duration", 0))))
    _cols[2].metric("🖼️ အရွယ်အစား", f"{_vi.get('width', '?')}×{_vi.get('height', '?')}")
    _cols[3].metric("💾 ဖိုင်ဆိုဒ်", human_bytes(int(_vi.get("size", Path(SS['video_path']).stat().st_size))))
    if st.button("🎬 ဒီဗီဒီယို အပြည့်ကို Recap Mode သို့ ပို့မည်", type="primary"):
        set_current(SS["video_path"], "ဗီဒီယို အပြည့်")
        st.success("✅ Recap Mode သို့ ပို့ပြီးပါပြီ — အောက်က အဆင့် 3️⃣ ကို ဆက်လုပ်ပါ")
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# 2️⃣ Long Video Splitter → Recap Mode transfer
# ─────────────────────────────────────────────────────────────────────────
if SS["video_path"] and Path(SS["video_path"]).exists():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 2️⃣ ✂️ Long Video Splitter — အပိုင်းခွဲပြီး Recap ပို့ရန်")
    st.caption("ဗီဒီယို ရှည်လျှင် အပိုင်းခွဲပြီး တစ်ပိုင်းချင်း Recap လုပ်တာက Streamlit Cloud မှာ "
               "အတည်ငြိမ်ဆုံး နည်းလမ်း ဖြစ်ပါတယ်။ အပိုင်းတိုင်းမှာ **🎬 Recap ပို့မည်** ခလုတ် ပါပါတယ်။")
    _sc1, _sc2 = st.columns([1, 2])
    _slice_min = _sc1.selectbox("အပိုင်း တစ်ပိုင်းရဲ့ အရှည်",
                                [3, 5, 8, 10, 15], index=2, format_func=lambda m: f"{m} မိနစ်")
    _sc2.write("")
    if _sc2.button("✂️ အပိုင်းခွဲထုတ်မည်", type="primary"):
        try:
            _pipe = RecapPipeline(None)
            _bar = st.progress(0.0, "ခွဲထုတ်နေပါသည်…")
            _parts = _pipe.split_video(
                SS["video_path"], int(_slice_min) * 60, "original",
                progress=lambda p, m: _bar.progress(min(1.0, p / 100.0), m),
                out_dir=WORK_DIR / "parts")
            SS["parts"] = _parts
            _bar.progress(1.0, f"✅ အပိုင်း {len(_parts)} ပိုင်း ရပါပြီ")
        except Exception as exc:
            st.error(f"ခွဲထုတ် မအောင်မြင်ပါ: {exc}")
    for _p in SS["parts"]:
        _full = str(recap_config.resolve(_p["path"])) if not Path(_p["path"]).is_absolute() else _p["path"]
        _pc = st.columns([3, 1, 1, 2])
        _pc[0].write(f"**အပိုင်း {_p['part']}** — `{_p['filename']}`")
        _pc[1].write(human_time(float(_p.get("duration") or 0)))
        _pc[2].write(human_bytes(int(_p.get("size") or 0)))
        with _pc[3]:
            _cc = st.columns(2)
            if Path(_full).exists():
                _cc[0].download_button("⬇️ ဒေါင်း", Path(_full).read_bytes(),
                                       file_name=_p["filename"], mime="video/mp4",
                                       key=f"dl_part_{_p['part']}")
            if _cc[1].button("🎬 Recap ပို့မည်", key=f"send_part_{_p['part']}"):
                set_current(_full, f"အပိုင်း {_p['part']}")
                st.success(f"✅ အပိုင်း {_p['part']} ကို Recap Mode သို့ ပို့ပြီးပါပြီ — အဆင့် 3️⃣ ဆက်လုပ်ပါ")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# 3️⃣ Recap setup + AI analysis
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 3️⃣ 🎬 Recap Mode — ဆက်တင် ရွေးပြီး Script ထုတ်ပါ")
_ci = current_info()
if not SS["current_path"] or _ci is None:
    st.info("အရင်ဆုံး အဆင့် 1️⃣ (ဗီဒီယို အပြည့်) ဒါမှမဟုတ် အဆင့် 2️⃣ (အပိုင်း တစ်ပိုင်း) ကနေ "
            "**Recap Mode သို့ ပို့မည်** ခလုတ်ကို နှိပ်ပါ။")
else:
    st.success(f"📼 လက်ရှိ: **{SS['current_label']}** • ⏱️ {human_time(float(_ci['duration']))} • "
               f"{_ci['width']}×{_ci['height']}")
    _r1, _r2, _r3 = st.columns(3)
    with _r1:
        lang_label = st.segmented_control(
            "Recap ဘာသာစကား",
            ["🇲🇲 မြန်မာ", "🇬🇧 English"], default="🇲🇲 မြန်မာ")
        lang = "my" if lang_label and "မြန်မာ" in lang_label else "en"
    with _r2:
        mode_label = st.radio("Mode", ["🎬 Recap (ဇာတ်လမ်း ပြန်ပြော)", "🎭 Dubbing (စကား ဘာသာပြန်)"],
                              help="Recap = တစ်ချိန်လုံး narration • Dubbing = မူရင်း စကားပြောချိန်အတိုင်း ဘာသာပြန်")
        fill_mode = "continuous" if mode_label.startswith("🎬") else "dialogue"
    with _r3:
        _voices = VOICE_CATALOG[lang]
        voice_key = st.selectbox("🎙️ အသံ", list(_voices.keys()),
                                 format_func=lambda k: _voices[k]["name"])
    _r4, _r5, _r6 = st.columns(3)
    aspect = _r4.selectbox("📐 ထွက်မည့် ပုံစံ", ["မူရင်း", "16:9", "9:16", "1:1"])
    sub_on = _r5.toggle("📝 စာတန်းထိုး ထည့်မည်", value=True)
    mute_org = _r6.toggle("🔇 မူရင်းအသံ ပိတ်မည်", value=True,
                          help="ပိတ်ထားရင် မူရင်း စကားသံကို ဖယ်ပြီး Recap အသံပဲ ထားပါမယ်")

    st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
    _go = st.button("🧠 AI ဖြင့် Recap Script ထုတ်မည်", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if _go:
        SS["logs"] = []
        _pool = make_pool()
        if not demo_script and _pool is None:
            st.error("Gemini Key ထည့်ရန် လိုပါသည် — ဘယ်ဘက် Sidebar မှာ ထည့်ပါ၊ "
                     "ဒါမှမဟုတ် 🎭 Demo Script Mode ဖွင့်ပြီး စမ်းပါ။")
        else:
            _dur = float(_ci["duration"])
            _status = st.status("🧠 AI ခွဲခြမ်းနေပါသည်…", expanded=True)
            try:
                def _prog(pct, msg):
                    try:
                        _status.update(label=f"🧠 {msg}")
                    except Exception:
                        pass
                if demo_script:
                    _data = demo_timeline(_dur, lang)
                    _result = {"hook_line1": _data["hook_line1"], "hook_line2": _data["hook_line2"],
                               "dialogues": _data["dialogues"], "coverage": {}}
                else:
                    _result = extract_timeline(
                        "", SS["current_path"], _dur,
                        language=lang, mode="auto", fill_mode=fill_mode, model=model,
                        progress=_prog,
                        log_fn=lambda m: SS["logs"].append(str(m)),
                        key_ring=_pool)
                SS["dialogues"] = _result.get("dialogues", [])
                SS["hook1"] = _result.get("hook_line1", "")
                SS["hook2"] = _result.get("hook_line2", "")
                SS["coverage"] = _result.get("coverage")
                SS["edited_dialogues"] = []
                SS["script_confirmed"] = False
                SS["outputs"] = None
                SS["script_version"] += 1
                # keep the chosen settings for the render step
                SS["render_cfg"] = {"lang": lang, "voice_key": voice_key, "aspect": aspect,
                                    "sub_on": sub_on, "mute_org": mute_org,
                                    "fill_mode": fill_mode}
                _status.update(label=f"✅ Script လိုင်း {len(SS['dialogues'])} လိုင်း ရပါပြီ — "
                                     f"အဆင့် 4️⃣ မှာ ပြင်ပြီး အတည်ပြုပါ", state="complete")
                st.rerun()
            except Exception as exc:
                _status.update(label="❌ မအောင်မြင်ပါ", state="error")
                st.error(str(exc))
                if SS["logs"]:
                    with st.expander("Log ကြည့်ရန်"):
                        st.code("\n".join(SS["logs"][-40:]))
    if SS["coverage"]:
        st.caption(f"📊 Coverage: {SS['coverage'].get('coverage_percent', '—')}% • "
                   f"လိုင်း {SS['coverage'].get('lines', len(SS['dialogues']))} လိုင်း")
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# 4️⃣ Script Edit — the gate before rendering
# ─────────────────────────────────────────────────────────────────────────
if SS["dialogues"]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 4️⃣ ✍️ Recap Script ပြင်ရန် — အတည်ပြုပြီးမှ Render သွားပါမယ်")
    st.caption("လိုင်းတိုင်းရဲ့ စချိန်/ဆုံးချိန်/စာသားကို ပြင်နိုင်ပါတယ်။ လိုင်းအသစ် ထည့်/ဖျက်လည်း ရပါတယ်။ "
               "⚠️ ပြထားတဲ့ လိုင်းသည် window ထက် စာရှည်လွန်းလို့ အသံမြန်သွားနိုင်တဲ့ လိုင်းပါ။")
    _cfg = SS.get("render_cfg", {"lang": "my"})
    _lang = _cfg.get("lang", "my")
    _cps = chars_per_second(_lang)
    _h1, _h2 = st.columns(2)
    SS["hook1"] = _h1.text_input("🪝 Hook စာကြောင်း ၁", SS["hook1"])
    SS["hook2"] = _h2.text_input("🪝 Hook စာကြောင်း ၂", SS["hook2"])

    import pandas as pd
    _editor_key = f"script_editor_v{SS['script_version']}"
    if _editor_key not in SS:
        SS[_editor_key] = pd.DataFrame([
            {"start": float(d.get("start", 0)), "end": float(d.get("end", 0)),
             "speaker": str(d.get("speaker", "")), "text": str(d.get("text", ""))}
            for d in SS["dialogues"]])
    _df = SS[_editor_key]
    _over = sum(1 for _, r in _df.iterrows()
                if len(str(r["text"])) / _cps > max(0.1, float(r["end"]) - float(r["start"])) * 1.12)
    _m = st.columns(3)
    _m[0].metric("📝 စာကြောင်း", len(_df))
    _m[1].metric("⚠️ ရှည်လွန်းသော လိုင်း", _over)
    _m[2].metric("🎙️ ပြောနှုန်း", f"{_cps} စာလုံး/စက္ကန့်")
    _edited = st.data_editor(
        _df, key=f"editor_widget_v{SS['script_version']}", num_rows="dynamic",
        use_container_width=True, height=420,
        column_config={
            "start": st.column_config.NumberColumn("စချိန် (s)", format="%.2f", min_value=0.0),
            "end": st.column_config.NumberColumn("ဆုံးချိန် (s)", format="%.2f", min_value=0.0),
            "speaker": st.column_config.TextColumn("ပြောသူ"),
            "text": st.column_config.TextColumn("Recap စာသား", width="large"),
        })
    _b1, _b2 = st.columns(2)
    if _b1.button("🔧 ရှည်နေသော စာကြောင်းများ အလိုအလျောက် ချုံ့မည်"):
        _rows = _edited.to_dict("records")
        _fixed, _n = fit_lines_to_windows(_rows, float(current_info()["duration"]), _lang,
                                          respect_end=(_cfg.get("fill_mode") == "dialogue"))
        SS[_editor_key] = pd.DataFrame(_fixed)
        st.success(f"✅ {_n} လိုင်းကို အချိန်ကိုက် ချုံ့ပြီးပါပြီ")
        st.rerun()
    st.markdown('<div class="okbtn">', unsafe_allow_html=True)
    if _b2.button("✅ ဒီ Script ကို အတည်ပြုမည် — Render အဆင့်သို့", use_container_width=True):
        _rows = []
        for _, r in _edited.iterrows():
            _text = str(r["text"]).strip()
            if not _text or _text == "nan":
                continue
            _s, _e = float(r["start"]), float(r["end"])
            if _e <= _s:
                _e = _s + 2.0
            _rows.append({"start": round(_s, 2), "end": round(_e, 2),
                          "speaker": str(r.get("speaker", "")), "text": _text})
        _rows.sort(key=lambda d: d["start"])
        if not _rows:
            st.error("စာကြောင်း တစ်ကြောင်းမှ မရှိပါ — အရင် စာထည့်ပါ")
        else:
            SS["edited_dialogues"] = _rows
            SS["script_confirmed"] = True
            st.success(f"✅ လိုင်း {len(_rows)} လိုင်း အတည်ပြုပြီးပါပြီ — အဆင့် 5️⃣ Render လုပ်နိုင်ပါပြီ")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# 5️⃣ Render — only after the script is confirmed
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 5️⃣ 🎬 Render — အသံသွင်း + ဗီဒီယို ထုတ်ရန်")
if not SS["script_confirmed"]:
    st.info("🔒 Script ကို အဆင့် 4️⃣ မှာ အတည်ပြုပြီးမှ Render လုပ်နိုင်ပါမယ်။ "
            "(Script မပြင်ဘဲ တန်းမထုတ်တဲ့ စနစ် — အသံ-ရုပ် လွဲတာ၊ စာမှားတာ ကို ကြိုပြင်နိုင်အောင်ပါ)")
else:
    _cfg = SS.get("render_cfg", {})
    _lang = _cfg.get("lang", "my")
    _info = current_info()
    _dur = float(_info["duration"]) if _info else 0.0
    st.success(f"📝 အတည်ပြုပြီး Script: **{len(SS['edited_dialogues'])} လိုင်း** • "
               f"ဘာသာစကား: **{'မြန်မာ' if _lang == 'my' else 'English'}** • "
               f"အသံ: **{VOICE_CATALOG[_lang][_cfg.get('voice_key', 'thiha' if _lang == 'my' else 'christopher')]['name']}**")
    st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
    _render = st.button("🎬 Render စတင်မည်", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if _render and _info:
        _job = WORK_DIR / f"job_{uuid.uuid4().hex[:8]}"
        _job.mkdir(parents=True, exist_ok=True)
        _status = st.status("🎙️ အသံ သွင်းနေပါသည်…", expanded=True)
        try:
            def _rp(pct, msg):
                try:
                    _status.update(label=str(msg))
                except Exception:
                    pass
            _voice_cfg = _voice_config(_lang, _cfg.get("voice_key", ""))
            _alt_cfg = _fallback_voice(_lang, _cfg.get("voice_key", ""))
            _engine = TTSEngine()
            _mix = _engine.build_narration(
                dialogues=SS["edited_dialogues"], voice_cfg=_voice_cfg, duration=_dur,
                work_dir=_job, tag="st", progress=_rp,
                fallback_voice_cfg=_alt_cfg,
                strict_timing=(_cfg.get("fill_mode") == "dialogue"))
            _status.update(label="🧵 အသံ ချိန်ညှိနေပါသည်…")
            _narration = _job / "narration.mp3"
            master_audio(_mix.audio_path, _narration, _dur,
                         loudnorm=recap_config.settings.loudness_normalize)
            _ass = None
            _srt = _job / "recap.srt"
            export_srt(SS["edited_dialogues"], _srt)
            _aspect = _cfg.get("aspect", "မူရင်း")
            if _aspect == "16:9":
                _size, _reframe = (1280, 720), "Center Crop"
            elif _aspect == "9:16":
                _size, _reframe = (720, 1280), "Smart Blur Background"
            elif _aspect == "1:1":
                _size, _reframe = (1080, 1080), "Smart Blur Background"
            else:
                _size, _reframe = (int(_info["width"]), int(_info["height"])), "Original (no reframe)"
            if _cfg.get("sub_on", True):
                _ass = _job / "recap.ass"
                build_ass(dialogues=SS["edited_dialogues"], duration=_dur, ass_path=_ass,
                          hook_line1=SS["hook1"], hook_line2=SS["hook2"],
                          font_size=42, v_margin=int(_size[1] * 0.22),
                          hex_color="#00F2FE", play_res=_size)
            _status.update(label="🎬 ဗီဒီယို Render လုပ်နေပါသည်… (အရှည်ပေါ် မူတည်ပြီး ကြာနိုင်ပါသည်)")
            _out = _job / "recap_final.mp4"
            _res = render_master(
                SS["current_path"], str(_narration), str(_out),
                ass_path=str(_ass) if _ass else None,
                reframe=_reframe, mute_original=bool(_cfg.get("mute_org", True)),
                quality="fast", target_size=_size, duration=_dur, progress=_rp)
            SS["outputs"] = {"mp4": str(_out), "srt": str(_srt), "mp3": str(_narration),
                              "size": _out.stat().st_size if _out.exists() else 0,
                              "warnings": list(_res.get("warnings", [])) + list(_mix.warnings)}
            _status.update(label="🎉 Render ပြီးပါပြီ!", state="complete")
            st.rerun()
        except Exception as exc:
            _status.update(label="❌ Render မအောင်မြင်ပါ", state="error")
            st.error(str(exc))

if SS["outputs"]:
    _o = SS["outputs"]
    st.markdown("#### 📥 ရလဒ် ဒေါင်းလုဒ်")
    if Path(_o["mp4"]).exists():
        st.video(_o["mp4"])
        _d = st.columns(3)
        _d[0].download_button("🎬 MP4 ဒေါင်းမည်", Path(_o["mp4"]).read_bytes(),
                              file_name="recap_final.mp4", mime="video/mp4", type="primary")
        if Path(_o["srt"]).exists():
            _d[1].download_button("📝 SRT ဒေါင်းမည်", Path(_o["srt"]).read_bytes(),
                                  file_name="recap.srt")
        if Path(_o["mp3"]).exists():
            _d[2].download_button("🎧 အသံ MP3 ဒေါင်းမည်", Path(_o["mp3"]).read_bytes(),
                                  file_name="narration.mp3", mime="audio/mpeg")
        st.caption(f"💾 MP4 ဆိုဒ်: {human_bytes(int(_o['size']))}")
    for _w in _o.get("warnings", []):
        st.markdown(f'<span class="warnline">⚠️ {_w}</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Recap Studio MM — Free Edition (Streamlit) • '
            'အခြေခံ engine: Recap Studio MM Pro v4.4 • Gemini Free + Edge TTS + FFmpeg</div>',
            unsafe_allow_html=True)
