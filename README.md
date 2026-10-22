# 🎬 Recap Studio MM — Free Edition (Streamlit)

ဘာသာစကားမရွေး ဗီဒီယိုတွေကို သဘာဝကျတဲ့ **🇲🇲 မြန်မာ / 🇬🇧 English Recap** အဖြစ် ပြောင်းပေးတဲ့
အခမဲ့ Webapp ပါ။ အခြေခံ engine သည် Recap Studio MM Pro v4.4 (Gemini video understanding,
zero-drift timing, Edge TTS) ဖြစ်ပြီး UI ကို Streamlit နဲ့ အသစ် ရေးထားပါတယ်။

## ✨ လုပ်ဆောင်ချက်များ

| အဆင့် | လုပ်ဆောင်ချက် |
|---|---|
| 🔑 | **Gemini Key Pool — အကန့်အသတ် မရှိ** (တစ်ကြောင်း တစ်ခု)၊ quota ပြည့်ရင် နောက် key သို့ အလိုအလျောက် ပြောင်း |
| 1️⃣ | ဗီဒီယို တင် / YouTube-TikTok လင့်ခ် ထည့် |
| 2️⃣ | **Long Video Splitter** — 3/5/8/10/15 မိနစ် အပိုင်းခွဲ၊ အပိုင်းတိုင်းမှာ **🎬 Recap ပို့မည်** (Transfer) ခလုတ် |
| 3️⃣ | Recap Setup — ဘာသာစကား 🇲🇲/🇬🇧, Recap/Dubbing mode, အသံ, ပုံစံ ရွေး → Gemini နဲ့ Script ထုတ် |
| 4️⃣ | **✍️ Script Editor** — စချိန်/ဆုံးချိန်/စာသား ပြင်၊ လိုင်း ထည့်/ဖျက်၊ ရှည်လွန်းတဲ့ လိုင်း ⚠️ အမှတ် + အလိုအလျောက် ချုံ့ခလုတ်။ **အတည်ပြုပြီးမှ Render ဖွင့်ပါတယ်** |
| 5️⃣ | Render — Edge TTS အသံသွင်း → timeline တိကျစွာ ချိန် → စာတန်းထိုး → MP4 + SRT + MP3 ဒေါင်းလုဒ် |

## 🆕 v1.3 (2026-10-22) — Font အတိအကျ + Mirror + Script ဟန်

- **မြန်မာစာ shaping အမှန်** — Noto Sans Myanmar ဖောင့်ကို base64 ပုံစံနဲ့ repo ထဲမှာ ထည့်ပြီး runtime မှာ ပြန် decode သုံးပါတယ် (Cloud မှာ fallback font ကြောင့် စာလုံး ပုံပျက်တဲ့ ပြဿနာ မရှိတော့ပါ)
- **🪞 Mirror Mode** — အဆင့် 3️⃣ မှာ toggle ထည့်ထားပါတယ် (ရုပ်ပဲ လှန်ပါတယ်၊ စာတန်းထိုး မှန်အတိုင်း ဖတ်လို့ရပါတယ်)
- **Script ဟန်** — AI ကို လူပြော စကားစစ်စစ်၊ ဇာတ်ဝင်ခန်း ကွက်တိ၊ generic စကား တားမြစ်ပြီး ရေးခိုင်းတဲ့ စည်းကမ်း အသစ်
- **အသံ-ရုပ် offset လျှော့ပြီး** — TTS clip တွေရဲ့ ခေါင်းပိုင်း encoder silence ကို ဖယ်လိုက်ပါတယ်
- **YouTube ဒေါင်းလုဒ်** — player client အမျိုးမျိုး (android/ios/tv/web) အလှည့်ကျ စမ်းပြီး မရမှ error�ြပါတယ်
- **Script Editor** — cell အလွတ်/NaN ကြောင့် crash မဖြစ်အောင် ကာကွယ်ပြီး၊ player ကိုလည်း ဗီဒီယို ratio အတိုင်း အရွယ်ချိန်ပါတယ်

## 🆕 v1.2 (2026-10-22) — Recap အသံ တောက်လျှောက်

- **Recap mode မှာ အသံ မပြတ်တော့ပါ** — စာကြောင်း ပြောပြီးသွားပေမယ့် ဇာတ်ဝင်ခန်း ကျန်နေသေးရင် အရင် စနစ်က အဲဒီ အပိုင်းကို အသံမပါဘဲ ထားခဲ့လို့ "ပြောလိုက် ရပ်လိုက်" ဖြစ်နေတာ။ အခု အသံပြီးရင် 0.45s ပဲ နားပြီး နောက်လိုင်းရဲ့ ဇာတ်ဝင်ခန်းကို တန်း ဆက်ပါတယ် (မလိုတဲ့ ရုပ်အပိုင်း ဖြတ်ချိန်ခြင်း)
- **Dubbing mode** ကတော့ မူရင်း အချိန်ကိုက် အတိအကျ ဆက်ထားပါတယ် (စကားပြောချိန်နဲ့ ကိုက်ဖို့)

## 🆕 v1.1 (2026-10-22) — အသံ 1.0x + Audio-Led Sync

- **အသံ အမြဲ သဘာဝနှုန်း 1.0x** — စာကြောင်း ရှည်လည်း အသံကို လုံးဝ မမြန်တော့ပါ (rate boost / atempo ဖျက်ပြီး)
- **အသံ-ရုပ် sync အတိအကျ** — စာကြောင်း ရှည်ရင် ရုပ်ကို အများဆုံး **0.9x** အထိ နှေး၊ မလောက်ရင် နောက်ဆုံး frame ရပ်ပြီး ဖြည့် (ဗီဒီယို ဘယ်လောက်ရှည်ရှည် မ drift ပါ)
- **စာတန်းထိုး ပြင်ပြီး** — ဖောင့် အရွယ်ကို frame နဲ့ အချိုးကျ ချိန်၊ Noto Sans Myanmar Unicode ဖောင့် သေချာစေပြီး၊ SRT ကို UTF-8 (BOM) နဲ့ ထုတ်
- **ဗီဒီယို ပြသပုံ** — player box ထဲမှာ ကိုယ့် aspect ratio အတိုင်းပဲ ပြပါတယ် (screen အပြည့် ဆွဲဆန့်ခြင်း မရှိပါ)
- **API Key** — browser localStorage မှာ သိမ်းလို့ page refresh လုပ်လည်း ပြန်ထည့်စရာ မလိုပါ (key သည် browser ထဲမှာပဲ နေပါသည်)
- **ခလုတ် lock** — အလုပ်လုပ်နေချိန် အခြားခလုတ်တွေ ပိတ်ထားလို့ မှားနှိပ်မိလည်း အလုပ် မပျက်ပါ
- **Workspace ရှင်းရန်** — sidebar မှာ 🧹 ခလုတ် + render ပြီးတိုင်း ယာယီဖိုင် အလိုအလျောက် ရှင်းပါတယ်

အသံများ (အခမဲ့ Edge TTS): မြန်မာ — Thiha (ကျား) / Nilar (မ) • English — Christopher / Jenny / Guy / Aria

## 🚀 Streamlit Cloud မှာ တင်နည်း (၅ မိနစ်ခန့်)

1. ဒီ folder တစ်ခုလုံးကို GitHub repo တစ်ခုထဲ push လုပ်ပါ
   (`streamlit_app.py` က repo root မှာ ရှိရပါမယ်)
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → repo ရွေး →
   Main file: `streamlit_app.py` → **Deploy**
3. App **Settings → Secrets** မှာ key တွေ ထည့်ပါ (မထည့်ဘဲ UI မှာလည်း ထည့်နိုင်ပါတယ်):

```toml
GEMINI_API_KEYS = ["AIza...key1...", "AIza...key2...", "AIza...key3..."]
```

⚠️ **Quota တကယ် တိုးချင်ရင်** key တွေကို **Google project မတူတူချင်း** ထုတ်ပါ။
Project တစ်ခုတည်းက key အများကြီးသည် quota တစ်ခုတည်းကို မျှသုံးနေတာ ဖြစ်ပါတယ်။

## 💻 Local မှာ run နည်း

```bash
pip install -r requirements.txt      # + system မှာ ffmpeg လိုပါတယ်
streamlit run streamlit_app.py       # http://localhost:8501
```

## ⚠️ Streamlit Cloud ရဲ့ ကန့်သတ်ချက်များ (အမှန်အတိုင်း)

- **RAM/CPU နည်းပါတယ်** — ဗီဒီယို ရှည်ရှည်ကို တစ်ခါတည်း render ရင် နှေး/ကျနိုင်ပါတယ်။
  ဒါကြောင့် Splitter နဲ့ **၈ မိနစ်အောက် အပိုင်းခွဲ** ပြီး တစ်ပိုင်းချင်း လုပ်ဖို့ အကြံပြုပါတယ်။
- **ဖိုင်တွေ မမြဲပါ** — app အိပ်/restart ဖြစ်ရင် upload/render ဖိုင်တွေ ပျောက်ပါတယ်။
  ပြီးတာနဲ့ ချက်ချင်း ဒေါင်းလုဒ် လုပ်ထားပါ။
- Upload limit ကို 1 GB ထားထားပေမယ့် Cloud ရဲ့ ကိုယ်ပိုင် ကန့်သတ်ချက် ရှိနိုင်ပါတယ် —
  ဖိုင်ကြီးရင် လင့်ခ် import (≤720p) က ပို အဆင်ပြေပါတယ်။
- Edge TTS သည် Microsoft ရဲ့ အခမဲ့ endpoint ဖြစ်လို့ တစ်ခါတလေ နှေးနိုင်/ပြန်စမ်းရနိုင်ပါတယ်။

## 🧪 စမ်းသပ်ချက်

```bash
RECAP_DATA_DIR=/tmp/recap_test RECAP_FAKE_TTS=1 RECAP_DEMO_MODE=1 \
  python tests/test_streamlit_core.py
# → key pool (25 keys) + failover, splitter, timeline, TTS, subtitle, render = 9 checks
```

## 📁 ဖွဲ့စည်းပုံ

```
streamlit_app.py     Streamlit UI (stepper workflow ၁–၅)
recapstudio/         engine (v4.4 မှ) + keys.py ထဲက UnlimitedKeyPool အသစ်
requirements.txt     Streamlit Cloud packages
packages.txt         ffmpeg (system)
.streamlit/          theme + upload config
dictionary.txt / pronunciation.txt   မြန်မာ အသံထွက်/စကားလုံး ပြင်ဆင်ချက် ဖိုင်များ
```
