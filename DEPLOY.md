# Deploying Recap Studio (Docker)

ဒီ repo မှာ `Dockerfile` ပါပြီးသား — Streamlit Cloud က throttle လုပ်တာ
မခံနိုင်ရင် အောက်က host တွေထဲက တစ်ခုခုမှာ ဒီ image ကိုပဲ run လို့ ရတယ်။
App က ဒီ image ထဲမှာပဲ အကုန် ပါတယ် (ffmpeg အပါ)။

## Option A — Hugging Face Spaces (FREE)

- Specs (free "CPU Basic"): **2 vCPU / 16 GB RAM / 50 GB disk** — Streamlit
  Cloud free (~1 GB RAM, shared CPU) ထက် RAM ၁၆ ဆ လောက် ပိုများ။
- 48 နာရီ မသုံးရင် sleep ဖြစ်တယ်၊ ပြန်ဖွင့်ရင် ခဏ စောင့်ရတယ်။

Steps:
1. huggingface.co → New Space → SDK: **Docker** → name ပေး → Create.
2. Space repo ထဲ ဒီ repo ရဲ့ ဖိုင်တွေ အကုန် တင်ပါ (GitHub ကနေ clone ပြီး
   HF repo ကို push ရင် ရတယ်)။
3. Build ပြီးရင် Space က app ကို port 7860 မှာ အလိုအလျောက် ဖွင့်ပေးတယ်
   (Dockerfile က `$PORT` ကို ဖတ်တယ်)။

## Option B — AWS EC2 (credit နဲ့ အမြန်ဆုံး)

Render အမြန်နှုန်း အတွက် dedicated CPU လိုတယ် — EC2 Compute Optimized
က အဲဒါပဲ။ **c7i.xlarge (4 vCPU / 8 GB)** ≈ $0.18/hr (US East, On-Demand).
မသုံးတဲ့ အချိန် instance ကို **Stop** ထားရင် compute ဖိုး မကျဘူး
(disk ဖိုး နည်းနည်းပဲ) — $100 credit က လနဲ့ ချီ ခံတယ်။

Steps:
1. EC2 → Launch instance → Ubuntu 24.04 → **c7i.xlarge** → storage 30 GB →
   Security group မှာ port **8501** (နဲ့ SSH 22) ဖွင့်ပါ → Launch.
2. SSH ဝင်ပြီး:
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   git clone https://github.com/htinkyawzaw2017-maker/recap-studio-streamlit.git
   cd recap-studio-streamlit
   sudo docker build -t recap-studio .
   sudo docker run -d --restart unless-stopped -p 8501:8501 --name recap recap-studio
   ```
3. Browser: `http://<instance-public-ip>:8501`

⚠️ သတိ: ဒီ app မှာ login မပါဘူး — URL သိသူ ဝင်သုံးလို့ ရတယ်။ ကိုယ်တိုင်
သုံးဖို့ဆို IP ကို လူသိ မခံပါနဲ့၊ ဒါမှမဟုတ် နောက်မှ HTTPS + password
(Caddy reverse proxy) ခံပါ။ Gemini key တွေက browser localStorage မှာပဲ
သိမ်းတာ — server ပေါ် မသိမ်းဘူး။

## Option C — AWS Lightsail (လစဉ် ပုံသေ)

- 2 vCPU / 8 GB = **$44/mo**, 4 vCPU / 16 GB = **$84/mo** (IPv4 bundle).
- တည်ဆောက်ပုံ EC2 နဲ့ အတူတူ (Docker တင်၊ အထက်က အဆင့်တွေ အတိုင်း)။
- ⚠️ Lightsail CPU က burstable — render အရှည်ကြီး ဆက်တိုက် လုပ်ရင်
  CPU credit ကုန်ပြီး နှေးသွားနိုင်။ Render အမြန်နှုန်း အဓိက ဆို
  Option B (EC2 c7i) က ပိုသေချာတယ်။
