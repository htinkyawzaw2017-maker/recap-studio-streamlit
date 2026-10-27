# Deploying Recap Studio (Docker)

ဒီ repo မှာ `Dockerfile` ပါပြီးသား — Streamlit Cloud က throttle လုပ်တာ
မခံနိုင်ရင် အောက်က host တွေထဲက တစ်ခုခုမှာ ဒီ image ကိုပဲ run လို့ ရတယ်။
App က ဒီ image ထဲမှာပဲ အကုန် ပါတယ် (ffmpeg အပါ)။

## Option A — Hugging Face Spaces (PRO REQUIRED since July 2026)

⚠️ **2026 ဇူလိုင်ကတည်းက HF က policy ပြောင်းလိုက်တယ်: Gradio နဲ့ Docker
Spaces က PRO subscription လိုတယ်** (တစ်လ $9) — free account နဲ့ Docker
Space အသစ် ဆောက်လို့ မရတော့ဘူး။ Static Space ပဲ free ကျန်တယ်၊
ဒါပေမဲ့ Static က HTML ပဲ serve လုပ်တာ — Streamlit/Python app လုံးဝ
run မရဘူး။ (Free personal account ရက် ၃၀+ ကျော် တွေမှာ Gradio
ZeroGPU Space ၂ ခု exception ရှိပေမဲ့ Gradio-only ဖြစ်လို့ ဒီ Docker
app နဲ့ မဆိုင်ဘူး။)

- PRO ရှိရင်: **CPU Basic (2 vCPU / 16 GB RAM / 50 GB disk)** hardware က
  အပိုကြေး မရှိ၊ Streamlit Cloud free (~1 GB RAM, shared CPU) ထက်
  RAM ၁၆ ဆ လောက် ပိုများ။
- 48 နာရီ မသုံးရင် sleep ဖြစ်တယ်၊ ပြန်ဖွင့်ရင် ခဏ စောင့်ရတယ်။

Steps:
1. **Account**: huggingface.co/join မှာ free account ဖွင့်ပါ။
   ပြီးရင် Settings → **Access Tokens** → New token (type: **Write**)
   တစ်ခု ထုတ်ထားပါ — push လုပ်တဲ့ အချိန် password နေရာမှာ သုံးရမယ်။
2. **Space**: huggingface.co → New Space → name (ဥပမာ `recap-studio`) →
   SDK: **Docker** → Visibility: Public → Create Space.
3. **ဖိုင် တင်ပါ** (Mac Terminal မှာ):
   ```bash
   git clone https://github.com/htinkyawzaw2017-maker/recap-studio-streamlit app
   git clone https://huggingface.co/spaces/<မင်း HF username>/recap-studio space
   rsync -a --exclude .git app/ space/
   cd space
   cp space_README.md README.md
   git add -A
   git commit -m "Recap Studio"
   git push        # username = HF username, password = အပေါ်က Write token
   ```
4. Space page မှာ Build လုပ်တာ စောင့်ပါ (ပထမ အကြိမ် မိနစ် အနည်းငယ်
   ကြာနိုင်) — ပြီးရင် Space URL ကနေ app သုံးလို့ ရပြီ။ Gemini key တွေ
   ပုံမှန် အတိုင်း app ထဲမှာပဲ ထည့်ပါ (server ပေါ် သိမ်းစရာ မလို)။

ပိုမြန်ချင်ရင်: Space Settings → Hardware → **CPU Upgrade**
(8 vCPU / 32 GB, HF docs အရ $0.03/hr) ကို နောက်မှ ပြောင်းလို့ ရတယ်။

## Option B — AWS EC2 (credit နဲ့ အမြန်ဆုံး)

Render အမြန်နှုန်း အတွက် dedicated CPU လိုတယ် — EC2 Compute Optimized
က အဲဒါပဲ။ **c7i.xlarge (4 vCPU / 8 GB)** ≈ $0.18/hr (US East, On-Demand).
မသုံးတဲ့ အချိန် instance ကို **Stop** ထားရင် compute ဖိုး မကျဘူး
(disk ဖိုး နည်းနည်းပဲ) — $100 credit က လနဲ့ ချီ ခံတယ်။

Steps:
1. EC2 → **Launch instance** →
   - Name: `recap-studio`
   - AMI: **Ubuntu Server 24.04 LTS** (64-bit x86)
   - Instance type: **c7i.xlarge** (4 vCPU / 8 GB)
   - Key pair: **Proceed without a key pair** (browser Connect သုံးမှာ)
   - Network: Security group အသစ် — **SSH (22)**: 0.0.0.0/0၊ **Custom TCP
     8501**: My IP (ကိုယ်တစ်ယောက်တည်း သုံးမှာမို့ app port ကို My IP ပဲ
     အလုံခြုံဆုံး။ ⚠️ SSH ကို My IP ပဲ ထားလို့ မရဘူး — EC2 Instance
     Connect က AWS ရဲ့ IP ကနေ ဝင်တာမို့ My IP rule နဲ့ Connect လုပ်လို့
     မရဘဲ "Failed to connect to your instance" error တက်တယ်။)
   - Storage: **30 GB gp3** → **Launch instance**.
2. Instance ကို ရွေး → **Connect** → **EC2 Instance Connect** →
   **Connect** (browser ထဲမှာ terminal ပွင့်လာမယ် — SSH key မလို).
3. Terminal မှာ အောက်က command တွေ တစ်ကြောင်းချင်း run ပါ
   (docker build က မိနစ် အနည်းငယ် ကြာမယ်):
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   git clone https://github.com/htinkyawzaw2017-maker/recap-studio-streamlit.git
   cd recap-studio-streamlit
   sudo docker build -t recap-studio .
   sudo docker run -d --restart unless-stopped -p 8501:8501 \
     -v recap-data:/app/data --name recap recap-studio
   sudo docker ps        # recap Running ဖြစ်ရမယ်
   # -v recap-data:/app/data က အရေးကြီးတယ်: container ဖျက်/အသစ်
   # ဆောက်လည်း upload/script/output တွေ volume ထဲမှာ ကျန်တယ်။
   # (volume မပါဘဲ run ထားတဲ့ container အဟောင်းရဲ့ data က
   # container နဲ့အတူ ပျက်သွားမယ် — rebuild မလုပ်ခင် ပြီးတဲ့
   # output ရှိရင် app ကနေ အရင် download လုပ်ထားပါ။)
   ```
4. Browser: `http://<instance-public-ip>:8501` (public IP က EC2
   instance စာမျက်နှာမှာ ကြည့်ပါ)။ App တက်လာရင် ပုံမှန် အတိုင်း
   Gemini key ထည့်ပြီး သုံးရုံပဲ။

💰 Credit ချွေတာနည်း: မသုံးတဲ့ အချိန် instance ကို **Stop** ထား —
compute ဖိုး မကျဘူး (disk ဖိုး နည်းနည်းပဲ)။ ပြန်သုံးချင် **Start** ပြန်
နှိပ်ရုံပဲ။ သတိထား: Stop/Start လုပ်တိုင်း public IP ပြောင်းတယ်
(Elastic IP တွဲထားရင် မပြောင်းဘူး — running instance နဲ့ တွဲထားရင်
Elastic IP က အခမဲ့)။ IP ပြောင်းသွားလို့ app ဖွင့်မရရင် EC2 မှာ IP
အသစ် ကြည့်ပါ။

⚠️ သတိ: ဒီ app မှာ login မပါဘူး — app port (8501) ကို My IP ပဲ
ဖွင့်ထားရင် ကိုယ်ကလွဲပြီး ဘယ်သူမှ ဝင်မရဘူး။ Gemini key တွေက browser
localStorage မှာပဲ သိမ်းတာ — server ပေါ် မသိမ်းဘူး။

### Connect မရရင် ("Failed to connect to your instance")

1. **စောင့်ကြည့်**: instance ခုမှ launch ရင် boot မပြီးသေးလို့ ဖြစ်တတ်တယ်။
   EC2 → Instances မှာ State **Running** ဖြစ်ပြီး Status check
   **2/2 checks passed** ဖြစ်မှ Connect ပြန်နှိပ်ပါ (၂-၃ မိနစ်)။
2. **Security group စစ်**: SSH (22) rule ရဲ့ Source က **0.0.0.0/0**
   ဖြစ်ရမယ်။ "My IP" ပဲ ထားရင် EC2 Instance Connect က AWS IP ကနေ
   ဝင်လို့ မရဘူး — ဒါ အဖြစ်များဆုံး အကြောင်းရင်းပဲ။
3. **နောက်နည်း (key pair)**: လုံခြုံချင်ရင် launch တုန်းက key pair
   တစ်ခုလုပ် (`.pem` download လုပ်) → Mac Terminal ကနေ:
   ```bash
   chmod 400 recap-key.pem
   ssh -i recap-key.pem ubuntu@<public-ip>
   ```
   ဒီနည်းက SSH rule ကို My IP ပဲ ထားလည်း ရတယ်။

## Option C — AWS Lightsail (လစဉ် ပုံသေ)

- 2 vCPU / 8 GB = **$44/mo**, 4 vCPU / 16 GB = **$84/mo** (IPv4 bundle).
- တည်ဆောက်ပုံ EC2 နဲ့ အတူတူ (Docker တင်၊ အထက်က အဆင့်တွေ အတိုင်း)။
- ⚠️ Lightsail CPU က burstable — render အရှည်ကြီး ဆက်တိုက် လုပ်ရင်
  CPU credit ကုန်ပြီး နှေးသွားနိုင်။ Render အမြန်နှုန်း အဓိက ဆို
  Option B (EC2 c7i) က ပိုသေချာတယ်။
