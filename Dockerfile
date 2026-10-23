# Recap Studio MM — Free Edition, portable image.
# Runs the same app as Streamlit Cloud on any Docker host:
# Hugging Face Spaces (Docker), AWS EC2 / Lightsail, a home server, …
FROM python:3.12-slim

# ffmpeg: video rendering / retime / subtitle burn (same role as packages.txt)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Job files live here (ephemeral — download finished videos promptly).
ENV RECAP_DATA_DIR=/app/data
RUN mkdir -p /app/data

EXPOSE 8501
# $PORT lets Hugging Face Spaces (7860) work without a rebuild.
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --server.maxUploadSize=2048"]
