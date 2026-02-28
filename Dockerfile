FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_DB_DIR=/app/data/chroma_db \
    CHAT_HISTORY_PATH=/app/data/chat_sessions \
    HF_HOME=/app/data/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libgomp1 \
    libsndfile1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data/chroma_db /app/data/chat_sessions /app/models && \
    chmod +x /app/start.sh

EXPOSE 8501

CMD ["./start.sh"]
