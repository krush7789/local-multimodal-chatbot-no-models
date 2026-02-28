## HospitalBot

Local multimodal assistant built with Streamlit + LangChain, supporting:
- text chat with local LLM
- PDF-grounded retrieval chat
- image understanding (LLaVA via `llama-cpp`)
- audio transcription (Whisper)

## Model Used

- `Mistral-7B-Instruct-v0.3-Q6_K.gguf` (local chat model)

## Main Imports (Libraries)

- `streamlit`
- `langchain`
- `llama_cpp_python`
- `chromadb`
- `transformers`
- `sentence-transformers`

## Project Layout

- `app.py`: Streamlit application entrypoint.
- `llm_chains.py`: LLM/retrieval chain factories.
- `pdf_handler.py`: PDF extraction and vector indexing.
- `audio_handler.py`: Whisper transcription utilities.
- `image_handler.py`: LLaVA image-chat utilities.
- `utils.py`: chat-session persistence helpers.
- `html_templates.py`: chat UI CSS/HTML templates.
- `config.yaml`: model paths and runtime configuration.
- `scripts/bootstrap_models.py`: startup model bootstrap for deployment.
- `Dockerfile`, `start.sh`, `render.yaml`: deployment artifacts.

## Requirements

- Python 3.10+ (3.11/3.12 recommended)
- Windows/Linux/macOS
- Sufficient RAM for local model loading
- Model files present under `./models/`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Deployment (Docker)

1. Build the image:
```bash
docker build -t hospitalbot:latest .
```

2. Run the container:
```bash
docker run --rm -p 8501:8501 \
  -e BOOTSTRAP_MODELS=1 \
  -e HF_MODEL_REPO=bartowski/Mistral-7B-Instruct-v0.3-GGUF \
  -e HF_MODEL_FILE=Mistral-7B-Instruct-v0.3-Q6_K.gguf \
  -e CHROMA_DB_DIR=/app/data/chroma_db \
  -e CHAT_HISTORY_PATH=/app/data/chat_sessions \
  -v hospitalbot-data:/app/data \
  hospitalbot:latest
```

3. Open:
```text
http://localhost:8501
```

## Deployment (Render)

This repository includes `render.yaml` for one-click blueprint deploy.

1. Push this repository to GitHub.
2. In Render, create a new Blueprint and point it to the repo.
3. Render will read `render.yaml` and create:
- a Docker web service
- a persistent disk at `/app/data`
- environment variables for model bootstrap and storage paths

Notes:
- First deployment downloads the configured model; this can take time.
- The persistent disk preserves model cache, vector DB, and chat sessions.
- If your model repo is private, set `HF_TOKEN` in Render environment variables.

## Configuration

Edit `config.yaml` to customize:
- `model_path`: local `.gguf` model files
- `model_type`: ctransformers model family
- `embeddings_path`: embedding model id
- `model_config`: generation parameters
- `chat_history_path`: session JSON directory

## Notes

- PDF uploads are indexed into local `chroma_db/`.
- Uploaded PDFs are deduplicated per app session.
- Chat sessions are stored as JSON under `chat_sessions/`.
- In container deployment, `CHROMA_DB_DIR` and `CHAT_HISTORY_PATH` can override local paths.
- Image chat requires LLaVA model files (`VISION_MODEL_PATH` and `VISION_CLIP_MODEL_PATH`) if image mode is used.
