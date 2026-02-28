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

- PDF uploads are indexed into local `chroma_db/`.
- Uploaded PDFs are deduplicated per app session.
- Chat sessions are stored as JSON under `chat_sessions/`.
- In container deployment, `CHROMA_DB_DIR` and `CHAT_HISTORY_PATH` can override local paths.
- Image chat requires LLaVA model files (`VISION_MODEL_PATH` and `VISION_CLIP_MODEL_PATH`) if image mode is used.
