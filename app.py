"""Streamlit entrypoint for the multimodal local chat assistant."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import streamlit as st
import yaml
from langchain.memory import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder

from audio_handler import transcribe_audio
from html_templates import CSS, get_bot_template, get_user_template
from image_handler import handle_image
from llm_chains import load_normal_chain, load_pdf_chat_chain
from pdf_handler import add_documents_to_db
from utils import get_timestamp, load_chat_history_json, save_chat_history_json

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

NEW_SESSION = "new_session"
CHAT_HISTORY_KEY = "history"
CONFIG_PATH = Path("config.yaml")

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

CHAT_HISTORY_DIR = Path(os.getenv("CHAT_HISTORY_PATH", CONFIG["chat_history_path"]))


def initialize_session_state() -> None:
    """Initialize all Streamlit state fields used by the app."""
    defaults = {
        "session_key": NEW_SESSION,
        "send_input": False,
        "user_question": "",
        "new_session_key": None,
        "session_index_tracker": NEW_SESSION,
        "processed_pdfs": set(),
        "pdf_chat": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_chat_sessions() -> list[str]:
    """Return available chat sessions ordered from newest to oldest."""
    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    sessions = sorted(
        (item.name for item in CHAT_HISTORY_DIR.glob("*.json")),
        reverse=True,
    )
    return [NEW_SESSION, *sessions]


def get_session_path(session_key: str) -> Path:
    """Build an absolute path for a chat session file."""
    return CHAT_HISTORY_DIR / session_key


def load_chain(chat_history: StreamlitChatMessageHistory):
    """Load the correct chain based on sidebar mode."""
    if st.session_state.pdf_chat:
        return load_pdf_chat_chain(chat_history)
    return load_normal_chain(chat_history)


def clear_input_field() -> None:
    """Move text input into the queued user question field."""
    if st.session_state.user_question:
        return

    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""


def set_send_input() -> None:
    """Mark input as ready to be sent from the text box callback."""
    st.session_state.send_input = True
    clear_input_field()


def toggle_pdf_chat() -> None:
    """Enable PDF chat when documents are uploaded."""
    st.session_state.pdf_chat = True


def save_chat_history() -> None:
    """Persist the current conversation state to a session file."""
    history = st.session_state.get(CHAT_HISTORY_KEY, [])
    if not history:
        return

    if st.session_state.session_key == NEW_SESSION:
        if st.session_state.new_session_key is None:
            st.session_state.new_session_key = f"{get_timestamp()}.json"
        target_session_key = st.session_state.new_session_key
    else:
        target_session_key = st.session_state.session_key

    save_chat_history_json(history, str(get_session_path(target_session_key)))


def load_current_session_history() -> None:
    """Hydrate Streamlit chat state from the selected session file."""
    if st.session_state.session_key == NEW_SESSION:
        st.session_state.history = []
        return

    session_path = get_session_path(st.session_state.session_key)
    st.session_state.history = load_chat_history_json(str(session_path))


def update_session_selector(chat_sessions: list[str]) -> None:
    """Render and synchronize sidebar chat-session selection."""
    if st.session_state.session_key == NEW_SESSION and st.session_state.new_session_key:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    if st.session_state.session_key not in chat_sessions:
        st.session_state.session_key = NEW_SESSION

    if st.session_state.session_index_tracker not in chat_sessions:
        st.session_state.session_index_tracker = NEW_SESSION

    selected_index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox(
        "Select a chat session",
        chat_sessions,
        key="session_key",
        index=selected_index,
    )
    st.session_state.session_index_tracker = st.session_state.session_key


def process_uploaded_pdfs(uploaded_pdf_files: Iterable) -> None:
    """Index only newly uploaded PDFs to avoid duplicate ingestion."""
    if not uploaded_pdf_files:
        return

    new_pdfs = []
    for pdf_file in uploaded_pdf_files:
        pdf_signature = f"{pdf_file.name}:{pdf_file.size}"
        if pdf_signature in st.session_state.processed_pdfs:
            continue

        new_pdfs.append(pdf_file)
        st.session_state.processed_pdfs.add(pdf_signature)

    if not new_pdfs:
        return

    with st.spinner("Processing pdf..."):
        chunk_count = add_documents_to_db(new_pdfs)

    if chunk_count:
        st.sidebar.success(f"Indexed {chunk_count} chunks from {len(new_pdfs)} PDF file(s).")
    else:
        st.sidebar.warning("No extractable text found in uploaded PDF file(s).")


def run_audio_prompt(audio_bytes: bytes, chat_history: StreamlitChatMessageHistory, summarize: bool) -> None:
    """Transcribe audio and send it to the active chain."""
    transcribed_audio = transcribe_audio(audio_bytes).strip()
    if not transcribed_audio:
        return

    user_prompt = f"Summarize this text: {transcribed_audio}" if summarize else transcribed_audio
    llm_chain = load_chain(chat_history)
    llm_chain.run(user_prompt)


def process_uploaded_audio(uploaded_audio, chat_history: StreamlitChatMessageHistory) -> None:
    """Handle user-provided audio file uploads."""
    if not uploaded_audio:
        return

    try:
        run_audio_prompt(uploaded_audio.getvalue(), chat_history, summarize=True)
    except Exception as exc:  # pragma: no cover - depends on user uploads/runtime.
        LOGGER.warning("Audio upload processing failed: %s", exc)
        st.sidebar.warning("Unable to process uploaded audio.")


def process_voice_recording(voice_recording, chat_history: StreamlitChatMessageHistory) -> None:
    """Handle microphone recording input."""
    if not voice_recording:
        return

    recording_bytes = voice_recording.get("bytes")
    if not recording_bytes:
        return

    try:
        run_audio_prompt(recording_bytes, chat_history, summarize=False)
    except Exception as exc:  # pragma: no cover - depends on mic/runtime.
        LOGGER.warning("Voice recording processing failed: %s", exc)
        st.warning("Unable to process voice recording.")


def process_message_submission(send_requested: bool, uploaded_image, chat_history: StreamlitChatMessageHistory) -> None:
    """Process text/image message submission from button or text callback."""
    if not send_requested:
        return

    if uploaded_image:
        try:
            with st.spinner("Processing image..."):
                user_message = st.session_state.user_question or "Describe this image in detail please."
                st.session_state.user_question = ""
                llm_answer = handle_image(uploaded_image.getvalue(), user_message)
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_answer)
        except Exception as exc:  # pragma: no cover - depends on image/model runtime.
            LOGGER.warning("Image processing failed: %s", exc)
            st.warning("Unable to process uploaded image.")

    if st.session_state.user_question:
        try:
            llm_chain = load_chain(chat_history)
            llm_chain.run(st.session_state.user_question)
            st.session_state.user_question = ""
        except RuntimeError as exc:
            LOGGER.warning("LLM runtime error: %s", exc)
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - model/runtime dependent.
            LOGGER.exception("Unexpected error while generating response: %s", exc)
            st.error("The model failed while generating a response. Please retry.")

    st.session_state.send_input = False


def render_chat_history(chat_history: StreamlitChatMessageHistory) -> None:
    """Render the latest chat history in reverse chronological order."""
    if not chat_history.messages:
        return

    st.write("Chat History:")
    for message in reversed(chat_history.messages):
        if message.type == "human":
            st.write(get_user_template(message.content), unsafe_allow_html=True)
        else:
            st.write(get_bot_template(message.content), unsafe_allow_html=True)


def main() -> None:
    """Render and execute the Streamlit app lifecycle."""
    initialize_session_state()

    st.title("Multimodal Local Chat App")
    st.write(CSS, unsafe_allow_html=True)

    st.sidebar.title("Chat Sessions")
    chat_sessions = get_chat_sessions()
    update_session_selector(chat_sessions)
    st.sidebar.toggle("PDF Chat", key="pdf_chat", value=False)

    load_current_session_history()
    chat_history = StreamlitChatMessageHistory(key=CHAT_HISTORY_KEY)

    st.text_input("Type your message here", key="user_input", on_change=set_send_input)

    voice_column, send_column = st.columns(2)
    with voice_column:
        voice_recording = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=True,
        )
    with send_column:
        send_button = st.button("Send", key="send_button", on_click=clear_input_field)

    uploaded_audio = st.sidebar.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "ogg"],
    )
    uploaded_image = st.sidebar.file_uploader(
        "Upload an image file",
        type=["jpg", "jpeg", "png"],
    )
    uploaded_pdf_files = st.sidebar.file_uploader(
        "Upload a pdf file",
        accept_multiple_files=True,
        key="pdf_upload",
        type=["pdf"],
        on_change=toggle_pdf_chat,
    )

    process_uploaded_pdfs(uploaded_pdf_files)
    process_uploaded_audio(uploaded_audio, chat_history)
    process_voice_recording(voice_recording, chat_history)

    process_message_submission(
        send_requested=bool(send_button or st.session_state.send_input),
        uploaded_image=uploaded_image,
        chat_history=chat_history,
    )

    with st.container():
        render_chat_history(chat_history)

    save_chat_history()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - Streamlit renders this path.
        LOGGER.exception("Application crashed: %s", exc)
        raise
