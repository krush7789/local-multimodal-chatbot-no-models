"""Reusable chain builders for normal and PDF-grounded conversations."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
import yaml
from chromadb.config import Settings
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from prompt_templates import memory_prompt_template

CONFIG_PATH = Path("config.yaml")
VECTOR_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
VECTOR_COLLECTION_NAME = "pdfs"
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "2"))
MEMORY_WINDOW_SIZE = 3
NOOP_TELEMETRY_IMPL = "chroma_noop_telemetry.NoOpProductTelemetryClient"
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "2048"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
DEFAULT_BATCH_SIZE = int(os.getenv("LLAMA_BATCH_SIZE", "32"))
DEFAULT_THREADS = int(os.getenv("LLAMA_THREADS", str(max(1, min(2, os.cpu_count() or 1)))))
FALLBACK_CONTEXT_LENGTH = int(os.getenv("FALLBACK_CONTEXT_LENGTH", "1024"))
FALLBACK_MAX_NEW_TOKENS = int(os.getenv("FALLBACK_MAX_NEW_TOKENS", "128"))
FALLBACK_BATCH_SIZE = int(os.getenv("FALLBACK_BATCH_SIZE", "8"))
FALLBACK_THREADS = int(os.getenv("FALLBACK_THREADS", "1"))

LOGGER = logging.getLogger(__name__)
ACTIVE_LLM_OVERRIDES: dict[str, Any] | None = None

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)


def build_stable_model_config(
    raw_config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Clamp model settings to safer defaults for llama-cpp stability."""
    config = dict(raw_config or {})
    if overrides:
        config.update(overrides)
    context_length = max(512, min(int(config.get("context_length", MAX_CONTEXT_LENGTH)), MAX_CONTEXT_LENGTH))
    max_new_tokens = max(64, min(int(config.get("max_new_tokens", MAX_NEW_TOKENS)), MAX_NEW_TOKENS))
    batch_size = max(1, min(int(config.get("batch_size", DEFAULT_BATCH_SIZE)), 512))
    threads = max(1, int(config.get("threads", DEFAULT_THREADS)))
    temperature = float(config.get("temperature", 0))
    gpu_layers = int(config.get("gpu_layers", 0))

    return {
        "n_ctx": context_length,
        "max_tokens": max_new_tokens,
        "n_batch": batch_size,
        "n_threads": threads,
        "n_gpu_layers": gpu_layers,
        "temperature": temperature,
        "verbose": False,
    }


def create_llm(
    model_path: str = CONFIG["model_path"]["large"],
    model_config: dict = CONFIG["model_config"],
    overrides: dict[str, Any] | None = None,
) -> LlamaCpp:
    """Create a llama-cpp-backed local language model."""
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    stable_config = build_stable_model_config(model_config, overrides=overrides)
    return LlamaCpp(model_path=str(model_file), **stable_config)


@lru_cache(maxsize=1)
def get_default_llm() -> LlamaCpp:
    """Load and cache default llama-cpp model instance."""
    return create_llm(overrides=ACTIVE_LLM_OVERRIDES)


def switch_to_fallback_mode() -> dict[str, Any]:
    """Enable conservative llama-cpp config for subsequent model creation."""
    global ACTIVE_LLM_OVERRIDES
    ACTIVE_LLM_OVERRIDES = {
        "context_length": FALLBACK_CONTEXT_LENGTH,
        "max_new_tokens": FALLBACK_MAX_NEW_TOKENS,
        "batch_size": FALLBACK_BATCH_SIZE,
        "threads": FALLBACK_THREADS,
    }
    get_default_llm.cache_clear()
    return dict(ACTIVE_LLM_OVERRIDES)


@lru_cache(maxsize=1)
def create_embeddings(
    embeddings_path: str = CONFIG["embeddings_path"],
) -> HuggingFaceInstructEmbeddings:
    """Create and cache text embeddings model."""
    return HuggingFaceInstructEmbeddings(model_name=embeddings_path)


def create_chat_memory(chat_history) -> ConversationBufferWindowMemory:
    """Create bounded chat memory around Streamlit message history."""
    return ConversationBufferWindowMemory(
        memory_key="history",
        chat_memory=chat_history,
        k=MEMORY_WINDOW_SIZE,
    )


def create_prompt_from_template(template: str) -> PromptTemplate:
    """Create a LangChain prompt from plain template text."""
    return PromptTemplate.from_template(template)


def create_llm_chain(llm: LlamaCpp, chat_prompt: PromptTemplate, memory: ConversationBufferWindowMemory) -> LLMChain:
    """Create a conversational LLM chain with memory."""
    return LLMChain(llm=llm, prompt=chat_prompt, memory=memory)


def load_normal_chain(chat_history):
    """Create the default non-retrieval chat chain."""
    return ChatChain(chat_history)


def load_vectordb(embeddings: HuggingFaceInstructEmbeddings) -> Chroma:
    """Load the persistent Chroma collection for PDF retrieval."""
    persistent_client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR,
        settings=Settings(
            anonymized_telemetry=False,
            chroma_product_telemetry_impl=NOOP_TELEMETRY_IMPL,
        ),
    )
    return Chroma(
        client=persistent_client,
        collection_name=VECTOR_COLLECTION_NAME,
        embedding_function=embeddings,
    )


def load_pdf_chat_chain(chat_history):
    """Create the retrieval-enabled PDF chat chain."""
    return PdfChatChain(chat_history)


def load_retrieval_chain(llm: LlamaCpp, memory: ConversationBufferWindowMemory, vector_db: Chroma) -> RetrievalQA:
    """Build retrieval QA chain using collection-backed retriever."""
    return RetrievalQA.from_llm(
        llm=llm,
        memory=memory,
        retriever=vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
    )


class PdfChatChain:
    """Conversation chain that answers with PDF retrieval context."""

    def __init__(self, chat_history) -> None:
        self.memory = create_chat_memory(chat_history)
        self.vector_db = load_vectordb(create_embeddings())
        self.llm_chain = load_retrieval_chain(get_default_llm(), self.memory, self.vector_db)

    def _rebuild_with_fallback_llm(self) -> None:
        fallback_overrides = switch_to_fallback_mode()
        LOGGER.warning("Rebuilding PDF chain with conservative fallback llama-cpp settings.")
        fallback_llm = create_llm(overrides=fallback_overrides)
        self.llm_chain = load_retrieval_chain(fallback_llm, self.memory, self.vector_db)

    def run(self, user_input: str) -> str:
        try:
            return self.llm_chain.run(query=user_input)
        except OSError as exc:
            if "access violation" in str(exc).lower():
                self._rebuild_with_fallback_llm()
                try:
                    return self.llm_chain.run(query=user_input)
                except Exception as retry_exc:
                    raise RuntimeError(
                        "The local LLM backend crashed (native access violation), even after fallback settings. "
                        "Use a smaller GGUF model or run with fewer threads."
                    ) from retry_exc
            raise


class ChatChain:
    """Conversation chain without external retrieval."""

    def __init__(self, chat_history) -> None:
        self.memory = create_chat_memory(chat_history)
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(get_default_llm(), chat_prompt, self.memory)

    def _rebuild_with_fallback_llm(self) -> None:
        fallback_overrides = switch_to_fallback_mode()
        LOGGER.warning("Rebuilding chat chain with conservative fallback llama-cpp settings.")
        fallback_llm = create_llm(overrides=fallback_overrides)
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(fallback_llm, chat_prompt, self.memory)

    def run(self, user_input: str) -> str:
        try:
            return self.llm_chain.run(human_input=user_input)
        except OSError as exc:
            if "access violation" in str(exc).lower():
                self._rebuild_with_fallback_llm()
                try:
                    return self.llm_chain.run(human_input=user_input)
                except Exception as retry_exc:
                    raise RuntimeError(
                        "The local LLM backend crashed (native access violation), even after fallback settings. "
                        "Try a smaller GGUF model or reduce thread count."
                    ) from retry_exc
            raise
