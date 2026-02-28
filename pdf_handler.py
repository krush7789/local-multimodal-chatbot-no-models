"""PDF text extraction and vector indexing helpers."""

from __future__ import annotations

import logging
from typing import Iterable

import pypdfium2
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from llm_chains import create_embeddings, load_vectordb

LOGGER = logging.getLogger(__name__)

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 50
CHUNK_SEPARATORS = ["\n", "\n\n"]


def get_pdf_texts(pdfs_bytes_list: Iterable) -> list[str]:
    """Extract text from a collection of uploaded PDF files."""
    texts: list[str] = []
    for pdf_file in pdfs_bytes_list:
        try:
            pdf_text = extract_text_from_pdf(pdf_file.getvalue()).strip()
        except Exception as exc:  # pragma: no cover - depends on user uploads.
            LOGGER.warning("Skipping unreadable PDF '%s': %s", getattr(pdf_file, "name", "<unknown>"), exc)
            continue

        if pdf_text:
            texts.append(pdf_text)

    return texts


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract full text from all pages of a PDF byte stream."""
    pdf_document = pypdfium2.PdfDocument(pdf_bytes)
    page_texts: list[str] = []

    try:
        for page_index in range(len(pdf_document)):
            page = pdf_document.get_page(page_index)
            try:
                text_page = page.get_textpage()
                try:
                    page_text = text_page.get_text_range().strip()
                    if page_text:
                        page_texts.append(page_text)
                finally:
                    text_page.close()
            finally:
                page.close()
    finally:
        pdf_document.close()

    return "\n".join(page_texts)


def get_text_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks for vector indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    return splitter.split_text(text)


def get_document_chunks(text_list: Iterable[str]) -> list[Document]:
    """Convert extracted texts into LangChain document chunks."""
    documents: list[Document] = []
    for text in text_list:
        documents.extend(Document(page_content=chunk) for chunk in get_text_chunks(text))
    return documents


def add_documents_to_db(pdfs_bytes: Iterable) -> int:
    """Extract, chunk, and add uploaded PDFs to the vector database."""
    if not pdfs_bytes:
        return 0

    texts = get_pdf_texts(pdfs_bytes)
    if not texts:
        return 0

    documents = get_document_chunks(texts)
    if not documents:
        return 0

    vector_db = load_vectordb(create_embeddings())
    vector_db.add_documents(documents)
    LOGGER.info("Documents indexed into vector DB: %s", len(documents))
    return len(documents)
