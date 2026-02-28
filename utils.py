"""Shared utility helpers for session persistence and timestamps."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from langchain.schema.messages import AIMessage, HumanMessage


def save_chat_history_json(chat_history: Iterable, file_path: str) -> None:
    """Persist chat message objects to JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_data = [message.dict() for message in chat_history]
    path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chat_history_json(file_path: str) -> list:
    """Load persisted chat messages as LangChain message instances."""
    path = Path(file_path)
    if not path.exists():
        return []

    try:
        json_data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    messages = []
    for message in json_data:
        message_type = message.get("type")
        message_content = message.get("content", "")
        additional_kwargs = message.get("additional_kwargs", {})

        if message_type == "human":
            messages.append(HumanMessage(content=message_content, additional_kwargs=additional_kwargs))
        elif message_type == "ai":
            messages.append(AIMessage(content=message_content, additional_kwargs=additional_kwargs))

    return messages


def get_timestamp() -> str:
    """Generate timestamp string suitable for file naming."""
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
