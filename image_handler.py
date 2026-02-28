"""Image handling utilities backed by local LLaVA via llama-cpp."""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

VISION_MODEL_PATH = Path(os.getenv("VISION_MODEL_PATH", "./models/llava/llava_ggml-model-q5_k.gguf"))
CLIP_MODEL_PATH = Path(os.getenv("VISION_CLIP_MODEL_PATH", "./models/llava/mmproj-model-f16.gguf"))


def convert_bytes_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 data URL."""
    if not image_bytes:
        raise ValueError("Image bytes are empty.")

    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"


@lru_cache(maxsize=1)
def get_image_llm() -> Llama:
    """Create and cache the multimodal LLaVA model."""
    if not VISION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Vision model not found: {VISION_MODEL_PATH}")
    if not CLIP_MODEL_PATH.exists():
        raise FileNotFoundError(f"CLIP model not found: {CLIP_MODEL_PATH}")

    chat_handler = Llava15ChatHandler(clip_model_path=str(CLIP_MODEL_PATH))
    return Llama(
        model_path=str(VISION_MODEL_PATH),
        chat_handler=chat_handler,
        logits_all=True,
        n_ctx=1024,  # Increase as needed for larger image contexts.
    )


def handle_image(image_bytes: bytes, user_message: str) -> str:
    """Generate model response for an image and user prompt."""
    llm = get_image_llm()
    image_base64 = convert_bytes_to_base64(image_bytes)

    output = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant who perfectly describes images.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_base64}},
                    {"type": "text", "text": user_message},
                ],
            },
        ]
    )
    return output["choices"][0]["message"]["content"]


def convert_image_to_base64(image_path: str) -> str:
    """Read an image file and return a base64 data URL."""
    image_file_path = Path(image_path)
    return convert_bytes_to_base64(image_file_path.read_bytes())


if __name__ == "__main__":
    image_path = "Image26.jpg"
    image_base64 = convert_image_to_base64(image_path)
    Path("image.txt").write_text(image_base64, encoding="utf-8")
