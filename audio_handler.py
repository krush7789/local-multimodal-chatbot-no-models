"""Audio transcription utilities backed by Whisper."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import warnings

import librosa
import torch

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.utils\._pytree\._register_pytree_node is deprecated.*",
    category=UserWarning,
    module=r"transformers\.utils\.generic",
)

from transformers import pipeline

TARGET_SAMPLE_RATE = 16000
WHISPER_MODEL_NAME = "openai/whisper-small"


def convert_bytes_to_array(audio_bytes: bytes):
    """Decode raw audio bytes into a normalized waveform array."""
    if not audio_bytes:
        raise ValueError("Audio bytes are empty.")

    audio_stream = BytesIO(audio_bytes)
    audio_array, _ = librosa.load(audio_stream, sr=TARGET_SAMPLE_RATE)
    return audio_array


@lru_cache(maxsize=1)
def get_asr_pipeline():
    """Create and cache automatic speech recognition pipeline."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="automatic-speech-recognition",
        model=WHISPER_MODEL_NAME,
        chunk_length_s=30,
        device=device,
    )


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe raw audio bytes into text."""
    pipe = get_asr_pipeline()
    audio_array = convert_bytes_to_array(audio_bytes)
    prediction = pipe(audio_array, batch_size=1)
    return prediction.get("text", "").strip()
