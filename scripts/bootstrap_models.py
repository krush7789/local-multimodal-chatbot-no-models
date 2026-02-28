"""Ensure required local model artifacts exist before app startup."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

LOGGER = logging.getLogger("bootstrap_models")

CONFIG_PATH = Path("config.yaml")
DEFAULT_MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"


def load_model_path_from_config() -> Path:
    """Read configured LLM path from config.yaml."""
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return Path(config["model_path"]["large"])


def resolve_target_model_path() -> Path:
    """Resolve target model path using env override when provided."""
    local_model_path = os.getenv("LOCAL_MODEL_PATH")
    if local_model_path:
        return Path(local_model_path)

    return load_model_path_from_config()


def ensure_model_present(dry_run: bool = False) -> Path:
    """Ensure configured model file is available on local disk."""
    model_path = resolve_target_model_path()
    if model_path.exists():
        LOGGER.info("Model already present: %s", model_path)
        return model_path

    if os.getenv("SKIP_MODEL_DOWNLOAD", "0") == "1":
        raise FileNotFoundError(
            f"Configured model file is missing and downloads are disabled: {model_path}"
        )

    model_repo = os.getenv("HF_MODEL_REPO", DEFAULT_MODEL_REPO)
    model_filename = os.getenv("HF_MODEL_FILE", model_path.name)
    hf_token = os.getenv("HF_TOKEN")

    LOGGER.info("Model not found at %s", model_path)
    LOGGER.info("Preparing download from repo=%s file=%s", model_repo, model_filename)

    if dry_run:
        LOGGER.info("Dry run enabled. Skipping download.")
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(
        hf_hub_download(
            repo_id=model_repo,
            filename=model_filename,
            local_dir=str(model_path.parent),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
    )

    if downloaded_path.name != model_path.name:
        downloaded_path.rename(model_path)
        LOGGER.info("Renamed downloaded model %s -> %s", downloaded_path, model_path)
    else:
        LOGGER.info("Downloaded model to %s", downloaded_path)

    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap required local model files.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths without downloading.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    ensure_model_present(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
