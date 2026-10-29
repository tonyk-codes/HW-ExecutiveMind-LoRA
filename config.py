"""Configuration loading for local runs and Hugging Face Spaces.

Environment variables take precedence over values in the optional local JSON
file. Secrets can therefore stay in Space Secrets or a local ignored file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppConfig:
    base_model: str = "Qwen/Qwen3.6-27B"
    lora_adapter: str = "tony-dev1/HW-ExecutiveMind-LoRA-Qwen3.6-27B"
    fal_key: str = ""
    hf_token: str = ""
    tts_model: str = "fal-ai/minimax/speech-02-turbo"
    model_load_mode: str = "bf16"
    merge_lora: bool = True
    max_queue_size: int = 10


def _read_local_config() -> dict[str, Any]:
    path = Path(os.getenv("CONFIG_FILE", "config.json"))
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a JSON object")
    return data


def _value(data: dict[str, Any], key: str, env_name: str, default: Any) -> Any:
    environment_value = os.getenv(env_name)
    if environment_value is not None and environment_value != "":
        return environment_value
    return data.get(key, default)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def load_config() -> AppConfig:
    data = _read_local_config()
    hf_token = _value(data, "hf_token", "HF_TOKEN", "")
    if not hf_token:
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
    model_load_mode = str(_value(data, "model_load_mode", "MODEL_LOAD_MODE", "bf16")).lower()
    if model_load_mode not in {"bf16", "4bit"}:
        raise ValueError("model_load_mode must be either 'bf16' or '4bit'")
    try:
        max_queue_size = int(_value(data, "max_queue_size", "MAX_QUEUE_SIZE", 10))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_queue_size must be an integer") from exc
    return AppConfig(
        base_model=str(_value(data, "base_model", "BASE_MODEL", "Qwen/Qwen3.6-27B")),
        lora_adapter=str(_value(data, "lora_adapter", "LORA_ADAPTER", "tony-dev1/HW-ExecutiveMind-LoRA-Qwen3.6-27B")),
        fal_key=str(_value(data, "fal_key", "FAL_KEY", "")),
        hf_token=str(hf_token),
        tts_model=str(_value(data, "tts_model", "FAL_TTS_MODEL", "fal-ai/minimax/speech-02-turbo")),
        model_load_mode=model_load_mode,
        merge_lora=_as_bool(_value(data, "merge_lora", "MERGE_LORA", True), True),
        max_queue_size=max(1, max_queue_size),
    )
