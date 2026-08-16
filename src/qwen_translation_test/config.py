from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    base_url: str
    api_key: str
    hy_model: str
    qwen_model: str
    database_path: Path
    hy_max_tokens: int
    qwen_max_tokens: int
    qwen_temperature: float
    qwen_system_prompt: str
    pivot_language: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            base_url=os.getenv(
                "OPENAI_BASE_URL", "http://macstudio:11888/v1"
            ).rstrip("/"),
            api_key=os.getenv("OPENAI_API_KEY", "local"),
            hy_model=os.getenv("HY_MODEL", "Hy-MT2-30B-A3B-MLX-4bit"),
            qwen_model=os.getenv("QWEN_MODEL", "Qwen3.8-27B-4bit"),
            database_path=Path(os.getenv("DATABASE_PATH", "chat.db")),
            hy_max_tokens=_int_env("HY_MAX_TOKENS", 4096),
            qwen_max_tokens=_int_env("QWEN_MAX_TOKENS", 32768),
            qwen_temperature=_float_env("QWEN_TEMPERATURE", 0.7),
            qwen_system_prompt=os.getenv(
                "QWEN_SYSTEM_PROMPT",
                "Answer the user's request clearly and accurately in "
                "{pivot_language}.",
            ),
            pivot_language=normalize_pivot_language(
                os.getenv("PIVOT_LANGUAGE", "English")
            ),
        )


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {value!r}") from error


def normalize_pivot_language(value: str) -> str:
    aliases = {
        "en": "English",
        "english": "English",
        "영어": "English",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "chinese": "Chinese",
        "중국어": "Chinese",
    }
    normalized = aliases.get(value.strip().lower())
    if normalized is None:
        raise ValueError(
            "PIVOT_LANGUAGE must be English or Chinese, "
            f"got {value!r}"
        )
    return normalized
