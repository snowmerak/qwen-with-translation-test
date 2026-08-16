from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import pydantic_monty
from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    enabled: bool
    timeout_seconds: float
    max_memory_bytes: int
    max_recursion_depth: int
    max_code_chars: int
    max_output_bytes: int
    max_tool_rounds: int

    @classmethod
    def from_env(cls) -> "SandboxSettings":
        load_dotenv()
        return cls(
            enabled=_bool_env("PYTHON_TOOL_ENABLED", False),
            timeout_seconds=_float_env("PYTHON_SANDBOX_TIMEOUT", 5.0),
            max_memory_bytes=_int_env(
                "PYTHON_SANDBOX_MAX_MEMORY_BYTES", 32 * 1024 * 1024
            ),
            max_recursion_depth=_int_env(
                "PYTHON_SANDBOX_MAX_RECURSION", 100
            ),
            max_code_chars=_int_env("PYTHON_SANDBOX_MAX_CODE_CHARS", 20_000),
            max_output_bytes=_int_env("PYTHON_SANDBOX_MAX_OUTPUT_BYTES", 20_000),
            max_tool_rounds=_int_env("PYTHON_TOOL_MAX_ROUNDS", 4),
        )


@dataclass(frozen=True, slots=True)
class SandboxResult:
    success: bool
    result: Any
    stdout: str
    stderr: str
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MontyPythonSandbox:
    """Run a restricted Python subset without host I/O capabilities."""

    def __init__(self, settings: SandboxSettings) -> None:
        self.settings = settings

    def execute(self, code: str) -> SandboxResult:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Python code must be a non-empty string")
        if len(code) > self.settings.max_code_chars:
            raise ValueError(
                "Python code exceeds the configured character limit "
                f"({self.settings.max_code_chars})"
            )

        collector = pydantic_monty.CollectStreams(
            max_bytes=self.settings.max_output_bytes
        )
        limits: pydantic_monty.ResourceLimits = {
            "max_duration_secs": self.settings.timeout_seconds,
            "max_memory": self.settings.max_memory_bytes,
            "max_recursion_depth": self.settings.max_recursion_depth,
        }

        try:
            with pydantic_monty.Monty(
                min_processes=1,
                max_processes=1,
                request_timeout=self.settings.timeout_seconds + 1,
            ) as pool:
                with pool.checkout(
                    script_name="qwen_tool.py",
                    limits=limits,
                ) as session:
                    result = session.feed_run(code, print_callback=collector)
            stdout, stderr = _split_streams(collector.output)
            return SandboxResult(
                success=True,
                result=_json_safe(result),
                stdout=stdout,
                stderr=stderr,
                error=None,
            )
        except (pydantic_monty.MontyError, MemoryError, TimeoutError) as error:
            stdout, stderr = _split_streams(collector.output)
            return SandboxResult(
                success=False,
                result=None,
                stdout=stdout,
                stderr=stderr,
                error=f"{type(error).__name__}: {error}",
            )


def _split_streams(
    output: list[tuple[str, str]],
) -> tuple[str, str]:
    stdout = "".join(text for stream, text in output if stream == "stdout")
    stderr = "".join(text for stream, text in output if stream == "stderr")
    return stdout, stderr


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


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
