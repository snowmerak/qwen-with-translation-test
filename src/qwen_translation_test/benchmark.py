from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import OpenAIError

from .config import (
    SUPPORTED_PIVOT_LANGUAGES,
    Settings,
    normalize_pivot_language,
)
from .database import ChatDatabase
from .pipeline import TranslationChatPipeline, TurnResult


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    category: str
    request_ko: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare pivot translations for Korean prompts."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/benchmark_cases.json"),
        help="JSON file containing benchmark cases.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=4,
        help="Repetitions per case (default: 4).",
    )
    parser.add_argument(
        "--pivot-languages",
        nargs="+",
        type=normalize_pivot_language,
        choices=SUPPORTED_PIVOT_LANGUAGES,
        default=list(SUPPORTED_PIVOT_LANGUAGES),
        help=(
            "Pivot languages to compare (default: English Chinese Korean "
            "Japanese)."
        ),
    )
    parser.add_argument(
        "--include-bypass",
        action="store_true",
        help="Also benchmark direct Korean Qwen calls without Hy-MT2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pivot_benchmark.jsonl"),
        help="Append-only JSONL output; completed records are skipped on rerun.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("benchmark.db"),
        help="SQLite database used for benchmark conversations.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("results/pivot_benchmark.csv"),
        help="Side-by-side CSV output for manual quality evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N comparison records (useful for a smoke test).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Legacy override for both Hy and Qwen output limits.",
    )
    parser.add_argument(
        "--hy-max-tokens",
        type=int,
        help="Override the Hy translation output limit.",
    )
    parser.add_argument(
        "--qwen-max-tokens",
        type=int,
        help="Override the Qwen output limit, including thinking tokens.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.runs < 1:
        raise SystemExit("error: --runs must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("error: --limit must be at least 1")
    if len(set(args.pivot_languages)) != len(args.pivot_languages):
        raise SystemExit("error: --pivot-languages must not contain duplicates")
    for name in ("max_tokens", "hy_max_tokens", "qwen_max_tokens"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"error: --{name.replace('_', '-')} must be at least 1")

    try:
        cases = load_cases(args.cases)
        comparison_modes = [*args.pivot_languages]
        if args.include_bypass:
            comparison_modes.append("Bypass")
        completed = load_completed_keys(args.output, comparison_modes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        base_settings = Settings.from_env()
        settings = replace(
            base_settings,
            hy_max_tokens=(
                args.hy_max_tokens
                if args.hy_max_tokens is not None
                else args.max_tokens
                if args.max_tokens is not None
                else base_settings.hy_max_tokens
            ),
            qwen_max_tokens=(
                args.qwen_max_tokens
                if args.qwen_max_tokens is not None
                else args.max_tokens
                if args.max_tokens is not None
                else base_settings.qwen_max_tokens
            ),
        )

        all_records = [
            (ordinal, case, repetition)
            for ordinal, (case, repetition) in enumerate(
                (
                    (case, repetition)
                    for case in cases
                    for repetition in range(1, args.runs + 1)
                ),
                start=1,
            )
        ]
        scheduled = [
            (ordinal, case, repetition)
            for ordinal, case, repetition in all_records
            if (case.id, repetition) not in completed
        ]
        if args.limit is not None:
            scheduled = scheduled[: args.limit]
        if not scheduled:
            export_csv(args.output, args.csv_output)
            print("No records to run; all requested records already exist.")
            return

        with ChatDatabase(args.database) as database:
            pipeline = TranslationChatPipeline(settings, database)
            for position, (ordinal, case, repetition) in enumerate(
                scheduled, start=1
            ):
                record = run_comparison_record(
                    pipeline,
                    database,
                    case,
                    repetition,
                    sequence=ordinal,
                    pivot_languages=args.pivot_languages,
                    include_bypass=args.include_bypass,
                )
                append_record(args.output, record)
                print(
                    f"[{position}/{len(scheduled)}] "
                    f"{case.id} run {repetition} saved",
                    flush=True,
                )
        export_csv(args.output, args.csv_output)
    except (OpenAIError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


def load_cases(path: Path) -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Benchmark cases must be a JSON array")
    cases = [BenchmarkCase(**item) for item in raw]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Benchmark case IDs must be unique")
    return cases


def load_completed_keys(
    path: Path,
    comparison_modes: list[str] | None = None,
) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            record_modes = record.get(
                "comparison_modes", ["English", "Chinese"]
            )
            if comparison_modes is None or record_modes == comparison_modes:
                completed.add(
                    (str(record["case_id"]), int(record["repetition"]))
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid JSONL record at {path}:{line_number}"
            ) from error
    return completed


def run_comparison_record(
    pipeline: TranslationChatPipeline,
    database: ChatDatabase,
    case: BenchmarkCase,
    repetition: int,
    *,
    sequence: int,
    pivot_languages: list[str] | None = None,
    include_bypass: bool = False,
) -> dict[str, Any]:
    configured_modes = list(pivot_languages or SUPPORTED_PIVOT_LANGUAGES)
    if include_bypass:
        configured_modes.append("Bypass")
    modes = (
        configured_modes
        if sequence % 2 == 1
        else list(reversed(configured_modes))
    )
    results: dict[str, Any] = {}
    for mode in modes:
        bypass = mode == "Bypass"
        pivot_language = "Korean" if bypass else mode
        title = f"benchmark:{case.id}:run-{repetition}:{mode}"
        conversation_id = database.create_conversation(
            pivot_language,
            title,
            translation_bypass=bypass,
        )
        started = perf_counter()
        result = pipeline.run_turn(conversation_id, case.request_ko)
        results[mode.lower()] = serialize_result(
            result,
            conversation_id=conversation_id,
            total_seconds=perf_counter() - started,
            tool_executions=serialize_tool_executions(
                database.get_tool_executions(conversation_id)
            ),
        )

    return {
        "case_id": case.id,
        "category": case.category,
        "repetition": repetition,
        "request_ko": case.request_ko,
        "comparison_modes": configured_modes,
        "execution_order": modes,
        "python_tool_enabled": pipeline.python_sandbox is not None,
        "hy_max_tokens": pipeline.settings.hy_max_tokens,
        "qwen_max_tokens": pipeline.settings.qwen_max_tokens,
        "qwen_temperature": pipeline.settings.qwen_temperature,
        "created_at": datetime.now(UTC).isoformat(),
        **results,
    }


def serialize_result(
    result: TurnResult,
    *,
    conversation_id: int,
    total_seconds: float,
    tool_executions: list[dict[str, Any]],
) -> dict[str, Any]:
    data = asdict(result)
    data["conversation_id"] = conversation_id
    data["total_seconds"] = total_seconds
    data["tool_call_count"] = len(tool_executions)
    data["tool_executions"] = tool_executions
    data["thinking_observed"] = bool(data.get("assistant_reasoning"))
    for key, value in list(data.items()):
        if key.endswith("_seconds"):
            data[key] = round(float(value), 4)
    return data


def serialize_tool_executions(rows: list[Any]) -> list[dict[str, Any]]:
    executions: list[dict[str, Any]] = []
    for row in rows:
        arguments = _parse_json_field(row["arguments_json"])
        result = _parse_json_field(row["result_json"])
        executions.append(
            {
                "tool_call_id": row["tool_call_id"],
                "tool_name": row["tool_name"],
                "arguments": arguments,
                "result": result,
                "created_at": row["created_at"],
            }
        )
    return executions


def _parse_json_field(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_csv(jsonl_path: Path, csv_path: Path) -> None:
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    available_modes = [
        mode
        for mode in [*SUPPORTED_PIVOT_LANGUAGES, "Bypass"]
        if any(mode.lower() in record for record in records)
    ]
    fieldnames = [
        "case_id",
        "category",
        "repetition",
        "request_ko",
        "comparison_modes",
        *[
            f"{mode.lower()}_{field}"
            for mode in available_modes
            for field in (
                "input_pivot",
                "qwen_pivot",
                "qwen_reasoning",
                "final_ko",
                "total_seconds",
                "tool_call_count",
                "tool_codes",
                "tool_results",
                "final_chars",
                "quality_score",
            )
        ],
        "preferred_mode",
        "evaluator_notes",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {
                "case_id": record["case_id"],
                "category": record["category"],
                "repetition": record["repetition"],
                "request_ko": record["request_ko"],
                "comparison_modes": ",".join(
                    record.get("comparison_modes", ["English", "Chinese"])
                ),
                "preferred_mode": "",
                "evaluator_notes": "",
            }
            for mode in available_modes:
                key = mode.lower()
                result = record.get(key)
                if result is None:
                    continue
                final_ko = result["assistant_ko"]
                row.update(
                    {
                        f"{key}_input_pivot": result["user_pivot"],
                        f"{key}_qwen_pivot": result["assistant_pivot"],
                        f"{key}_qwen_reasoning": result.get(
                            "assistant_reasoning", ""
                        ),
                        f"{key}_final_ko": final_ko,
                        f"{key}_total_seconds": result["total_seconds"],
                        f"{key}_tool_call_count": result.get(
                            "tool_call_count", 0
                        ),
                        f"{key}_tool_codes": _tool_codes_for_csv(result),
                        f"{key}_tool_results": _tool_results_for_csv(result),
                        f"{key}_final_chars": len(final_ko),
                        f"{key}_quality_score": "",
                    }
                )
            writer.writerow(row)


def _tool_codes_for_csv(result: dict[str, Any]) -> str:
    codes = [
        execution.get("arguments", {}).get("code", "")
        for execution in result.get("tool_executions", [])
        if isinstance(execution.get("arguments"), dict)
    ]
    return json.dumps(codes, ensure_ascii=False)


def _tool_results_for_csv(result: dict[str, Any]) -> str:
    results = [
        execution.get("result")
        for execution in result.get("tool_executions", [])
    ]
    return json.dumps(results, ensure_ascii=False)


if __name__ == "__main__":
    main()
