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

from .config import Settings
from .database import ChatDatabase
from .pipeline import TranslationChatPipeline, TurnResult


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    category: str
    request_ko: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired English/Chinese pivot comparisons for Korean prompts."
        )
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
        help="Run only the first N paired records (useful for a smoke test).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum output tokens for both Hy and Qwen (default: 1024).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.runs < 1:
        raise SystemExit("error: --runs must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("error: --limit must be at least 1")
    if args.max_tokens < 1:
        raise SystemExit("error: --max-tokens must be at least 1")

    try:
        cases = load_cases(args.cases)
        completed = load_completed_keys(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        settings = replace(
            Settings.from_env(),
            hy_max_tokens=args.max_tokens,
            qwen_max_tokens=args.max_tokens,
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
                record = run_paired_record(
                    pipeline,
                    database,
                    case,
                    repetition,
                    sequence=ordinal,
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


def load_completed_keys(path: Path) -> set[tuple[str, int]]:
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
            completed.add((str(record["case_id"]), int(record["repetition"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid JSONL record at {path}:{line_number}"
            ) from error
    return completed


def run_paired_record(
    pipeline: TranslationChatPipeline,
    database: ChatDatabase,
    case: BenchmarkCase,
    repetition: int,
    *,
    sequence: int,
) -> dict[str, Any]:
    languages = (
        ["English", "Chinese"]
        if sequence % 2 == 1
        else ["Chinese", "English"]
    )
    results: dict[str, Any] = {}
    for language in languages:
        title = f"benchmark:{case.id}:run-{repetition}:{language}"
        conversation_id = database.create_conversation(language, title)
        started = perf_counter()
        result = pipeline.run_turn(conversation_id, case.request_ko)
        results[language.lower()] = serialize_result(
            result,
            conversation_id=conversation_id,
            total_seconds=perf_counter() - started,
        )

    return {
        "case_id": case.id,
        "category": case.category,
        "repetition": repetition,
        "request_ko": case.request_ko,
        "execution_order": languages,
        "created_at": datetime.now(UTC).isoformat(),
        "english": results["english"],
        "chinese": results["chinese"],
    }


def serialize_result(
    result: TurnResult,
    *,
    conversation_id: int,
    total_seconds: float,
) -> dict[str, Any]:
    data = asdict(result)
    data["conversation_id"] = conversation_id
    data["total_seconds"] = total_seconds
    for key, value in list(data.items()):
        if key.endswith("_seconds"):
            data[key] = round(float(value), 4)
    return data


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_csv(jsonl_path: Path, csv_path: Path) -> None:
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fieldnames = [
        "case_id",
        "category",
        "repetition",
        "request_ko",
        "english_input_pivot",
        "english_qwen_pivot",
        "english_final_ko",
        "chinese_input_pivot",
        "chinese_qwen_pivot",
        "chinese_final_ko",
        "english_total_seconds",
        "chinese_total_seconds",
        "english_final_chars",
        "chinese_final_chars",
        "final_length_delta",
        "english_quality_score",
        "chinese_quality_score",
        "preferred_pivot",
        "evaluator_notes",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            english = record["english"]
            chinese = record["chinese"]
            english_final = english["assistant_ko"]
            chinese_final = chinese["assistant_ko"]
            writer.writerow(
                {
                    "case_id": record["case_id"],
                    "category": record["category"],
                    "repetition": record["repetition"],
                    "request_ko": record["request_ko"],
                    "english_input_pivot": english["user_pivot"],
                    "english_qwen_pivot": english["assistant_pivot"],
                    "english_final_ko": english_final,
                    "chinese_input_pivot": chinese["user_pivot"],
                    "chinese_qwen_pivot": chinese["assistant_pivot"],
                    "chinese_final_ko": chinese_final,
                    "english_total_seconds": english["total_seconds"],
                    "chinese_total_seconds": chinese["total_seconds"],
                    "english_final_chars": len(english_final),
                    "chinese_final_chars": len(chinese_final),
                    "final_length_delta": len(chinese_final) - len(english_final),
                    "english_quality_score": "",
                    "chinese_quality_score": "",
                    "preferred_pivot": "",
                    "evaluator_notes": "",
                }
            )


if __name__ == "__main__":
    main()
