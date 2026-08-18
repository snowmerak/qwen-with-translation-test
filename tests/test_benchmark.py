from pathlib import Path

from qwen_translation_test.benchmark import (
    build_parser,
    load_cases,
    load_completed_keys,
    serialize_tool_executions,
)


def test_benchmark_has_ten_unique_cases() -> None:
    cases = load_cases(Path("data/benchmark_cases.json"))

    assert len(cases) == 10
    assert len({case.id for case in cases}) == 10
    assert all(case.request_ko for case in cases)


def test_tool_execution_trace_is_decoded_for_benchmark() -> None:
    rows = [
        {
            "tool_call_id": "call-1",
            "tool_name": "execute_python",
            "arguments_json": '{"code":"print(42)"}',
            "result_json": '{"success":true,"stdout":"42\\n"}',
            "created_at": "2026-08-16 00:00:00",
        }
    ]

    executions = serialize_tool_executions(rows)

    assert executions[0]["arguments"]["code"] == "print(42)"
    assert executions[0]["result"]["stdout"] == "42\n"


def test_benchmark_accepts_separate_generation_limits() -> None:
    args = build_parser().parse_args(
        ["--hy-max-tokens", "4096", "--qwen-max-tokens", "32768"]
    )

    assert args.hy_max_tokens == 4096
    assert args.qwen_max_tokens == 32768
    assert args.max_tokens is None


def test_benchmark_defaults_to_four_pivots_and_can_include_bypass() -> None:
    default_args = build_parser().parse_args([])
    bypass_args = build_parser().parse_args(
        ["--pivot-languages", "한국어", "일어", "--include-bypass"]
    )

    assert default_args.pivot_languages == [
        "English",
        "Chinese",
        "Korean",
        "Japanese",
    ]
    assert bypass_args.pivot_languages == ["Korean", "Japanese"]
    assert bypass_args.include_bypass is True


def test_completed_benchmark_keys_are_scoped_to_comparison_modes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results.jsonl"
    output.write_text(
        '{"case_id":"one","repetition":1,'
        '"comparison_modes":["English","Chinese"]}\n',
        encoding="utf-8",
    )

    assert load_completed_keys(output, ["English", "Chinese"]) == {
        ("one", 1)
    }
    assert load_completed_keys(output, ["Korean", "Japanese"]) == set()
