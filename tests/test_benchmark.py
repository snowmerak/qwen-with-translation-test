from pathlib import Path

from qwen_translation_test.benchmark import load_cases


def test_benchmark_has_ten_unique_cases() -> None:
    cases = load_cases(Path("data/benchmark_cases.json"))

    assert len(cases) == 10
    assert len({case.id for case in cases}) == 10
    assert all(case.request_ko for case in cases)
