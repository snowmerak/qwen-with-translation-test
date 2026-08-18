import pytest

from qwen_translation_test.config import normalize_pivot_language


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("en", "English"),
        ("중국어", "Chinese"),
        ("한국어", "Korean"),
        ("ko", "Korean"),
        ("일어", "Japanese"),
        ("ja", "Japanese"),
    ],
)
def test_normalize_pivot_language_aliases(alias: str, expected: str) -> None:
    assert normalize_pivot_language(alias) == expected
