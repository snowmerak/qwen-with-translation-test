from pathlib import Path

from qwen_translation_test.cli import _conversation_id, build_parser
from qwen_translation_test.database import ChatDatabase


def test_cli_accepts_new_pivots_and_bypass_alias() -> None:
    korean = build_parser().parse_args(["--pivot-language", "한국어"])
    japanese = build_parser().parse_args(["--pivot-language", "일어"])
    bypass = build_parser().parse_args(["--bypass"])

    assert korean.pivot_language == "Korean"
    assert japanese.pivot_language == "Japanese"
    assert bypass.bypass_translation is True


def test_cli_creates_bypass_as_direct_korean_conversation(
    tmp_path: Path,
) -> None:
    with ChatDatabase(tmp_path / "test.db") as database:
        conversation_id = _conversation_id(
            database,
            requested=None,
            pivot_language="English",
            bypass_translation=True,
        )

        assert database.get_pivot_language(conversation_id) == "Korean"
        assert database.is_translation_bypassed(conversation_id) is True
