from pathlib import Path

from qwen_translation_test.database import ChatDatabase


def test_stores_bilingual_pairs_and_builds_english_context(tmp_path: Path) -> None:
    with ChatDatabase(tmp_path / "test.db") as database:
        conversation_id = database.create_conversation()
        database.append_turn(
            conversation_id,
            user_ko="안녕",
            user_en="Hello",
            assistant_en="Hello! How can I help?",
            assistant_ko="안녕하세요! 무엇을 도와드릴까요?",
        )

        assert database.get_english_context(conversation_id) == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]

        stored = database.get_messages(conversation_id)
        assert stored[0].content_ko == "안녕"
        assert stored[0].content_en == "Hello"
        assert stored[1].content_ko == "안녕하세요! 무엇을 도와드릴까요?"
        assert stored[1].content_en == "Hello! How can I help?"
