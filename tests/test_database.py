import sqlite3
from pathlib import Path

from qwen_translation_test.database import ChatDatabase


def test_stores_bilingual_pairs_and_builds_pivot_context(tmp_path: Path) -> None:
    with ChatDatabase(tmp_path / "test.db") as database:
        conversation_id = database.create_conversation("Chinese")
        database.append_turn(
            conversation_id,
            user_ko="안녕",
            user_pivot="你好",
            assistant_pivot="你好！有什么可以帮助你的？",
            assistant_ko="안녕하세요! 무엇을 도와드릴까요?",
        )

        assert database.get_pivot_language(conversation_id) == "Chinese"
        assert database.get_pivot_context(conversation_id) == [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
        ]

        stored = database.get_messages(conversation_id)
        assert stored[0].content_ko == "안녕"
        assert stored[0].content_pivot == "你好"
        assert stored[1].content_ko == "안녕하세요! 무엇을 도와드릴까요?"
        assert stored[1].content_pivot == "你好！有什么可以帮助你的？"


def test_stores_translation_bypass_on_conversation(tmp_path: Path) -> None:
    with ChatDatabase(tmp_path / "test.db") as database:
        translated_id = database.create_conversation("Korean")
        bypass_id = database.create_conversation(
            "Korean", translation_bypass=True
        )

        assert database.is_translation_bypassed(translated_id) is False
        assert database.is_translation_bypassed(bypass_id) is True
        rows = {row["id"]: row for row in database.list_conversations()}
        assert rows[bypass_id]["translation_bypass"] == 1


def test_migrates_existing_english_database(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL DEFAULT 'New conversation',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            sequence INTEGER NOT NULL,
            role TEXT NOT NULL,
            content_en TEXT NOT NULL,
            content_ko TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO conversations(title) VALUES ('Legacy');
        INSERT INTO messages(
            conversation_id, sequence, role, content_en, content_ko
        ) VALUES (1, 1, 'user', 'Hello', '안녕');
        """
    )
    connection.close()

    with ChatDatabase(path) as database:
        assert database.get_pivot_language(1) == "English"
        assert database.is_translation_bypassed(1) is False
        assert database.get_messages(1)[0].content_pivot == "Hello"
