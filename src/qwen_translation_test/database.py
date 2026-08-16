from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class StoredMessage:
    role: Role
    content_pivot: str
    content_ko: str


class ChatDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New conversation',
                pivot_language TEXT NOT NULL DEFAULT 'English',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content_pivot TEXT NOT NULL,
                content_ko TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id) ON DELETE CASCADE,
                UNIQUE (conversation_id, sequence)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, sequence);

            CREATE TABLE IF NOT EXISTS tool_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                tool_call_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_tool_executions_conversation
                ON tool_executions(conversation_id, id);
            """
        )
        self._migrate_english_schema()
        self.connection.commit()

    def _migrate_english_schema(self) -> None:
        conversation_columns = {
            row["name"]
            for row in self.connection.execute(
                "PRAGMA table_info(conversations)"
            ).fetchall()
        }
        if "pivot_language" not in conversation_columns:
            self.connection.execute(
                "ALTER TABLE conversations ADD COLUMN "
                "pivot_language TEXT NOT NULL DEFAULT 'English'"
            )

        message_columns = {
            row["name"]
            for row in self.connection.execute(
                "PRAGMA table_info(messages)"
            ).fetchall()
        }
        if "content_pivot" not in message_columns and "content_en" in message_columns:
            self.connection.execute(
                "ALTER TABLE messages RENAME COLUMN content_en TO content_pivot"
            )

    def create_conversation(
        self,
        pivot_language: str = "English",
        title: str = "New conversation",
    ) -> int:
        cursor = self.connection.execute(
            "INSERT INTO conversations(title, pivot_language) VALUES (?, ?)",
            (title, pivot_language),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def conversation_exists(self, conversation_id: int) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return row is not None

    def get_pivot_language(self, conversation_id: int) -> str:
        row = self.connection.execute(
            "SELECT pivot_language FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Conversation {conversation_id} does not exist")
        return str(row["pivot_language"])

    def append_turn(
        self,
        conversation_id: int,
        *,
        user_ko: str,
        user_pivot: str,
        assistant_pivot: str,
        assistant_ko: str,
    ) -> None:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) AS last_sequence "
            "FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        sequence = int(row["last_sequence"]) + 1

        title = " ".join(user_ko.split())[:80] or "New conversation"
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO messages(
                    conversation_id, sequence, role, content_pivot, content_ko
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (conversation_id, sequence, "user", user_pivot, user_ko),
                    (
                        conversation_id,
                        sequence + 1,
                        "assistant",
                        assistant_pivot,
                        assistant_ko,
                    ),
                ],
            )
            self.connection.execute(
                """
                UPDATE conversations
                SET title = CASE
                        WHEN title = 'New conversation' THEN ?
                        ELSE title
                    END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (title, conversation_id),
            )

    def get_messages(self, conversation_id: int) -> list[StoredMessage]:
        rows = self.connection.execute(
            """
            SELECT role, content_pivot, content_ko
            FROM messages
            WHERE conversation_id = ?
            ORDER BY sequence
            """,
            (conversation_id,),
        ).fetchall()
        return [
            StoredMessage(
                role=row["role"],
                content_pivot=row["content_pivot"],
                content_ko=row["content_ko"],
            )
            for row in rows
        ]

    def get_pivot_context(self, conversation_id: int) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content_pivot}
            for message in self.get_messages(conversation_id)
        ]

    def list_conversations(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT c.id, c.title, c.pivot_language, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations AS c
            LEFT JOIN messages AS m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.id DESC
            """
        ).fetchall()

    def record_tool_execution(
        self,
        conversation_id: int,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments_json: str,
        result_json: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO tool_executions(
                conversation_id,
                tool_call_id,
                tool_name,
                arguments_json,
                result_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                tool_call_id,
                tool_name,
                arguments_json,
                result_json,
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ChatDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
