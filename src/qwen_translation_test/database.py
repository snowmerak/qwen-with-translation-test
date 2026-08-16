from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class StoredMessage:
    role: Role
    content_en: str
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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content_en TEXT NOT NULL,
                content_ko TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id) ON DELETE CASCADE,
                UNIQUE (conversation_id, sequence)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, sequence);
            """
        )
        self.connection.commit()

    def create_conversation(self, title: str = "New conversation") -> int:
        cursor = self.connection.execute(
            "INSERT INTO conversations(title) VALUES (?)", (title,)
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def conversation_exists(self, conversation_id: int) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return row is not None

    def append_turn(
        self,
        conversation_id: int,
        *,
        user_ko: str,
        user_en: str,
        assistant_en: str,
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
                    conversation_id, sequence, role, content_en, content_ko
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (conversation_id, sequence, "user", user_en, user_ko),
                    (
                        conversation_id,
                        sequence + 1,
                        "assistant",
                        assistant_en,
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
            SELECT role, content_en, content_ko
            FROM messages
            WHERE conversation_id = ?
            ORDER BY sequence
            """,
            (conversation_id,),
        ).fetchall()
        return [
            StoredMessage(
                role=row["role"],
                content_en=row["content_en"],
                content_ko=row["content_ko"],
            )
            for row in rows
        ]

    def get_english_context(self, conversation_id: int) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content_en}
            for message in self.get_messages(conversation_id)
        ]

    def list_conversations(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations AS c
            LEFT JOIN messages AS m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.id DESC
            """
        ).fetchall()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ChatDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
