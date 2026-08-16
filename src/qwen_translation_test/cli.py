from __future__ import annotations

import argparse
import sys

from openai import OpenAIError

from .config import Settings
from .database import ChatDatabase
from .pipeline import TranslationChatPipeline, TurnResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with Qwen through Korean-English-Korean translation."
    )
    parser.add_argument(
        "--conversation",
        type=int,
        help="Resume an existing SQLite conversation ID.",
    )
    parser.add_argument(
        "--once",
        metavar="KOREAN_TEXT",
        help="Run one turn and exit. Only the final Korean response is printed.",
    )
    parser.add_argument(
        "--show-english",
        action="store_true",
        help="Show the Korean-to-English input and Qwen's English response.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List model IDs exposed by the configured server and exit.",
    )
    parser.add_argument(
        "--list-conversations",
        action="store_true",
        help="List saved conversations and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        with ChatDatabase(settings.database_path) as database:
            pipeline = TranslationChatPipeline(settings, database)

            if args.list_models:
                for model_id in pipeline.list_model_ids():
                    print(model_id)
                return

            if args.list_conversations:
                _print_conversations(database)
                return

            conversation_id = _conversation_id(database, args.conversation)
            if args.once is not None:
                result = pipeline.run_turn(conversation_id, args.once)
                _print_result(result, args.show_english, one_shot=True)
                return

            _interactive_chat(pipeline, database, conversation_id, args.show_english)
    except (OpenAIError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


def _conversation_id(database: ChatDatabase, requested: int | None) -> int:
    if requested is None:
        return database.create_conversation()
    if not database.conversation_exists(requested):
        raise ValueError(f"Conversation {requested} does not exist")
    return requested


def _interactive_chat(
    pipeline: TranslationChatPipeline,
    database: ChatDatabase,
    conversation_id: int,
    show_english: bool,
) -> None:
    print(f"conversation: {conversation_id}")
    print("종료: /quit, 저장 내역: /history")
    while True:
        try:
            user_ko = input("\n나> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_ko:
            continue
        if user_ko in {"/quit", "/exit"}:
            return
        if user_ko == "/history":
            _print_history(database, conversation_id)
            continue

        result = pipeline.run_turn(conversation_id, user_ko)
        _print_result(result, show_english, one_shot=False)


def _print_result(
    result: TurnResult, show_english: bool, *, one_shot: bool
) -> None:
    if show_english:
        print(f"[입력 영문]\n{result.user_en}", file=sys.stderr)
        print(f"[Qwen 영문]\n{result.assistant_en}", file=sys.stderr)
    if one_shot:
        print(result.assistant_ko)
    else:
        print(f"\nQwen> {result.assistant_ko}")


def _print_history(database: ChatDatabase, conversation_id: int) -> None:
    messages = database.get_messages(conversation_id)
    if not messages:
        print("저장된 메시지가 없습니다.")
        return
    for message in messages:
        speaker = "나" if message.role == "user" else "Qwen"
        print(f"\n{speaker} [KO]> {message.content_ko}")
        print(f"{speaker} [EN]> {message.content_en}")


def _print_conversations(database: ChatDatabase) -> None:
    rows = database.list_conversations()
    if not rows:
        print("저장된 대화가 없습니다.")
        return
    for row in rows:
        print(
            f"{row['id']:>4}  {row['message_count']:>4} messages  "
            f"{row['updated_at']}  {row['title']}"
        )


if __name__ == "__main__":
    main()
