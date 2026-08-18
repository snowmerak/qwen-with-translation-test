from __future__ import annotations

import argparse
import sys

from openai import OpenAIError

from .config import (
    SUPPORTED_PIVOT_LANGUAGES,
    Settings,
    normalize_pivot_language,
)
from .database import ChatDatabase
from .pipeline import TranslationChatPipeline, TurnResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with Qwen through a configurable pivot translation."
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
        "--pivot-language",
        type=normalize_pivot_language,
        choices=SUPPORTED_PIVOT_LANGUAGES,
        help=(
            "Pivot language for a new conversation "
            "(English, Chinese, Korean, or Japanese)."
        ),
    )
    parser.add_argument(
        "--bypass-translation",
        "--bypass",
        dest="bypass_translation",
        action="store_true",
        help=(
            "Send Korean directly to Qwen and use its Korean response without "
            "calling Hy-MT2. The mode is saved with the new conversation."
        ),
    )
    parser.add_argument(
        "--show-pivot",
        "--show-english",
        dest="show_pivot",
        action="store_true",
        help="Show the translated input and Qwen's response before Korean translation.",
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bypass_translation and args.pivot_language is not None:
        parser.error(
            "--bypass-translation cannot be combined with --pivot-language"
        )
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

            conversation_id = _conversation_id(
                database,
                args.conversation,
                args.pivot_language or settings.pivot_language,
                args.bypass_translation,
            )
            if args.once is not None:
                result = pipeline.run_turn(conversation_id, args.once)
                _print_result(result, args.show_pivot, one_shot=True)
                return

            _interactive_chat(pipeline, database, conversation_id, args.show_pivot)
    except (OpenAIError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


def _conversation_id(
    database: ChatDatabase,
    requested: int | None,
    pivot_language: str,
    bypass_translation: bool,
) -> int:
    if requested is None:
        if bypass_translation:
            pivot_language = "Korean"
        return database.create_conversation(
            pivot_language,
            translation_bypass=bypass_translation,
        )
    if not database.conversation_exists(requested):
        raise ValueError(f"Conversation {requested} does not exist")
    if bypass_translation and not database.is_translation_bypassed(requested):
        raise ValueError(
            "--bypass-translation can only create a new bypass conversation; "
            "the requested conversation uses translation"
        )
    return requested


def _interactive_chat(
    pipeline: TranslationChatPipeline,
    database: ChatDatabase,
    conversation_id: int,
    show_pivot: bool,
) -> None:
    pivot_language = database.get_pivot_language(conversation_id)
    mode = (
        "translation bypass"
        if database.is_translation_bypassed(conversation_id)
        else f"{pivot_language} pivot"
    )
    print(f"conversation: {conversation_id} ({mode})")
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
        _print_result(result, show_pivot, one_shot=False)


def _print_result(
    result: TurnResult, show_pivot: bool, *, one_shot: bool
) -> None:
    if show_pivot:
        label = _language_label(result.pivot_language)
        if result.translation_bypassed:
            label = "KO/direct"
        print(f"[입력 {label}]\n{result.user_pivot}", file=sys.stderr)
        print(f"[Qwen {label}]\n{result.assistant_pivot}", file=sys.stderr)
    if one_shot:
        print(result.assistant_ko)
    else:
        print(f"\nQwen> {result.assistant_ko}")


def _print_history(database: ChatDatabase, conversation_id: int) -> None:
    messages = database.get_messages(conversation_id)
    if not messages:
        print("저장된 메시지가 없습니다.")
        return
    pivot_language = database.get_pivot_language(conversation_id)
    label = _language_label(pivot_language)
    if database.is_translation_bypassed(conversation_id):
        label = "KO/direct"
    for message in messages:
        speaker = "나" if message.role == "user" else "Qwen"
        print(f"\n{speaker} [KO]> {message.content_ko}")
        print(f"{speaker} [{label}]> {message.content_pivot}")


def _print_conversations(database: ChatDatabase) -> None:
    rows = database.list_conversations()
    if not rows:
        print("저장된 대화가 없습니다.")
        return
    for row in rows:
        mode = (
            "Bypass"
            if row["translation_bypass"]
            else str(row["pivot_language"])
        )
        print(
            f"{row['id']:>4}  {row['message_count']:>4} messages  "
            f"{mode:<8}  {row['updated_at']}  {row['title']}"
        )


def _language_label(language: str) -> str:
    return {
        "English": "EN",
        "Chinese": "ZH",
        "Korean": "KO",
        "Japanese": "JA",
    }[language]


if __name__ == "__main__":
    main()
