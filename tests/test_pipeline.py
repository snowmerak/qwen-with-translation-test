from pathlib import Path
from types import SimpleNamespace

from qwen_translation_test.config import Settings
from qwen_translation_test.database import ChatDatabase
from qwen_translation_test.pipeline import TranslationChatPipeline


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.outputs = iter(
            [
                "What is Python?",
                "Python is a general-purpose programming language.",
                "Python은 범용 프로그래밍 언어입니다.",
            ]
        )

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=next(self.outputs)))
            ]
        )


def settings(database_path: Path) -> Settings:
    return Settings(
        base_url="http://example.test/v1",
        api_key="local",
        hy_model="hy",
        qwen_model="qwen",
        database_path=database_path,
        hy_max_tokens=4096,
        qwen_max_tokens=4096,
        qwen_temperature=0.7,
        qwen_system_prompt="Answer clearly.",
    )


def test_pipeline_calls_hy_qwen_hy_and_persists_pair(tmp_path: Path) -> None:
    fake_completions = FakeCompletions()
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=fake_completions)
    )

    with ChatDatabase(tmp_path / "test.db") as database:
        conversation_id = database.create_conversation()
        pipeline = TranslationChatPipeline(
            settings(tmp_path / "test.db"), database, fake_client
        )

        result = pipeline.run_turn(conversation_id, "Python이 뭐야?")

        assert result.assistant_ko == "Python은 범용 프로그래밍 언어입니다."
        assert [call["model"] for call in fake_completions.calls] == [
            "hy",
            "qwen",
            "hy",
        ]
        qwen_messages = fake_completions.calls[1]["messages"]
        assert qwen_messages == [
            {"role": "system", "content": "Answer clearly."},
            {"role": "user", "content": "What is Python?"},
        ]
        assert database.get_english_context(conversation_id)[-1] == {
            "role": "assistant",
            "content": "Python is a general-purpose programming language.",
        }


def test_next_turn_uses_only_english_history(tmp_path: Path) -> None:
    fake_completions = FakeCompletions()
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=fake_completions)
    )

    with ChatDatabase(tmp_path / "test.db") as database:
        conversation_id = database.create_conversation()
        database.append_turn(
            conversation_id,
            user_ko="이전 질문",
            user_en="Previous question",
            assistant_en="Previous answer",
            assistant_ko="이전 답변",
        )
        pipeline = TranslationChatPipeline(
            settings(tmp_path / "test.db"), database, fake_client
        )

        pipeline.run_turn(conversation_id, "Python이 뭐야?")

        qwen_messages = fake_completions.calls[1]["messages"]
        assert {"role": "user", "content": "Previous question"} in qwen_messages
        assert {"role": "assistant", "content": "Previous answer"} in qwen_messages
        assert all("이전" not in message["content"] for message in qwen_messages)
