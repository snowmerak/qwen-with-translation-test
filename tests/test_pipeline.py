from pathlib import Path
from types import SimpleNamespace

from qwen_translation_test.config import Settings
from qwen_translation_test.database import ChatDatabase
from qwen_translation_test.pipeline import TranslationChatPipeline
from qwen_translation_test.sandbox import SandboxResult, SandboxSettings


class FakeCompletions:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.outputs = iter(
            outputs
            or [
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
        qwen_system_prompt="Answer clearly in {pivot_language}.",
        pivot_language="English",
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
            {"role": "system", "content": "Answer clearly in English."},
            {"role": "user", "content": "What is Python?"},
        ]
        assert database.get_pivot_context(conversation_id)[-1] == {
            "role": "assistant",
            "content": "Python is a general-purpose programming language.",
        }


def test_chinese_pipeline_uses_only_chinese_history(tmp_path: Path) -> None:
    fake_completions = FakeCompletions(
        [
            "Python是什么？",
            "Python是一种通用编程语言。",
            "Python은 범용 프로그래밍 언어입니다.",
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=fake_completions)
    )

    with ChatDatabase(tmp_path / "test.db") as database:
        conversation_id = database.create_conversation("Chinese")
        database.append_turn(
            conversation_id,
            user_ko="이전 질문",
            user_pivot="上一个问题",
            assistant_pivot="上一个回答",
            assistant_ko="이전 답변",
        )
        pipeline = TranslationChatPipeline(
            settings(tmp_path / "test.db"), database, fake_client
        )

        pipeline.run_turn(conversation_id, "Python이 뭐야?")

        qwen_messages = fake_completions.calls[1]["messages"]
        assert qwen_messages[0] == {
            "role": "system",
            "content": "Answer clearly in Chinese.",
        }
        assert {"role": "user", "content": "上一个问题"} in qwen_messages
        assert {"role": "assistant", "content": "上一个回答"} in qwen_messages
        assert all("이전" not in message["content"] for message in qwen_messages)
        assert database.get_pivot_language(conversation_id) == "Chinese"

        first_hy_prompt = fake_completions.calls[0]["messages"][0]["content"]
        final_hy_prompt = fake_completions.calls[2]["messages"][0]["content"]
        assert first_hy_prompt.startswith("将以下文本翻译为中文")
        assert final_hy_prompt.startswith("将以下文本翻译为韩语")


class FakeToolCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                id="call_python_1",
                type="function",
                function=SimpleNamespace(
                    name="execute_python",
                    arguments='{"code":"print(6 * 7)"}',
                ),
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
        else:
            message = SimpleNamespace(
                content="The calculated answer is 42.", tool_calls=None
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeSandbox:
    def __init__(self) -> None:
        self.codes: list[str] = []

    def execute(self, code: str) -> SandboxResult:
        self.codes.append(code)
        return SandboxResult(
            success=True,
            result=None,
            stdout="42\n",
            stderr="",
            error=None,
        )


def test_qwen_tool_call_runs_python_and_returns_result(tmp_path: Path) -> None:
    completions = FakeToolCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    sandbox = FakeSandbox()
    sandbox_settings = SandboxSettings(
        enabled=True,
        timeout_seconds=1,
        max_memory_bytes=16 * 1024 * 1024,
        max_recursion_depth=100,
        max_code_chars=2_000,
        max_output_bytes=2_000,
        max_tool_rounds=2,
    )

    with ChatDatabase(tmp_path / "test.db") as database:
        conversation_id = database.create_conversation("English")
        pipeline = TranslationChatPipeline(
            settings(tmp_path / "test.db"),
            database,
            client,
            sandbox_settings=sandbox_settings,
            python_sandbox=sandbox,
        )

        answer = pipeline.answer_in_pivot_language(
            conversation_id, "Calculate 6 * 7 with Python.", "English"
        )

        assert answer == "The calculated answer is 42."
        assert sandbox.codes == ["print(6 * 7)"]
        assert "tools" in completions.calls[0]
        tool_description = completions.calls[0]["tools"][0]["function"][
            "description"
        ]
        assert tool_description.startswith("Execute a restricted Python subset")
        second_messages = completions.calls[1]["messages"]
        assert second_messages[-1]["role"] == "tool"
        assert "42" in second_messages[-1]["content"]
        stored = database.connection.execute(
            "SELECT tool_name, arguments_json, result_json FROM tool_executions"
        ).fetchone()
        assert stored["tool_name"] == "execute_python"
        assert "print(6 * 7)" in stored["arguments_json"]
        assert len(database.get_tool_executions(conversation_id)) == 1


def test_chinese_qwen_receives_chinese_python_tool_description(
    tmp_path: Path,
) -> None:
    completions = FakeToolCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    sandbox = FakeSandbox()
    sandbox_settings = SandboxSettings(
        enabled=True,
        timeout_seconds=1,
        max_memory_bytes=16 * 1024 * 1024,
        max_recursion_depth=100,
        max_code_chars=2_000,
        max_output_bytes=2_000,
        max_tool_rounds=2,
    )

    with ChatDatabase(tmp_path / "test.db") as database:
        conversation_id = database.create_conversation("Chinese")
        pipeline = TranslationChatPipeline(
            settings(tmp_path / "test.db"),
            database,
            client,
            sandbox_settings=sandbox_settings,
            python_sandbox=sandbox,
        )

        pipeline.answer_in_pivot_language(
            conversation_id, "请用 Python 计算 6 * 7。", "Chinese"
        )

        description = completions.calls[0]["tools"][0]["function"][
            "description"
        ]
        assert description.startswith("在隔离的 Monty 工作进程中")


class FakeReasoningCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_: object) -> SimpleNamespace:
        self.calls += 1
        if self.calls == 1:
            message = SimpleNamespace(content="What is 6 * 7?")
        elif self.calls == 2:
            message = SimpleNamespace(
                content="The answer is 42.",
                reasoning_content="I should multiply 6 by 7.",
                tool_calls=None,
            )
        else:
            message = SimpleNamespace(content="정답은 42입니다.")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_pipeline_captures_qwen_reasoning_content(tmp_path: Path) -> None:
    completions = FakeReasoningCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with ChatDatabase(tmp_path / "test.db") as database:
        conversation_id = database.create_conversation("English")
        pipeline = TranslationChatPipeline(
            settings(tmp_path / "test.db"), database, client
        )

        result = pipeline.run_turn(conversation_id, "6 곱하기 7은?")

        assert result.assistant_reasoning == "I should multiply 6 by 7."
