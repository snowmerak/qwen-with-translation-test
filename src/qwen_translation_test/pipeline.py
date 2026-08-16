from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from openai import OpenAI

from .config import Settings
from .database import ChatDatabase
from .sandbox import MontyPythonSandbox, SandboxSettings


ENGLISH_TRANSLATION_PROMPT = """Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:

{source_text}"""

CHINESE_TRANSLATION_PROMPT = """将以下文本翻译为{target_lang}，注意只需要输出翻译后的结果，不要额外解释：

{source_text}"""

CHINESE_LANGUAGE_NAMES = {
    "Chinese": "中文",
    "English": "英语",
    "Korean": "韩语",
}

PYTHON_TOOL_ENGLISH = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute a restricted Python subset in a host-isolated Monty worker. "
            "There is no filesystem, environment, network, or third-party package "
            "access. Use it for calculations, data processing, or verifying code. "
            "Print the values needed to answer the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python source code to execute.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}

PYTHON_TOOL_CHINESE = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "在隔离的 Monty 工作进程中执行受限的 Python 子集。"
            "无法访问文件系统、环境变量、网络或第三方包。"
            "适合用于计算、数据处理或验证代码。请打印回答用户所需的值。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的完整 Python 源代码。",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class TurnResult:
    user_ko: str
    user_pivot: str
    assistant_reasoning: str | None
    assistant_pivot: str
    assistant_ko: str
    pivot_language: str
    input_translation_seconds: float
    qwen_seconds: float
    output_translation_seconds: float


@dataclass(frozen=True, slots=True)
class QwenAnswer:
    content: str
    reasoning: str | None


class TranslationChatPipeline:
    def __init__(
        self,
        settings: Settings,
        database: ChatDatabase,
        client: OpenAI | Any | None = None,
        sandbox_settings: SandboxSettings | None = None,
        python_sandbox: MontyPythonSandbox | Any | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.client = client or OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=300.0,
            max_retries=1,
        )
        self.sandbox_settings = sandbox_settings or SandboxSettings.from_env()
        self.python_sandbox = python_sandbox
        if self.python_sandbox is None and self.sandbox_settings.enabled:
            self.python_sandbox = MontyPythonSandbox(self.sandbox_settings)

    def translate(
        self,
        source_text: str,
        target_lang: str,
        *,
        prompt_language: str = "English",
    ) -> str:
        if prompt_language == "Chinese":
            prompt = CHINESE_TRANSLATION_PROMPT.format(
                target_lang=CHINESE_LANGUAGE_NAMES[target_lang],
                source_text=source_text,
            )
        else:
            prompt = ENGLISH_TRANSLATION_PROMPT.format(
                target_lang=target_lang,
                source_text=source_text,
            )
        response = self.client.chat.completions.create(
            model=self.settings.hy_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=1.0,
            max_tokens=self.settings.hy_max_tokens,
        )
        return _completion_text(response)

    def answer_in_pivot_language(
        self,
        conversation_id: int,
        user_pivot: str,
        pivot_language: str,
    ) -> str:
        return self._answer_with_reasoning(
            conversation_id, user_pivot, pivot_language
        ).content

    def _answer_with_reasoning(
        self,
        conversation_id: int,
        user_pivot: str,
        pivot_language: str,
    ) -> QwenAnswer:
        system_prompt = self.settings.qwen_system_prompt.replace(
            "{pivot_language}", pivot_language
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *self.database.get_pivot_context(conversation_id),
            {"role": "user", "content": user_pivot},
        ]
        reasoning_parts: list[str] = []
        for _ in range(self.sandbox_settings.max_tool_rounds + 1):
            request: dict[str, Any] = {
                "model": self.settings.qwen_model,
                "messages": messages,
                "temperature": self.settings.qwen_temperature,
                "max_tokens": self.settings.qwen_max_tokens,
            }
            if self.python_sandbox is not None:
                request["tools"] = [
                    PYTHON_TOOL_CHINESE
                    if pivot_language == "Chinese"
                    else PYTHON_TOOL_ENGLISH
                ]
                request["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**request)
            if not response.choices:
                raise RuntimeError("The model returned no choices")
            message = response.choices[0].message
            reasoning = _message_reasoning(message)
            if reasoning:
                reasoning_parts.append(reasoning)
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                return QwenAnswer(
                    content=_message_text(message),
                    reasoning=("\n\n".join(reasoning_parts) or None),
                )

            messages.append(_assistant_tool_message(message))
            for tool_call in tool_calls:
                arguments_json = tool_call.function.arguments
                result_json = self._execute_tool_call(
                    tool_call.function.name,
                    arguments_json,
                )
                self.database.record_tool_execution(
                    conversation_id,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    arguments_json=arguments_json,
                    result_json=result_json,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_json,
                    }
                )

        raise RuntimeError("The model exceeded the maximum Python tool rounds")

    def _execute_tool_call(self, name: str, arguments_json: str) -> str:
        if name != "execute_python":
            return json.dumps({"error": f"Unknown tool: {name}"})
        if self.python_sandbox is None:
            return json.dumps({"error": "Python sandbox is disabled"})
        try:
            arguments = json.loads(arguments_json)
            code = arguments["code"]
            result = self.python_sandbox.execute(code)
            return json.dumps(result.to_dict(), ensure_ascii=False)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            return json.dumps({"error": str(error)}, ensure_ascii=False)

    def run_turn(self, conversation_id: int, user_ko: str) -> TurnResult:
        user_ko = user_ko.strip()
        if not user_ko:
            raise ValueError("The user message cannot be empty")
        if not self.database.conversation_exists(conversation_id):
            raise ValueError(f"Conversation {conversation_id} does not exist")

        pivot_language = self.database.get_pivot_language(conversation_id)
        started = perf_counter()
        user_pivot = self.translate(
            user_ko,
            pivot_language,
            prompt_language=pivot_language,
        )
        input_translation_seconds = perf_counter() - started

        started = perf_counter()
        answer = self._answer_with_reasoning(
            conversation_id, user_pivot, pivot_language
        )
        assistant_pivot = answer.content
        qwen_seconds = perf_counter() - started

        started = perf_counter()
        assistant_ko = self.translate(
            assistant_pivot,
            "Korean",
            prompt_language=pivot_language,
        )
        output_translation_seconds = perf_counter() - started

        self.database.append_turn(
            conversation_id,
            user_ko=user_ko,
            user_pivot=user_pivot,
            assistant_pivot=assistant_pivot,
            assistant_ko=assistant_ko,
        )
        return TurnResult(
            user_ko=user_ko,
            user_pivot=user_pivot,
            assistant_reasoning=answer.reasoning,
            assistant_pivot=assistant_pivot,
            assistant_ko=assistant_ko,
            pivot_language=pivot_language,
            input_translation_seconds=input_translation_seconds,
            qwen_seconds=qwen_seconds,
            output_translation_seconds=output_translation_seconds,
        )

    def list_model_ids(self) -> list[str]:
        return sorted(model.id for model in self.client.models.list().data)


def _completion_text(response: Any) -> str:
    if not response.choices:
        raise RuntimeError("The model returned no choices")
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The model returned an empty text response")
    return content.strip()


def _message_text(message: Any) -> str:
    content = message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The model returned an empty text response")
    return content.strip()


def _message_reasoning(message: Any) -> str | None:
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return None


def _assistant_tool_message(message: Any) -> dict[str, Any]:
    data = {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in getattr(message, "tool_calls", None) or []
        ],
    }
    reasoning = _message_reasoning(message)
    if reasoning:
        data["reasoning_content"] = reasoning
    return data
