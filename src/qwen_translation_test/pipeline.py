from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .config import Settings
from .database import ChatDatabase


ENGLISH_TRANSLATION_PROMPT = """Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:

{source_text}"""

CHINESE_TRANSLATION_PROMPT = """将以下文本翻译为{target_lang}，注意只需要输出翻译后的结果，不要额外解释：

{source_text}"""

CHINESE_LANGUAGE_NAMES = {
    "Chinese": "中文",
    "English": "英语",
    "Korean": "韩语",
}


@dataclass(frozen=True, slots=True)
class TurnResult:
    user_ko: str
    user_pivot: str
    assistant_pivot: str
    assistant_ko: str
    pivot_language: str


class TranslationChatPipeline:
    def __init__(
        self,
        settings: Settings,
        database: ChatDatabase,
        client: OpenAI | Any | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.client = client or OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=300.0,
            max_retries=1,
        )

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
        system_prompt = self.settings.qwen_system_prompt.replace(
            "{pivot_language}", pivot_language
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *self.database.get_pivot_context(conversation_id),
            {"role": "user", "content": user_pivot},
        ]
        response = self.client.chat.completions.create(
            model=self.settings.qwen_model,
            messages=messages,
            temperature=self.settings.qwen_temperature,
            max_tokens=self.settings.qwen_max_tokens,
        )
        return _completion_text(response)

    def run_turn(self, conversation_id: int, user_ko: str) -> TurnResult:
        user_ko = user_ko.strip()
        if not user_ko:
            raise ValueError("The user message cannot be empty")
        if not self.database.conversation_exists(conversation_id):
            raise ValueError(f"Conversation {conversation_id} does not exist")

        pivot_language = self.database.get_pivot_language(conversation_id)
        user_pivot = self.translate(
            user_ko,
            pivot_language,
            prompt_language=pivot_language,
        )
        assistant_pivot = self.answer_in_pivot_language(
            conversation_id, user_pivot, pivot_language
        )
        assistant_ko = self.translate(
            assistant_pivot,
            "Korean",
            prompt_language=pivot_language,
        )

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
            assistant_pivot=assistant_pivot,
            assistant_ko=assistant_ko,
            pivot_language=pivot_language,
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
