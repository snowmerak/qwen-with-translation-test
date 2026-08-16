from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .config import Settings
from .database import ChatDatabase


TRANSLATION_PROMPT = """Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:

{source_text}"""


@dataclass(frozen=True, slots=True)
class TurnResult:
    user_ko: str
    user_en: str
    assistant_en: str
    assistant_ko: str


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

    def translate(self, source_text: str, target_lang: str) -> str:
        prompt = TRANSLATION_PROMPT.format(
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

    def answer_in_english(
        self, conversation_id: int, user_en: str
    ) -> str:
        messages = [
            {"role": "system", "content": self.settings.qwen_system_prompt},
            *self.database.get_english_context(conversation_id),
            {"role": "user", "content": user_en},
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

        user_en = self.translate(user_ko, "English")
        assistant_en = self.answer_in_english(conversation_id, user_en)
        assistant_ko = self.translate(assistant_en, "Korean")

        self.database.append_turn(
            conversation_id,
            user_ko=user_ko,
            user_en=user_en,
            assistant_en=assistant_en,
            assistant_ko=assistant_ko,
        )
        return TurnResult(
            user_ko=user_ko,
            user_en=user_en,
            assistant_en=assistant_en,
            assistant_ko=assistant_ko,
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
