from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from openai import OpenAI

from pydantic import Field, PrivateAttr


class customLLM(LLM):
    system_prompt: str = Field(...)
    api_key: str = Field(...)
    base_url: str = Field(...)
    model_name: str = Field(...)
    temperature: float = 0.0
    max_tokens: int = 2048

    _client: OpenAI = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    def _call(
        self,
        prompt=str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
        )

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return response.choices[0].message.content
