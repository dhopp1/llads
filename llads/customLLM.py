from pydantic import Field, PrivateAttr
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from openai import OpenAI

from llads.tooling import create_final_pandas_instructions, gen_tool_call


class customLLM(LLM):
    api_key: str = Field(...)
    base_url: str = Field(...)
    model_name: str = Field(...)
    system_prompt: str = "you are a chatbot"
    temperature: float = 0.0
    max_tokens: int = 2048

    _client: OpenAI = PrivateAttr()
    _data: dict = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self._data = {}

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    def _call(
        self,
        prompt: str,
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

    def gen_tool_call(self, tools, prompt):
        "determine which tools to call and call them"
        return gen_tool_call(self, tools, prompt)

    def gen_pandas_df(self, tools, tool_result, prompt):
        "execute pandas manipulations to answer prompt"
        result = create_final_pandas_instructions(
            self._data, tools, tool_result, prompt
        )
        llm_call = (
            self(result["pd_instructions"]).split("```python")[1].replace("```", "")
        )
        exec(llm_call)

        return {
            "data_desc": result["data_desc"],
            "pd_code": llm_call,
        }

    def explain_pandas_df(self, result, prompt):
        "explain steps taken for data manipulation"
        instructions = f"""
An LLM was given this initial prompt to answer: {prompt}

It was given this raw data to answer it: {result["data_desc"]}

It then used that raw data to generate this Pandas code: {result["pd_code"]}

Given that information, explain step by step what was done to end up with a final dataset that best answers the original prompt. Don't go into the details of code calls, just give higher-level overviews of steps taken.
"""

        explanation = self(instructions)

        return explanation
