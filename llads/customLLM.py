import datetime
import pandas as pd
from pydantic import Field, PrivateAttr
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from openai import OpenAI

from llads.tooling import create_final_pandas_instructions, gen_plot_call, gen_tool_call


today = datetime.date.today()
date_string = today.strftime("%Y-%m-%d")


class customLLM(LLM):
    api_key: str = Field(...)
    base_url: str = Field(...)
    model_name: str = Field(...)
    system_prompts: pd.DataFrame = Field(...)
    system_prompt: str = ""  # for every call
    temperature: float = 0.0
    max_tokens: int = 2048

    _client: OpenAI = PrivateAttr()
    _data: dict = PrivateAttr()
    _system_prompts: pd.DataFrame = PrivateAttr()

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
        result = create_final_pandas_instructions(self, tools, tool_result, prompt)
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

        instructions = (
            self.system_prompts.loc[
                lambda x: x["step"] == "pandas explanation call", "prompt"
            ]
            .values[0]
            .format(
                prompt=prompt,
                data_desc=result["data_desc"],
                pd_code=result["pd_code"],
            )
        )

        explanation = self(instructions)

        return explanation

    def gen_final_commentary(self, tool_result, prompt, validate=True):
        "generate the final commentary on the dataset"
        query_id = tool_result["query_id"]

        # initial commentary
        commentary_instructions = (
            self.system_prompts.loc[
                lambda x: x["step"] == "initial commentary call", "prompt"
            ]
            .values[0]
            .format(
                date_string=date_string,
                prompt=prompt,
                result_df_markdown=self._data[f"{query_id}_result"].to_markdown(
                    index=False
                ),
            )
        )
        commentary = self(commentary_instructions)

        # validation commentary
        if validate:
            validation_instructions = (
                self.system_prompts.loc[
                    lambda x: x["step"] == "initial commentary call", "prompt"
                ]
                .values[0]
                .format(
                    date_string=date_string,
                    prompt=prompt,
                    result_df_markdown=self._data[f"{query_id}_result"].to_markdown(
                        index=False
                    ),
                    commentary=commentary,
                )
            )
            commentary = self(validation_instructions)

        return commentary

    def gen_plot_call(self, tools, tool_result, prompt):
        "generate a visual aid plot"

        plot_result = gen_plot_call(self, tools, tool_result, prompt)

        return {
            "visualization_call": plot_result["visualiation_call"],
            "plots": plot_result["invoked_result"],
        }
