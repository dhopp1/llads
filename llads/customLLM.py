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
    _query_results: list = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self._data = {}
        self._query_results = {}

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

    def gen_tool_call(self, tools, prompt, addt_context=None):
        "determine which tools to call and call them"
        return gen_tool_call(self, tools, prompt, addt_context)

    def gen_pandas_df(self, tools, tool_result, prompt, addt_context=None):
        "execute pandas manipulations to answer prompt"
        if addt_context is not None:
            prompt += addt_context

        result = create_final_pandas_instructions(self, tools, tool_result, prompt)
        llm_call = (
            self(result["pd_instructions"]).split("```python")[1].replace("```", "")
        )
        exec(llm_call)

        return {
            "data_desc": result["data_desc"],
            "pd_code": llm_call,
        }

    def explain_pandas_df(self, result, prompt, addt_context=None):
        "explain steps taken for data manipulation"

        if addt_context is not None:
            prompt += addt_context

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

    def gen_final_commentary(
        self, tool_result, prompt, validate=True, addt_context=None
    ):
        "generate the final commentary on the dataset"

        if addt_context is not None:
            prompt += addt_context

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

    def gen_plot_call(self, tools, tool_result, prompt, addt_context=None):
        "generate a visual aid plot"

        if addt_context is not None:
            prompt += addt_context

        plot_result = gen_plot_call(self, tools, tool_result, prompt)

        return {
            "visualization_call": plot_result["visualiation_call"],
            "invoked_result": plot_result["invoked_result"],
        }

    def gen_free_plot(self, tool_result, prompt, addt_context=None):
        "give more freedom to the LLM to produce whatever plot it wants with matplotlib"

        if addt_context is not None:
            prompt += addt_context

        query_id = tool_result["query_id"]

        instructions = (
            self.system_prompts.loc[
                lambda x: x["step"] == "free plot tool call", "prompt"
            ]
            .values[0]
            .format(
                prompt=prompt,
                result_df_name=f"self._data[{query_id}_result]",
                markdown_result_df=self._data[f"{query_id}_result"],
                plot_name=f"_{query_id.replace('-', '_')}_plot",
            )
        )

        plot_call = self(instructions).split("```python")[1].replace("```", "")
        exec(plot_call)

        return {
            "visualization_call": [plot_call],
            "invoked_result": [eval(f"_{query_id.replace('-', '_')}_plot")],
        }

    def gen_complete_response(
        self,
        prompt,
        tools=None,
        plot_tools=None,
        validate=True,
        use_free_plot=False,
        n_retries=5,
        addt_context_gen_tool_call=None,
        addt_context_gen_pandas_df=None,
        addt_context_explain_pandas_df=None,
        addt_context_gen_final_commentary=None,
        addt_context_gen_plot_call=None,
        quiet=False,
    ):
        "run the entire pipeline from one function"
        # raw data call
        if not (quiet):
            print("Determining which tools to use...")
        attempts = 0
        while attempts < n_retries:
            try:
                tool_result = self.gen_tool_call(
                    tools=tools,
                    prompt=prompt,
                    addt_context=addt_context_gen_tool_call,
                )
                attempts = n_retries
            except Exception:
                attempts += 1
                if attempts == n_retries:
                    print(
                        "An error occurred during the tool determination step. Please try again or reformulate your query."
                    )

        # pandas manipulation
        if not (quiet):
            print("Transforming the data...")
        attempts = 0
        while attempts < n_retries:
            try:
                result = self.gen_pandas_df(
                    tools=tools,
                    tool_result=tool_result,
                    prompt=prompt,
                    addt_context=addt_context_gen_pandas_df,
                )
                attempts = n_retries
            except Exception:
                attempts += 1
                if attempts == n_retries:
                    print(
                        "An error occurred during the data transformation step. Please try again or reformulate your query."
                    )

        # explanation of pandas manipulation
        if not (quiet):
            print("Explaining the transformations...")
        attempts = 0
        while attempts < n_retries:
            try:
                explanation = self.explain_pandas_df(
                    result, prompt=prompt, addt_context=addt_context_explain_pandas_df
                )
                attempts = n_retries
            except Exception:
                attempts += 1
                if attempts == n_retries:
                    print(
                        "An error occurred during the data explanation step. Please try again or reformulate your query."
                    )

        # commentary on the result
        if not (quiet):
            print("Generating commentary...")
        attempts = 0
        while attempts < n_retries:
            try:
                commentary = self.gen_final_commentary(
                    tool_result,
                    prompt=prompt,
                    validate=validate,
                    addt_context=addt_context_gen_final_commentary,
                )
                attempts = n_retries
            except Exception:
                attempts += 1
                if attempts == n_retries:
                    print(
                        "An error occurred during the commentary step. Please try again or reformulate your query."
                    )

        # generating a plot
        if not (quiet):
            print("Generating a visualization...")
        attempts = 0
        while attempts < n_retries:
            try:
                if use_free_plot:
                    plots = self.gen_free_plot(
                        tool_result=tool_result,
                        prompt=prompt,
                        addt_context=addt_context_gen_plot_call,
                    )
                else:
                    plots = self.gen_plot_call(
                        tools=plot_tools,
                        tool_result=tool_result,
                        prompt=prompt,
                        addt_context=addt_context_gen_plot_call,
                    )
                attempts = n_retries
            except Exception:
                attempts += 1
                if attempts == n_retries:
                    print(
                        "An error occurred during the data visualization step. Please try again or reformulate your query."
                    )

        return {
            "initial_prompt": prompt,
            "tool_result": tool_result,
            "pd_code": result,
            "dataset": self._data[f"{tool_result['query_id']}_result"],
            "explanation": explanation,
            "commentary": commentary,
            "plots": plots,
        }

    def chat(
        self,
        prompt,
        tools=None,
        plot_tools=None,
        validate=True,
        use_free_plot=False,
        prior_query_id=None,
        n_retries=5,
        addt_context_gen_tool_call=None,
        addt_context_gen_pandas_df=None,
        addt_context_explain_pandas_df=None,
        addt_context_gen_final_commentary=None,
        addt_context_gen_plot_call=None,
        quiet=False,
    ):
        "same as gen_complete_response, but if given a list of complete responses, generate a followup context-rich prompt given a new prompt first"
        context_query_ids = []  # which prior queries went into this context
        if prior_query_id is None:
            result = self.gen_complete_response(
                prompt=prompt,
                tools=tools,
                plot_tools=plot_tools,
                validate=validate,
                use_free_plot=use_free_plot,
                n_retries=n_retries,
                addt_context_gen_tool_call=addt_context_gen_tool_call,
                addt_context_gen_pandas_df=addt_context_gen_pandas_df,
                addt_context_explain_pandas_df=addt_context_explain_pandas_df,
                addt_context_gen_final_commentary=addt_context_gen_final_commentary,
                addt_context_gen_plot_call=addt_context_gen_plot_call,
                quiet=quiet,
            )
        else:
            context_rich_prompt = (
                self.system_prompts.loc[
                    lambda x: x["step"] == "context rich prompt start", "prompt"
                ]
                .values[0]
                .format(prompt=prompt)
            )

            # dynamically calculating prior messages given only the prior query id
            prior_query_ids = [prior_query_id] + self._query_results[prior_query_id][
                "context_query_ids"
            ]
            complete_responses = [self._query_results[_] for _ in prior_query_ids]

            for i in range(len(complete_responses)):
                context_query_ids.append(
                    complete_responses[i]["tool_result"]["query_id"]
                )

                # reiteration of old prompt
                ex_num = i + 1

                context_rich_prompt += (
                    self.system_prompts.loc[
                        lambda x: x["step"] == "context rich prompt body", "prompt"
                    ]
                    .values[0]
                    .format(
                        ex_num=ex_num,
                        initial_prompt=complete_responses[i]["initial_prompt"],
                        data_desc=complete_responses[i]["pd_code"]["data_desc"],
                        pd_code=complete_responses[i]["pd_code"]["pd_code"],
                        commentary=complete_responses[i]["commentary"],
                        visualization_code=complete_responses[i]["plots"][
                            "visualization_call"
                        ][0],
                    )
                )

            result = self.gen_complete_response(
                prompt=context_rich_prompt,
                tools=tools,
                plot_tools=plot_tools,
                validate=validate,
                use_free_plot=use_free_plot,
                n_retries=n_retries,
                addt_context_gen_tool_call=addt_context_gen_tool_call,
                addt_context_gen_pandas_df=addt_context_gen_pandas_df,
                addt_context_explain_pandas_df=addt_context_explain_pandas_df,
                addt_context_gen_final_commentary=addt_context_gen_final_commentary,
                addt_context_gen_plot_call=addt_context_gen_plot_call,
                quiet=quiet,
            )

        if prior_query_id is None:
            result["context_rich_prompt"] = ""
        else:
            result["initial_prompt"] = prompt
            result["context_rich_prompt"] = context_rich_prompt

        result["context_query_ids"] = context_query_ids

        # saving result
        self._query_results[result["tool_result"]["query_id"]] = result

        return result
