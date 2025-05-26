import datetime
import inspect
import pandas as pd
from pydantic import Field, PrivateAttr
import re
import time
from typing import Any, List, Optional, Union

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from openai import OpenAI

from llads.tooling import (
    count_tokens,
    create_final_pandas_instructions,
    gen_plot_call,
    gen_tool_call,
)


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
    reasoning_effort: Optional[Union[str, None]] = Field(
        default=None
    )  # "low", "medium", or "high" for Gemini 2.5 Flash. Otherwise None.

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
            reasoning_effort=self.reasoning_effort,
            messages=messages,
        )

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return response.choices[0].message.content

    def gen_tool_call(self, tools, prompt, addt_context=None):
        "determine which tools to call and call them"
        return gen_tool_call(self, tools, prompt, addt_context)

    def gen_pandas_df(
        self,
        tools,
        tool_result,
        prompt,
        data_desc_unique_threshold=25,
        data_desc_top_n_values=10,
        addt_context=None,
    ):
        "execute pandas manipulations to answer prompt"
        start_time = time.time()

        if addt_context is not None:
            prompt += addt_context

        try:
            result = create_final_pandas_instructions(
                self,
                tools,
                tool_result,
                prompt,
                data_desc_unique_threshold=data_desc_unique_threshold,
                data_desc_top_n_values=data_desc_top_n_values,
            )

            n_tokens_input = count_tokens(result["pd_instructions"])

            llm_call = self(result["pd_instructions"])

            n_tokens_output = count_tokens(llm_call)

            try:
                llm_call = llm_call.split("```python")[1].replace("```", "")
            except:
                pass

            exec(llm_call)

            output = {
                "data_desc": result["data_desc"],
                "pd_code": llm_call,
                "n_tokens_input": n_tokens_input,
                "n_tokens_output": n_tokens_output,
            }
        except:
            output = {
                "data_desc": "error",
                "pd_code": "error",
                "n_tokens_input": 0,
                "n_tokens_output": 0,
            }

        end_time = time.time()
        output["seconds_taken"] = end_time - start_time

        return output

    def explain_pandas_df(self, result, prompt, addt_context=None):
        "explain steps taken for data manipulation"
        start_time = time.time()

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

        n_tokens_input = count_tokens(instructions)

        try:
            explanation = self(instructions)
            n_tokens_output = count_tokens(explanation)
        except:
            explanation = "error"
            n_tokens_input = 0
            n_tokens_output = 0

        end_time = time.time()

        return {
            "explanation": explanation,
            "n_tokens_input": n_tokens_input,
            "n_tokens_output": n_tokens_output,
            "seconds_taken": end_time - start_time,
        }

    def gen_final_commentary(
        self, tool_result, prompt, validate=True, addt_context=None
    ):
        "generate the final commentary on the dataset"
        start_time = time.time()

        if addt_context is not None:
            prompt += addt_context

        query_id = tool_result["query_id"]

        try:
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

            n_tokens_input = count_tokens(commentary_instructions)

            commentary = self(commentary_instructions)

            n_tokens_output = count_tokens(commentary)

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

                n_tokens_input += count_tokens(validation_instructions)

                commentary = self(validation_instructions)

                n_tokens_output += count_tokens(commentary)
        except:
            commentary = "error"
            n_tokens_input = 0
            n_tokens_output = 0

        end_time = time.time()

        return {
            "commentary": commentary,
            "n_tokens_input": n_tokens_input,
            "n_tokens_output": n_tokens_output,
            "seconds_taken": end_time - start_time,
        }

    def gen_plot_call(self, tools, tool_result, prompt, addt_context=None):
        "generate a visual aid plot"

        if addt_context is not None:
            prompt += addt_context

        plot_result = gen_plot_call(self, tools, tool_result, prompt)

        return plot_result

    def gen_free_plot(self, tool_result, prompt, addt_context=None):
        "give more freedom to the LLM to produce whatever plot it wants with matplotlib"
        start_time = time.time()

        if addt_context is not None:
            prompt += addt_context

        query_id = tool_result["query_id"]

        try:
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

            n_tokens_input = count_tokens(instructions)

            plot_call = self(instructions)

            n_tokens_output = count_tokens(plot_call)

            try:
                plot_call = plot_call.split("```python")[1].replace("```", "")
            except:
                pass

            exec(plot_call)

            output = {
                "visualization_call": [plot_call],
                "invoked_result": [eval(f"_{query_id.replace('-', '_')}_plot")],
                "n_tokens_input": n_tokens_input,
                "n_tokens_output": n_tokens_output,
            }
        except:
            output = {
                "visualization_call": ["error"],
                "invoked_result": ["error"],
                "n_tokens_input": 0,
                "n_tokens_output": 0,
            }

        end_time = time.time()
        output["seconds_taken"] = end_time - start_time

        return output

    def gen_complete_response(
        self,
        prompt,
        tools=None,
        plot_tools=None,
        validate=True,
        use_free_plot=False,
        addt_context_gen_tool_call=None,
        addt_context_gen_pandas_df=None,
        addt_context_explain_pandas_df=None,
        addt_context_gen_final_commentary=None,
        addt_context_gen_plot_call=None,
        run_gen_pandas_df=True,
        run_explain_pandas_df=True,
        run_gen_final_commentary=True,
        run_gen_plot=True,
        n_retries=3,
        modules=None,
        prior_query_id=None,
        data_desc_unique_threshold=25,
        data_desc_top_n_values=10,
        quiet=False,
    ):
        "run the entire pipeline from one function"
        # raw data call
        if not (quiet):
            print("Determining which tools to use...")
        attempts = 0
        while attempts < n_retries:
            tool_result = self.gen_tool_call(
                tools=tools,
                prompt=prompt,
                addt_context=addt_context_gen_tool_call,
            )
            if isinstance(tool_result["invoked_result"], list):
                if isinstance(tool_result["invoked_result"][0], str):
                    if tool_result["invoked_result"][0] != "error":
                        condition = True
                    else:
                        condition = False
                else:
                    condition = True
            else:
                condition = True
            if condition:
                attempts = n_retries
            else:
                attempts += 1

        # pandas manipulation
        if run_gen_pandas_df:
            if not (quiet):
                print("Transforming the data...")
            attempts = 0
            while attempts < n_retries:
                result = self.gen_pandas_df(
                    tools=tools,
                    tool_result=tool_result,
                    prompt=prompt,
                    data_desc_unique_threshold=data_desc_unique_threshold,
                    data_desc_top_n_values=data_desc_top_n_values,
                    addt_context=addt_context_gen_pandas_df,
                )
                if result["pd_code"] != "error":
                    attempts = n_retries
                else:
                    attempts += 1

        # explanation of pandas manipulation
        if run_explain_pandas_df:
            if not (quiet):
                print("Explaining the transformations...")
            attempts = 0
            while attempts < n_retries:
                explanation = self.explain_pandas_df(
                    result, prompt=prompt, addt_context=addt_context_explain_pandas_df
                )
                if explanation != "error":
                    attempts = n_retries
                else:
                    attempts += 1

        # commentary on the result
        if run_gen_final_commentary:
            if not (quiet):
                print("Generating commentary...")
            attempts = 0
            while attempts < n_retries:
                commentary = self.gen_final_commentary(
                    tool_result,
                    prompt=prompt,
                    validate=validate,
                    addt_context=addt_context_gen_final_commentary,
                )
                if commentary != "error":
                    attempts = n_retries
                else:
                    attempts += 1

        # generating a plot
        if run_gen_plot:
            if not (quiet):
                print("Generating a visualization...")
            attempts = 0
            while attempts < n_retries:
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
                if isinstance(plots["invoked_result"], list):
                    if isinstance(plots["invoked_result"][0], str):
                        if plots["invoked_result"][0] != "error":
                            condition = True
                        else:
                            condition = False
                    else:
                        condition = True
                else:
                    condition = True
                if condition:
                    attempts = n_retries
                else:
                    attempts += 1

        # final dataframe:
        try:
            dataframe = self._data[f"{tool_result['query_id']}_result"]
        except:
            dataframe = pd.DataFrame()

        ### create full runnable python script
        try:
            python_script = ""

            # adding full text of modules
            if modules is not None and prior_query_id is None:
                python_script += (
                    "### data and visualization call function definitions\n\n"
                )

                if not (isinstance(modules, list)):
                    modules = [modules]

                for module in modules:
                    module_text = inspect.getsource(module).splitlines()
                    module_text = [_ for _ in module_text if "langchain" not in _]
                    module_text = "\n".join(module_text)

                    python_script += module_text + "\n\n"

                python_script += (
                    "### data and visualization call function definitions\n\n"
                )

            # initial tool calls - function definitions
            if modules is None:
                python_script += "### data call function definitions\n\n"
                for tool in tools:
                    if tool.name in [_["name"] for _ in tool_result["tool_call"]]:
                        python_script += inspect.getsource(tool.func) + "\n"

                python_script += "### data call function definitions\n\n"

            # actual function calls
            python_script += "### raw data calls\n\ndata_dict = {}\n\n"

            for i in range(len(tool_result["tool_call"])):
                python_script += f"# function call {i+1}\n"
                python_script += f"""data_dict["{tool_result["query_id"]}_{i}"] = {tool_result["tool_call"][i]["name"]}(**{tool_result["tool_call"][i]["arguments"]})\n\n"""

            python_script += "### raw data calls\n\n"

            # llm pandas
            if run_gen_pandas_df:
                python_script += "### Pandas data manipulation\n"
                python_script += f"""{result["pd_code"]}\n\n"""
                python_script += "### Pandas data manipulation\n\n"

            # visualization tool calls - function definitions
            if run_gen_plot:
                if modules is None:
                    if not (use_free_plot):
                        python_script += "### visualization function definitions\n\n"
                        for plot_tool in plot_tools:
                            if plot_tool.name in [
                                _["name"] for _ in plots["visualization_call"]
                            ]:
                                python_script += (
                                    inspect.getsource(plot_tool.func) + "\n"
                                )

                        python_script += "### visualization function definitions\n\n"

                # actual visualization calls
                python_script += "### visualization calls\n"

                for i in range(len(plots["visualization_call"])):
                    python_script += f"# visualization call {i+1}\n"
                    if not (use_free_plot):
                        python_script += f"""plot_{i+1} = {plots["visualization_call"][i]["name"]}(**{plots["visualization_call"][i]["arguments"]})\n\n"""
                    else:
                        python_script += f"""{plots["visualization_call"][i]}\n\n"""

                python_script += "### visualization calls\n"

                python_script = f"""```py\n{python_script}\n```""".replace(
                    "@tool", ""
                ).replace("self._data", "data_dict")
                python_script = re.sub(
                    r"(\S+?)_result\.csv",
                    lambda m: f"data_dict[{m.group(1)}_result']",
                    python_script,
                ).replace("result']'", "result']")

        except:
            python_script = "error"

        # final return dictionary
        final_dict = {}
        final_dict["initial_prompt"] = prompt
        final_dict["tool_result"] = tool_result
        try:
            final_dict["pd_code"] = result
        except:
            final_dict["pd_code"] = None
        try:
            final_dict["dataset"] = dataframe
        except:
            final_dict["dataset"] = None
        try:
            final_dict["explanation"] = explanation
        except:
            final_dict["explanation"] = None
        try:
            final_dict["commentary"] = commentary
        except:
            final_dict["commentary"] = None
        try:
            final_dict["plots"] = plots
        except:
            final_dict["plots"] = None
        final_dict["python_script"] = python_script

        return final_dict

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
        run_gen_pandas_df=True,
        run_explain_pandas_df=True,
        run_gen_final_commentary=True,
        run_gen_plot=True,
        modules=None,
        data_desc_unique_threshold=25,  # maximum number of cardinal entries in a column to show all options in the PD data description step
        data_desc_top_n_values=10,  # if maximum cardinal entries reached, top how many to show to give idea of values in columns
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
                run_gen_pandas_df=run_gen_pandas_df,
                run_explain_pandas_df=run_explain_pandas_df,
                run_gen_final_commentary=run_gen_final_commentary,
                run_gen_plot=run_gen_plot,
                modules=modules,
                prior_query_id=prior_query_id,
                data_desc_unique_threshold=data_desc_unique_threshold,
                data_desc_top_n_values=data_desc_top_n_values,
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
                        data_desc=(
                            complete_responses[i]["pd_code"]["data_desc"]
                            if complete_responses[i]["pd_code"] is not None
                            else None
                        ),
                        pd_code=(
                            complete_responses[i]["pd_code"]["pd_code"]
                            if complete_responses[i]["pd_code"] is not None
                            else None
                        ),
                        commentary=complete_responses[i]["commentary"],
                        visualization_code=(
                            complete_responses[i]["plots"]["visualization_call"][0]
                            if complete_responses[i]["plots"] is not None
                            else None
                        ),
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
                run_gen_pandas_df=run_gen_pandas_df,
                run_explain_pandas_df=run_explain_pandas_df,
                run_gen_final_commentary=run_gen_final_commentary,
                run_gen_plot=run_gen_plot,
                modules=modules,
                prior_query_id=prior_query_id,
                data_desc_unique_threshold=data_desc_unique_threshold,
                data_desc_top_n_values=data_desc_top_n_values,
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

        # appending former python scripts
        if prior_query_id is not None:
            result["python_script"] = result["python_script"].replace(
                "data_dict = {}", ""
            )  # remove this line if chat history, don't want to delete old datasetes
            result["python_script"] = (
                f"""{self._query_results[prior_query_id]["python_script"]}\n\n##### ----- next query in chat -----\n\n{result["python_script"]}"""
            )
            result["python_script"] = (
                result["python_script"].replace("```py", "").replace("```", "")
            )
            result["python_script"] = "```py\n" + result["python_script"] + "\n```"

        return result
