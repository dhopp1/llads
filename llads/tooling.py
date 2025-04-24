import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
import os
import uuid

today = datetime.date.today()
date_string = today.strftime("%Y-%m-%d")


def gen_tool_call(llm, tools, prompt, addt_context=None):
    "bind tools to a custom LLM"

    if addt_context is not None:
        prompt += addt_context

    def tool_chain(model_output):
        tool_map = {tool.name: tool for tool in tools}
        chosen_tool = tool_map[model_output["name"]]
        return itemgetter("arguments") | chosen_tool

    # render tools as a string
    rendered_tools = render_text_description(tools)

    system_prompt = (
        llm.system_prompts.loc[lambda x: x["step"] == "raw data tool call", "prompt"]
        .values[0]
        .format(date_string=date_string, rendered_tools=rendered_tools)
    )

    # choosing tool call
    combined_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    select_tool_chain = combined_prompt | llm | JsonOutputParser()

    try:
        tool_call = select_tool_chain.invoke({"input": prompt})
    except:
        tool_call = "error"

    # actual running of tool
    if type(tool_call) != list:
        tool_call = [tool_call]

    invoked_results = []
    for i in range(len(tool_call)):
        tool_i = RunnableLambda(lambda args: tool_call[i]) | tool_chain

        try:
            invoked_results.append(tool_i.invoke(""))
        except:
            invoked_results = ["error"]

    return {
        "query_id": str(uuid.uuid4()),
        "tool_call": tool_call,
        "invoked_result": invoked_results,
    }


def gen_plot_call(llm, tools, tool_result, prompt):
    "generate a plot call"
    llm._data[f'{tool_result["query_id"]}_result'].to_csv(
        f'{tool_result["query_id"]}_result.csv', index=False
    )

    def tool_chain(model_output):
        tool_map = {tool.name: tool for tool in tools}
        chosen_tool = tool_map[model_output["name"]]
        return itemgetter("arguments") | chosen_tool

    # render visualizations as a string
    rendered_tools = render_text_description(tools)

    system_prompt = (
        llm.system_prompts.loc[lambda x: x["step"] == "plot tool call", "prompt"]
        .values[0]
        .format(
            rendered_tools=rendered_tools,
            csv_path=f'{tool_result["query_id"]}_result.csv',
            markdown_result_df=llm._data[f'{tool_result["query_id"]}_result']
            .head()
            .to_markdown(index=False),
        )
    )

    # choosing tool call
    combined_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    select_tool_chain = combined_prompt | llm | JsonOutputParser()

    try:
        tool_call = select_tool_chain.invoke({"input": prompt})
    except:
        tool_call = "error"

    # actual running of tool
    if type(tool_call) != list:
        tool_call = [tool_call]

    invoked_results = []
    for i in range(len(tool_call)):
        tool_i = RunnableLambda(lambda args: tool_call[i]) | tool_chain

        try:
            invoked_results.append(tool_i.invoke(""))
        except:
            invoked_results = ["error"]

    # remove temporary csv
    os.remove(f'{tool_result["query_id"]}_result.csv')

    return {
        "visualiation_call": tool_call,
        "invoked_result": invoked_results,
    }


def gen_description(llm, tool, tool_call, invoked_result):
    "generate a full description of a single tool and result"
    # metadata
    name = tool_call["name"]
    arguments = str(tool_call["arguments"])
    tool_desc = render_text_description(tool)

    # actual data
    actual_data = invoked_result.head().to_markdown(index=False)

    # final prompt
    desc = (
        llm.system_prompts.loc[
            lambda x: x["step"] == "generate call description", "prompt"
        ]
        .values[0]
        .format(
            name=name,
            arguments=arguments,
            tool_desc=tool_desc,
            actual_data=actual_data,
        )
    )

    return desc


def create_data_dictionary(llm, data, tools, tool_result):
    "given the result of a tool call, create data dictionary so the LLM can access the resulting data"

    # creating the data dictionary
    for i in range(len(tool_result["tool_call"])):
        data[f"{tool_result['query_id']}_{i}"] = tool_result["invoked_result"][i]

    # looping through and creating the input for the LLM
    instructions = llm.system_prompts.loc[
        lambda x: x["step"] == "data dictionary intro", "prompt"
    ].values[0]
    for i in range(len(tool_result["tool_call"])):
        intermediate_dataset_name = f"""self._data["{tool_result['query_id']}_{i}"]"""
        tool_descriptions = gen_description(
            llm,
            [_ for _ in tools if _.name == tool_result["tool_call"][i]["name"]],
            tool_result["tool_call"][i],
            tool_result["invoked_result"][i],
        )

        instructions += (
            llm.system_prompts.loc[
                lambda x: x["step"] == "data dictionary body", "prompt"
            ]
            .values[0]
            .format(
                intermediate_dataset_name=intermediate_dataset_name,
                tool_descriptions=tool_descriptions,
            )
        )
    return instructions


def create_final_pandas_instructions(llm, tools, tool_result, prompt):
    "create final prompt for the LLM to manipulate the Pandas data"
    data_dict_desc = create_data_dictionary(llm, llm._data, tools, tool_result)

    instructions = (
        llm.system_prompts.loc[
            lambda x: x["step"] == "pandas manipulation call", "prompt"
        ]
        .values[0]
        .format(
            date_string=date_string,
            prompt=prompt,
            data_dict_desc=data_dict_desc,
            result_dataset_name=f"""self._data["{tool_result['query_id']}_result"]""",
        )
    )

    return {
        "data_desc": data_dict_desc,
        "pd_instructions": instructions,
    }
