import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
import uuid

today = datetime.date.today()
date_string = today.strftime("%Y-%m-%d")


def gen_tool_call(llm, tools, prompt):
    "bind tools to a custom LLM"

    def tool_chain(model_output):
        tool_map = {tool.name: tool for tool in tools}
        chosen_tool = tool_map[model_output["name"]]
        return itemgetter("arguments") | chosen_tool

    # render tools as a string
    rendered_tools = render_text_description(tools)

    system_prompt = f"""You are an assistant that has access to the following set of tools. The current date is {date_string}. Here are the names and descriptions for each tool:

    {rendered_tools}

    Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys and nothing else. If you need multiple tools to answer the user's query, return a list of JSON blobs for each tool"""

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


def gen_description(tool, tool_call, invoked_result):
    "generate a full description of a single tool and result"
    # metadata
    name = tool_call["name"]
    arguments = str(tool_call["arguments"])
    tool_desc = render_text_description(tool)

    # actual data
    actual_data = invoked_result.head().to_markdown(index=False)

    # final prompt
    desc = f"""
Result of function name: {name},

With function arguments: {arguments}

The function's docstring: {tool_desc}

The contents of the first couple rows of the dataset: 
    
{actual_data}
"""
    return desc


def create_data_dictionary(data, tools, tool_result):
    "given the result of a tool call, create data dictionary so the LLM can access the resulting data"

    # creating the data dictionary
    for i in range(len(tool_result["tool_call"])):
        data[f"{tool_result['query_id']}_{i}"] = tool_result["invoked_result"][i]

    # looping through and creating the input for the LLM
    instructions = "You have the following data available to you:\n\n"
    for i in range(len(tool_result["tool_call"])):
        instructions += f"""
Information on the variable named 'self._data["{tool_result['query_id']}_{i}"]':
    
{gen_description([_ for _ in tools if _.name == tool_result["tool_call"][i]["name"]], tool_result["tool_call"][i], tool_result["invoked_result"][i])}

----next dataset-------
   
"""
    return instructions


def create_final_pandas_instructions(data, tools, tool_result, prompt):
    "create final prompt for the LLM to manipulate the Pandas data"
    data_dict_desc = create_data_dictionary(data, tools, tool_result)

    instructions = f"""
You are given this initial prompt (the current date is {date_string}): {prompt}
    
You have the following datasets available to you: {data_dict_desc}

Using Pandas, manipulate the dataset so that you can best answer the initial prompt. Save the output in a variable called 'self._data["{tool_result['query_id']}_result"]'. Output only Python code.
"""

    return {
        "data_desc": data_dict_desc,
        "pd_instructions": instructions,
    }
