from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter


def gen_tool_call(llm, tools, prompt):
    "bind tools to a custom LLM"

    def tool_chain(model_output):
        tool_map = {tool.name: tool for tool in tools}
        chosen_tool = tool_map[model_output["name"]]
        return itemgetter("arguments") | chosen_tool

    # render tools as a string
    rendered_tools = render_text_description(tools)

    system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

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
        "tool_call": tool_call,
        "invoked_result": invoked_results,
    }
