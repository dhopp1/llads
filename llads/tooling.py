from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter


def gen_tool_call(llm, tools, prompt):
    "bind tools to a custom LLM"

    # render tools as a string
    rendered_tools = render_text_description(tools)

    system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

    {rendered_tools}

    Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys and nothing else."""

    # choosing tool call
    combined_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    tool_chain = combined_prompt | llm | JsonOutputParser()

    try:
        tool_call = tool_chain.invoke({"input": prompt})
    except:
        tool_call = "error"

    # actual running of tool
    try:
        invoked_chain = (
            tool_chain
            | itemgetter("arguments")
            | [_ for _ in tools if _.name == tool_call["name"]][0]
        )
        invoked_result = invoked_chain.invoke({"input": prompt})
    except:
        invoked_chain = "error"
        invoked_result = "error"

    return {
        "tool_chain": tool_chain,
        "tool_call": tool_call,
        "invoked_chain": invoked_chain,
        "invoked_result": invoked_result,
    }
