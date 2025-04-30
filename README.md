# llads
'Large Language Data and Statistics'. A library to generate LLM insights to data.

## Installation
Install the package from [PyPI](https://pypi.org/project/llads/), as well as the libraries in `requirements.txt`.

## Usage
### LLM
You can use any LLM that works with the OpenAI API syntax, including a local LlamaCPP server. Note that the LLM needs to be powerful enough to properly parse and produce the expected outputs for the various steps of the chain. The following information is necessary for creating the LLM:

- api key (if using a cloud LLM provider)
- base url
- model name

### Example
```py
import pandas as pd

from llads.customLLM import customLLM
from llads.tools import get_world_bank_gdp_data # this is a custom tool included as an example. You can define and pass your own tools
from llads.visualizations import gen_plot # this is a line/bar plot visualization tool included as an example. You can define and pass your own visualization tools

system_prompts = pd.read_csv("https://raw.githubusercontent.com/dhopp1/llads/refs/heads/main/system_prompts.csv") # a good default is included in the repo, but you can edit to your own needs

# creating the LLM (gemini 2.0 flash as an example)
llm = customLLM(
        api_key="API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model_name="gemini-2.0-flash",
        temperature=0.0,
        max_tokens=2048,
        system_prompts=system_prompts,
)

# defining which tools the LLM has available to it
tools = [get_world_bank_gdp_data]
plot_tools = [gen_plot]

# generating a response
prompt = "What is the GDP of Italy and the UK as a % of Germany over the last 5 years?" # the user's initial question
results = llm.chat(
	prompt=prompt, 
	tools=tools, 
	plot_tools=plot_tools, 
	validate=True, # if True, the LLM will perform an additional validation step on its commentary
	use_free_plot=False, # if False, the LLM will have to use one of the plot_tools, if True, it will be free to make its own matplotlib plot
	prior_query_id=None, # None, because this is the first query in the chat history
)

# follow-up question
new_query = "Add France to the analysis"
followup_result = llm.chat(
	prompt=new_query, 
	tools=tools, 
	plot_tools=plot_tools, 
	validate=True,
	use_free_plot=False,
	prior_query_id=results["tool_result"]["query_id"], # pass our prior query id to make message history available
)

# you can access prior query results via the query id
llm._query_results[query_id]
```

### Interpreting output
The `chat()` function will produce a dictionary with the following values:

- _initial\_prompt_: The question passed by the user
- _tool\_result_: A dictionary with the following information:
	- _query\_id_: The unique ID number of this query
	- _tool\_call_: The name and arguments of the tools the LLM called
	- _invoked\_result_: The actual DataFrame resulting from the tool calls
	- _n\_tokens\_input_: The number of tokens consumed by the LLM for input in this step
	- _n\_tokens\_output_: The number of tokens consumed by the LLM for output in this step
	- _seconds\_taken_: How many seconds this step took to run
- _pd\_code_: A dictionary with the following information:
	- _data\_desc_: A text description of the data made available to the LLM via the tool call
	- _pd\_code_: The Python code the LLM executed to edit the raw data available to it
	- _n\_tokens\_input_: The number of tokens consumed by the LLM for input in this step
	- _n\_tokens\_output_: The number of tokens consumed by the LLM for output in this step
	- _seconds\_taken_: How many seconds this step took to run
- _dataset_: The actual DataFrame that is the result of the `pd_code` call
- _explanation_: A dictionary with the following information:
	- _explanation_: The LLM's explanation of the data manipulation process undergone to answer the user's question 
	- _n\_tokens\_input_: The number of tokens consumed by the LLM for input in this step
	- _n\_tokens\_output_: The number of tokens consumed by the LLM for output in this step
	- _seconds\_taken_: How many seconds this step took to run
- _commentary_: A dictionary with the following information:
	- _commentary_: The LLM's commentary on the final dataset answering the user's question
	- _n\_tokens\_input_: The number of tokens consumed by the LLM for input in this step
	- _n\_tokens\_output_: The number of tokens consumed by the LLM for output in this step
	- _seconds\_taken_: How many seconds this step took to run

- _plots_: A dictionary with the following values:
	- _visualization\_call_: A list of either the matplotlib code written or the plotting function call run to create the visualization to answer the user's question
	- _invoked\_result_: A list of the actual plot figures produced to answer the user's question
	- _n\_tokens\_input_: The number of tokens consumed by the LLM for input in this step
	- _n\_tokens\_output_: The number of tokens consumed by the LLM for output in this step
	- _seconds\_taken_: How many seconds this step took to run
- _context\_rich\_prompt_: The prompt passed to the LLM containing the prior context. Empty string if it's the first question in the chat.

## Explanation of steps/chain
Note that for each step in the chain, you can provide additional context information specifically for that step by passing a string to any of the following arguments in the `chat()` function:

	- `addt_context_gen_tool_call`
	- `addt_context_gen_pandas_df`
	- `addt_context_explain_pandas_df`
	- `addt_context_gen_final_commentary`
	- `addt_context_gen_plot_call`

1. The LLM determines which raw data functions it wants to call with which arguments via the  `llm.gen_tool_call()` function, the calls and generates the raw datasets. 
2. Given the raw data available from the previous step, the `llm.gen_pandas_df()` produces Python code to create a final result dataset.
3. The LLM explains the data transformation steps via the `llm.explain_pandas_df()` function.
4. The LLM is given the final full result dataset and writes commentary answering the user's question via the `llm.gen_final_commentary()` funcrtion.
5. If `validate=True` in the `llm.gen_final_commentary()` call, the LLM performs a validation step on its commentary to look for and correct errors.
6. The LLM produces a visualization to help answer the user's question, via either the `llm.gen_free_plot()` function (if `use_free_plot=True`) or the `llm.gen_plot_call()` function. The former allows the LLM to create any Matplotlib plot, the latter restricts it to calling one of the predefined visualization tools. Useful if you want to customize style, etc.

If a `prior_query_id` is passed, at the very beginning of the pipeline the user's prompt will be augmented with the full context history of previous messages, including tool calls, data manipulation steps, commentary provided, and visualizations created.

## Defining your own datasets/tools
The library contains the `get_world_bank_gdp_data` function as an example. To make additional data available to the LLM, you can define your own tools. For example, say we wanted to add a simple addition tool:

```py
from langchain_core.tools import tool

@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int
    
tools = [add, get_world_bank_gdp_data] # the LLM will now be able to choose either the addition tool, or the World Bank GDP tool.
```

As long as the input and outputs of the function are well defined, the LLM should be able to use it if helpful to answer a user's question.