from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

from llads.customLLM import customLLM
from llads.tools import get_world_bank_gdp_data
from llads.visualizations import gen_plot


# parameters
api_key = "API_KEY"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
model_name = "gemini-2.0-flash"
temperature = 0.0
max_tokens = 4096
validate = True
use_free_plot = True
system_prompts = pd.read_csv(
    "https://raw.githubusercontent.com/dhopp1/llads/refs/heads/main/system_prompts.csv"
)

# app
app = FastAPI()

# creating the LLM (gemini 2.0 flash as an example)
llm = customLLM(
    api_key=api_key,
    base_url=base_url,
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
    system_prompts=system_prompts,
)

# defining which tools the LLM has available to it
tools = [get_world_bank_gdp_data]
plot_tools = [gen_plot]


class ChatRequest(BaseModel):
    prompt: str
    prior_query_id: str = None


@app.post("/api/v1/chat/")
async def vector_search(request: ChatRequest):
    results = llm.chat(
        prompt=request.prompt,
        tools=tools,
        plot_tools=plot_tools,
        validate=validate,
        use_free_plot=use_free_plot,
        prior_query_id=request.prior_query_id,
    )

    return results["tool_result"]["query_id"]
