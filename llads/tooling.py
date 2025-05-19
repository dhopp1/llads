import datetime
import inspect
import io
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
import os
import pandas as pd
import re
import time
import uuid

today = datetime.date.today()
date_string = today.strftime("%Y-%m-%d")


def df_description(
    df,
    unique_threshold=25,
    top_n_values=5,  # Number of top values to show for high cardinality
):
    """
    Generates a markdown table describing a Pandas DataFrame

    Includes column names, data types, numeric ranges.
    For string/object columns:
    - Lists unique values if count <= unique_threshold.
    - Shows unique count and top_n most frequent values (with counts)
      if count > unique_threshold (Approach 4).

    Args:
        df (pd.DataFrame): The DataFrame to describe.
        unique_threshold (int): Maximum number of unique values to list for
                                string/object columns before switching to
                                showing value counts.
        top_n_values (int): The number of most frequent values to display
                            for high-cardinality string/object columns.

    Returns:
        str: A markdown formatted string describing the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        return "Error: Input is not a Pandas DataFrame."

    buffer = io.StringIO()  # Use StringIO to build the string efficiently

    # Write Markdown table header
    buffer.write("| Column Name | Data Type | Details |\n")
    buffer.write("|---|---|---|\n")

    for col_name in df.columns:
        col_data = df[col_name]
        dtype = col_data.dtype
        has_missing = col_data.isnull().any()  # Check for missing values once

        details = ""

        # Check for predominantly missing columns
        if col_data.isnull().all():
            details = "All missing values"
            has_missing = False  # Already captured by the main detail

        # Numeric types (integer or float)
        elif pd.api.types.is_numeric_dtype(dtype):
            min_val = col_data.min(skipna=True)
            max_val = col_data.max(skipna=True)

            # Check if min/max calculation resulted in NaN (can happen if all are NaN)
            if pd.isna(min_val) and pd.isna(max_val):
                details = "Numeric (contains only missing values)"
                has_missing = False  # Covered by detail message
            else:
                # Format based on dtype to avoid unnecessary decimals for ints
                if pd.api.types.is_integer_dtype(dtype):
                    # Use try-except for potential large numbers that can't be int
                    try:
                        details = (
                            f"Numeric (Range: ${int(min_val):,}$ - ${int(max_val):,}$)"
                        )
                    except (ValueError, TypeError):
                        details = f"Numeric (Range: ${min_val:,}$ - ${max_val:,}$)"  # Fallback for large ints
                else:
                    # Simple formatting, adjust precision as needed
                    details = f"Numeric (Range: ${min_val:,.2f}$ - ${max_val:,.2f}$)"

        # String / Object types
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            try:  # Use try-except for mixed types that might fail nunique/unique
                # Drop NA for counting unique values accurately for the threshold check
                col_data_non_null = col_data.dropna()
                num_unique = col_data_non_null.nunique()

                if num_unique == 0 and not has_missing:  # Only non-nulls were checked
                    details = (
                        "String/Object (No unique values found besides potential NaNs)"
                    )
                elif num_unique <= unique_threshold:
                    unique_values = col_data_non_null.unique()
                    # Format unique values, escaping backticks and handling potential non-string types
                    formatted_uniques = [
                        str(v).replace("`", "\\`") for v in unique_values
                    ]
                    details = f"String/Object ({num_unique} unique): `{', '.join(formatted_uniques)}`"
                else:
                    # High cardinality: show count and top N value counts (Approach 4)
                    top_counts = col_data.value_counts().head(
                        top_n_values
                    )  # value_counts handles NaNs by default
                    # Format top counts, escaping backticks in keys (values)
                    formatted_counts = [
                        f"`{k_escaped}` ({v})"
                        for k, v in top_counts.items()
                        for k_escaped in [
                            str(k).replace("`", "\\`")
                        ]  # Perform the replacement here
                    ]
                    details = (
                        f"String/Object ({num_unique} unique values. "
                        f"Top {len(formatted_counts)} counts: {', '.join(formatted_counts)})"
                    )

            except Exception as e:
                details = f"Object (Could not analyze uniques/counts: {e})"

        # Datetime types
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            try:
                min_date_obj = col_data.min(skipna=True)
                max_date_obj = col_data.max(skipna=True)
                min_date = (
                    min_date_obj.strftime("%Y-%m-%d")
                    if pd.notna(min_date_obj)
                    else "N/A"
                )
                max_date = (
                    max_date_obj.strftime("%Y-%m-%d")
                    if pd.notna(max_date_obj)
                    else "N/A"
                )

                if min_date == "N/A" and max_date == "N/A":
                    details = "Datetime (contains only missing values)"
                    has_missing = False
                else:
                    details = f"Datetime (Range: {min_date} - {max_date})"

            except Exception as e:
                details = f"Datetime (Error analyzing range: {e})"

        # Boolean types
        elif pd.api.types.is_bool_dtype(dtype):
            counts = (
                col_data.value_counts()
            )  # Includes NaN counts if specified, default excludes
            true_count = counts.get(True, 0)
            false_count = counts.get(False, 0)
            details = f"Boolean (True: {true_count:,}, False: {false_count:,})"

        # Categorical types
        elif pd.api.types.is_categorical_dtype(dtype):
            num_categories = len(col_data.cat.categories)
            if num_categories <= unique_threshold:
                # Format categories, escaping backticks
                formatted_cats = [
                    str(c).replace("`", "\\`") for c in col_data.cat.categories
                ]
                details = f"Categorical ({num_categories} categories): `{', '.join(formatted_cats)}`"
            else:
                # High cardinality categoricals: Show count and top N value counts
                top_counts = col_data.value_counts().head(top_n_values)
                # Format top counts, escaping backticks in keys (categories)
                formatted_counts = [
                    f"`{k_escaped}` ({v})"
                    for k, v in top_counts.items()
                    for k_escaped in [
                        str(k).replace("`", "\\`")
                    ]  # Perform the replacement here
                ]
                details = (
                    f"Categorical ({num_categories} categories. "
                    f"Top {len(formatted_counts)} counts: {', '.join(formatted_counts)})"
                )

        # Other types
        else:
            details = f"Type: {dtype} (Specific details not generated)"

        # Append missing value info if relevant and not already covered
        if has_missing:
            missing_count = col_data.isnull().sum()
            # Add count only if > 0, otherwise has_missing might be True from bool None
            if missing_count > 0:
                details += f" ({missing_count:,} missing values)"

        # Escape pipe characters in column name and details for markdown
        safe_col_name = str(col_name).replace("|", "\\|")
        safe_details = details.replace("|", "\\|")

        buffer.write(f"| {safe_col_name} | {dtype} | {safe_details} |\n")

    return buffer.getvalue()


def count_tokens(text):
    tokens_whitespace = text.split()
    tokens_punctuation = re.findall(r"\b\w+\b|[^\w\s]", text)

    return len(tokens_whitespace) + len(tokens_punctuation)


def gen_tool_call(llm, tools, prompt, addt_context=None):
    "bind tools to a custom LLM"
    start_time = time.time()

    if addt_context is not None:
        prompt += addt_context

    try:

        def tool_chain(model_output):
            tool_map = {tool.name: tool for tool in tools}
            chosen_tool = tool_map[model_output["name"]]
            return itemgetter("arguments") | chosen_tool

        # render tools as a string
        rendered_tools = render_text_description(tools)

        system_prompt = (
            llm.system_prompts.loc[
                lambda x: x["step"] == "raw data tool call", "prompt"
            ]
            .values[0]
            .format(date_string=date_string, rendered_tools=rendered_tools)
        )

        # choosing tool call
        combined_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "{input}")]
        )

        n_tokens_input = count_tokens(system_prompt + prompt)

        select_tool_chain = combined_prompt | llm | JsonOutputParser()

        try:
            tool_call = select_tool_chain.invoke({"input": prompt})
        except:
            tool_call = "error"

        n_tokens_output = count_tokens(str(tool_call))

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

        output = {
            "query_id": str(uuid.uuid4()),
            "tool_call": tool_call,
            "invoked_result": invoked_results,
            "n_tokens_input": n_tokens_input,
            "n_tokens_output": n_tokens_output,
        }
    except:
        output = {
            "query_id": str(uuid.uuid4()),
            "tool_call": "error",
            "invoked_result": ["error"],
            "n_tokens_input": 0,
            "n_tokens_output": 0,
        }

    end_time = time.time()
    output["seconds_taken"] = end_time - start_time

    return output


def gen_plot_call(llm, tools, tool_result, prompt):
    "generate a plot call"
    start_time = time.time()

    try:
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

        n_tokens_input = count_tokens(system_prompt + prompt)

        # choosing tool call
        combined_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "{input}")]
        )

        select_tool_chain = combined_prompt | llm | JsonOutputParser()

        try:
            tool_call = select_tool_chain.invoke({"input": prompt})
        except:
            tool_call = "error"

        n_tokens_output = count_tokens(str(tool_call))

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

        output = {
            "visualization_call": tool_call,
            "invoked_result": invoked_results,
            "n_tokens_input": n_tokens_input,
            "n_tokens_output": n_tokens_output,
        }
    except:
        output = {
            "visualization_call": "error",
            "invoked_result": ["error"],
            "n_tokens_input": 0,
            "n_tokens_output": 0,
        }

    end_time = time.time()

    output["seconds_taken"] = end_time - start_time

    return output


def gen_description(
    llm,
    tool,
    tool_call,
    invoked_result,
    data_desc_unique_threshold=25,
    data_desc_top_n_values=10,
):
    "generate a full description of a single tool and result"
    # metadata
    name = "`" + tool_call["name"] + "`"
    arguments = "`" + str(tool_call["arguments"]) + "`"
    tool_desc = "\n\n```\n\n" + render_text_description(tool) + "\n\n```\n\n"

    # actual data
    actual_data = df_description(
        invoked_result,
        unique_threshold=data_desc_unique_threshold,
        top_n_values=data_desc_top_n_values,
    )

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


def create_data_dictionary(
    llm,
    data,
    tools,
    tool_result,
    data_desc_unique_threshold=25,
    data_desc_top_n_values=10,
):
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
            data_desc_unique_threshold=data_desc_unique_threshold,
            data_desc_top_n_values=data_desc_top_n_values,
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


def create_final_pandas_instructions(
    llm,
    tools,
    tool_result,
    prompt,
    data_desc_unique_threshold=25,
    data_desc_top_n_values=10,
):
    "create final prompt for the LLM to manipulate the Pandas data"
    data_dict_desc = create_data_dictionary(
        llm,
        llm._data,
        tools,
        tool_result,
        data_desc_unique_threshold=data_desc_unique_threshold,
        data_desc_top_n_values=data_desc_top_n_values,
    )

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
