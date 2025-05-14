from langchain_core.tools import tool
import matplotlib.pyplot as plt
import pandas as pd
import requests


@tool
def get_world_bank_gdp_data(
    country_code: str, start_year: int, end_year: int
) -> pd.DataFrame:
    """
    Fetch GDP data from World Bank API for a specific country

    Parameters:
        country_code (str): ISO 3-letter country code
        start_year (int): start year of the data
        end_year (int): end year of the data

    Returns:
        pandas.DataFrame: DataFrame containing the GDP data
    """
    indicator = "NY.GDP.MKTP.CD"

    # Build the API URL
    base_url = (
        f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
    )
    params = {
        "format": "json",
        "per_page": 100,  # Maximum number of results per page
        "date": f"{str(start_year)}:{str(end_year)}",  # Data range from 1960 to most recent available
    }

    # Make the API request
    response = requests.get(base_url, params=params)

    # Check if request was successful
    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        return None

    # Parse JSON response
    data = response.json()

    # The actual data is in the second element of the returned list
    if len(data) < 2:
        print("Error: No data returned from API")
        return None

    records = data[1]

    # Create a list to store the data
    gdp_data = []

    for record in records:
        if record["value"] is not None:  # Some years might not have data
            gdp_data.append(
                {
                    "Year": record["date"],
                    "GDP (current US$)": record["value"],
                    "Country": record["country"]["value"],
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(gdp_data)

    # Convert Year to integer and sort by year
    df["Year"] = df["Year"].astype(int)
    df = df.sort_values("Year")

    # Reset index
    df = df.reset_index(drop=True)

    return df


@tool
def gen_plot(
    df,
    x_col: str,
    y_col: str,
    group_col: str = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    plot_type: str = "line",
):
    """
    Creates a line plot or bar plot.

    Parameters:
    - df: pandas.DataFrame in long format
    - x_col: column name for x-axis
    - y_col: column name for y-axis
    - group_col: column name for grouping (optional)
    - title: plot title
    - xlabel: label for x-axis
    - ylabel: label for y-axis
    - plot_type: 'line' or 'bar'
    """

    try:
        df = pd.read_csv(df)
    except:
        pass

    fig, ax = plt.subplots(figsize=(8, 6))

    # If grouped
    if group_col:
        groups = df[group_col].unique()
        for group in groups:
            subset = df[df[group_col] == group]
            if plot_type == "line":
                ax.plot(subset[x_col], subset[y_col], marker="o", label=str(group))
            elif plot_type == "bar":
                # Offset bar positions for grouped bar plot
                x_vals = list(sorted(df[x_col].unique()))
                width = 0.8 / len(groups)
                idx = list(groups).index(group)
                offset = [-0.4 + width / 2 + i * width for i in range(len(groups))]
                bar_positions = [x + offset[idx] for x in range(len(x_vals))]
                y_vals = [
                    (
                        subset[subset[x_col] == x][y_col].values[0]
                        if not subset[subset[x_col] == x].empty
                        else 0
                    )
                    for x in x_vals
                ]
                ax.bar(
                    bar_positions, y_vals, width=width, label=str(group), align="center"
                )
                ax.set_xticks(range(len(x_vals)))
                ax.set_xticklabels(x_vals)
            else:
                raise ValueError("plot_type must be 'line' or 'bar'")
    else:
        if plot_type == "line":
            ax.plot(df[x_col], df[y_col], marker="o")
        elif plot_type == "bar":
            ax.bar(df[x_col], df[y_col])
        else:
            raise ValueError("plot_type must be 'line' or 'bar'")

    # Labels and title
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # rotate x-axis labels 45 degrees
    ax.tick_params(axis="x", rotation=90)

    # Legend if grouped
    if group_col:
        ax.legend(title=group_col)

    # Grid and clean style
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    return fig
