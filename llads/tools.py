from langchain_core.tools import tool
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
