from langchain_core.tools import tool
import matplotlib.pyplot as plt
import pandas as pd


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

    df = pd.read_csv(df)

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

    # Legend if grouped
    if group_col:
        ax.legend(title=group_col)

    # Grid and clean style
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    return fig
