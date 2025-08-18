import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_handling.constants import HOUSE_COLORS
from data_analysis.saver import save_plot


def prepare_plot_data(df):
    if "Index" in df.columns:
        df = df.drop(columns=["Index"])
    return df


def create_subplot_grid(num_columns):
    n_cols = 5
    n_rows = int(np.ceil(num_columns / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    return fig, axes.flatten()


def plot_house_histograms(df, num_df, houses, axes):
    for idx, col in enumerate(num_df.columns):
        ax = axes[idx]
        for house in houses:
            house_data = df[df["Hogwarts House"] == house][col].dropna()
            sns.histplot(
                house_data,
                label=house,
                kde=False,
                stat="density",
                bins=30,
                alpha=0.5,
                ax=ax,
                color=HOUSE_COLORS[house],
            )
        ax.set_title(f"{col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
    return idx


def finalize_plot(fig, axes, last_idx):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title="House", prop={"size": 15}, loc="lower right"
    )

    for j in range(last_idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()


def display_histogram(df):
    df = prepare_plot_data(df)

    houses = df["Hogwarts House"].dropna().unique()
    num_df = df.select_dtypes(include=["number"])

    fig, axes = create_subplot_grid(len(num_df.columns))
    last_idx = plot_house_histograms(df, num_df, houses, axes)
    finalize_plot(fig, axes, last_idx)

    save_plot(fig, "histogram.png")
    plt.show()
