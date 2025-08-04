import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def display_histogram(df: pd.DataFrame):
    if "Hogwarts House" not in df.columns:
        print("Error: 'Hogwarts House' column not found in the dataset.")
        exit(1)

    houses = df["Hogwarts House"].dropna().unique()

    if "Index" in df.columns:
        df = df.drop(columns=["Index"])

    num_df = df.select_dtypes(include=["number"])

    # create a single plot with all the histograms
    n_cols = 5
    n_rows = int(np.ceil(len(num_df.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    # plot histograms for each column, for each house
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
            )
        ax.set_title(f"{col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")

    # Add a single legend for all houses outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title="House", prop={"size": 15}, loc="lower right"
    )
    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
