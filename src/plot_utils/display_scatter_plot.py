import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def display_scatter_plot(df: pd.DataFrame):
    sns.scatterplot(
        x=df["Defense Against the Dark Arts"],
        y=df["Astronomy"],
        hue="Hogwarts House",
        data=df,
    )
    plt.show()
