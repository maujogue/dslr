import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def display_scatter_plot(df: pd.DataFrame):
    """
    Displaying this, we can see a strong correlation between Defense Against
    the Dark Arts and Astronomy. We will only keep one of the two for training.
    """
    sns.scatterplot(
        x=df["Defense Against the Dark Arts"],
        y=df["Astronomy"],
        hue="Hogwarts House",
        data=df,
    )
    plt.show()
