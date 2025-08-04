import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def display_pair_plot(df: pd.DataFrame):
    """
    Displaying this, we will keep those features for training: potions, charms,
    transfiguration, astronomy, divination, ancient runes, muggle studies,
    """
    if "Index" in df.columns:
        df = df.drop(columns=["Index"])

    sns.pairplot(
        df,
        diag_kind="hist",
        hue="Hogwarts House",
        plot_kws={"alpha": 0.6},
        corner=True,
        height=2,
    )
    plt.tight_layout()
    plt.show()
