import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools.constants import HOUSE_COLORS
from tools.dataset_utils import save_plot


def display_pair_plot(df: pd.DataFrame):
    if "Index" in df.columns:
        df = df.drop(columns=["Index"])

    pair_plot = sns.pairplot(
        df,
        diag_kind="hist",
        hue="Hogwarts House",
        plot_kws={"alpha": 0.6},
        corner=True,
        height=2,
        palette=HOUSE_COLORS
    )
    
    save_plot(pair_plot.figure, "pair_plot.png")
    plt.tight_layout()
    plt.show()
