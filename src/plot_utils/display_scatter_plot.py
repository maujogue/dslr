import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools.constants import HOUSE_COLORS
from tools.dataset_utils import save_plot


def display_scatter_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.scatterplot(
        x=df["Defense Against the Dark Arts"],
        y=df["Astronomy"],
        hue="Hogwarts House",
        data=df,
        palette=HOUSE_COLORS,
        ax=ax
    )
    
    ax.set_title("Defense Against the Dark Arts vs Astronomy", fontsize=14, fontweight='bold')
    ax.set_xlabel("Defense Against the Dark Arts", fontsize=12)
    ax.set_ylabel("Astronomy", fontsize=12)
    
    save_plot(fig, "scatter_plot.png")
    plt.show()
