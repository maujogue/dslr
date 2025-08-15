import os
import matplotlib.pyplot as plt


def save_plot(fig: plt.Figure, filename: str) -> None:
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {filepath}")
