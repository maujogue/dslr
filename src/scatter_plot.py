import argparse
from tools.load import load
from plot_utils.display_scatter_plot import display_scatter_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display a scatter plot of a dataset"
    )
    parser.add_argument(
        "file",
        type=str,
        help="The file to display the scatter plot of",
        default="datasets/dataset_train.csv",
        nargs="?",
    )
    args = parser.parse_args()

    df = load(args.file)
    if df is None:
        exit(1)
    display_scatter_plot(df)
