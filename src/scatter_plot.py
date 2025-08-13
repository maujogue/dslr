import logging
from plot_utils.display_scatter_plot import display_scatter_plot
from tools.dataset_utils import (
    parse_arguments,
    load_dataset,
    validate_hogwarts_dataset
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    args = parse_arguments("Display a scatter plot of a dataset")
    df = load_dataset(args.file)
    df = validate_hogwarts_dataset(df)
    display_scatter_plot(df)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)
