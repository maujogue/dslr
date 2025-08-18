import argparse
import logging
import pandas as pd
from data_analysis.statistics import ft_describe
from data_handling.validator import parse_arguments, load_dataset
from data_handling.constants import BLUE, GREEN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_arguments_with_advanced() -> argparse.Namespace:
    additional_args = [
        (
            "-a",
            {
                "action": "store_true",
                "help": "Include advanced statistics (missing, unique, iqr)",
            },
        ),
    ]

    return parse_arguments(
        "Describe a dataset", additional_args=additional_args
    )


def display_statistics(df: pd.DataFrame, advanced: bool = False) -> None:
    print(f"\n{BLUE}Original describe function:")
    print(df.describe())

    print(f"\n{GREEN}Custom describe function:")
    ft_describe(df, advanced)


def main():
    args = parse_arguments_with_advanced()
    df = load_dataset(args.file)
    display_statistics(df, args.advanced)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)
