import argparse
import logging
import sys
import pandas as pd
from typing import Any
from data_handling.loader import load
from data_handling.constants import DEFAULT_DATASET

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_arguments(
    description: str,
    default_file: str = DEFAULT_DATASET,
    additional_args: list[tuple[str, dict[str, Any]]] = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "file",
        type=str,
        help="The file to process",
        default=default_file,
        nargs="?",
    )

    if additional_args:
        for arg_name, arg_kwargs in additional_args:
            parser.add_argument(arg_name, **arg_kwargs)

    try:
        return parser.parse_args()
    except SystemExit as e:
        logger.error("Invalid command line arguments.")
        sys.exit(e.code)


def load_dataset(file_path: str) -> pd.DataFrame:
    df = load(file_path)
    if df is None:
        logger.error(f"Failed to load file: {file_path}")
        sys.exit(1)

    return df


def validate_required_columns(
        df: pd.DataFrame,
        required_columns: list[str]) -> pd.DataFrame:
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        sys.exit(1)

    return df


def validate_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.select_dtypes(include=["number"])
    if num_df.empty:
        logger.error("No numerical columns found in the dataset.")
        sys.exit(1)

    logger.info(f"Found {len(num_df.columns)} numerical columns")
    return num_df


def validate_hogwarts_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = validate_required_columns(df, ["Hogwarts House"])
    validate_numerical_columns(df)
    return df
