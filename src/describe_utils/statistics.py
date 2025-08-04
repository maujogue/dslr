from typing import Any

import pandas as pd


def check_args(args: Any) -> bool:
    """
    Check if the arguments are valid numbers.
    """
    if not args:
        raise ValueError("ERROR")
    if any(not isinstance(arg, (float, int)) for arg in args):
        raise ValueError("Value Error: All args must be numbers")
    if any(
        arg != arg or arg == float("inf") or arg == float("-inf")
        for arg in args
    ):
        raise ValueError("Value Error: All args must be valid numbers")


def calculate_quartile(numbers: list, percentile: float) -> float:
    """
    Calculate quartile using linear interpolation.
    percentile should be 0.25 for Q1 and 0.75 for Q3
    """
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    pos = (n - 1) * percentile
    i = int(pos)
    frac = pos - i
    if i == n - 1:
        return float(sorted_numbers[i])
    difference = sorted_numbers[i + 1] - sorted_numbers[i]
    return float(sorted_numbers[i] + frac * difference)


def calculate_stats(numerical_df: pd.DataFrame, advanced: bool = False):
    """
    Calculate statistics for numerical columns in the DataFrame.
    Returns (columns, stat_names, stats) tuple.

    Args:
        numerical_df: DataFrame with numerical columns
        advanced: If True, includes missing, unique, and iqr statistics
    """
    columns = list(numerical_df.columns)

    # Basic statistics (always included)
    basic_stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    # Advanced statistics (only when advanced=True)
    advanced_stats = ["missing", "unique", "iqr"]

    if advanced:
        stat_names = basic_stats + advanced_stats
    else:
        stat_names = basic_stats

    stats = {name: [] for name in stat_names}

    for column in columns:
        try:
            col = numerical_df[column]
            col_values = col.dropna().tolist()

            # Handle case where column has only empty/NaN values
            if not col_values:
                for name in stat_names:
                    if name != "count" and name != "missing":
                        stats[name].append("NaN")
                stats["count"].append(0)
                if "missing" in stat_names:
                    stats["missing"].append(col.isna().sum())
                continue

            check_args(col_values)

            count = len(col_values)
            mean = sum(col_values) / count
            variance = sum([(x - mean) ** 2 for x in col_values]) / count
            std = variance**0.5
            min_val = min(col_values)
            q1 = calculate_quartile(col_values, 0.25)
            median_val = calculate_quartile(col_values, 0.5)
            q3 = calculate_quartile(col_values, 0.75)
            max_val = max(col_values)
            missing_val = col.isna().sum()
            unique_val = len(set(col_values))
            iqr_val = q3 - q1

            # Always add basic statistics
            stats["count"].append(count)
            stats["mean"].append(mean)
            stats["std"].append(std)
            stats["min"].append(min_val)
            stats["25%"].append(q1)
            stats["50%"].append(median_val)
            stats["75%"].append(q3)
            stats["max"].append(max_val)

            # Add advanced statistics only if requested
            if advanced:
                stats["missing"].append(missing_val)
                stats["unique"].append(unique_val)
                stats["iqr"].append(iqr_val)
        except ValueError as e:
            for name in stat_names:
                stats[name].append("NaN")
            print(f"Error processing column '{column}': {e}")
            print()
    return columns, stat_names, stats


def print_stats(columns, stat_names, stats):
    """
    Print the statistics in a formatted table.
    """
    header = [""] + columns

    # Determine the width for each column:
    # max of column name length and max value length in that column
    col_widths = []
    for idx, col in enumerate(columns):
        col_name_len = len(str(col))
        max_val_len = col_name_len
        for stat in stat_names:
            val = stats[stat][idx]
            if isinstance(val, (float, int)):
                val_str = f"{val:.6f}"
            else:
                val_str = str(val)
            if len(val_str) > max_val_len:
                max_val_len = len(val_str)
        max_val_len += 1
        col_widths.append(max_val_len)

    stat_col_width = max(len(stat) for stat in stat_names)

    # Build the row format string dynamically
    row_format = f"{{:<{stat_col_width}}}"
    for w in col_widths:
        row_format += f" {{:>{w}}}"

    # Print header with the format
    print(row_format.format(*header))
    for stat in stat_names:
        row = [stat]
        for val in stats[stat]:
            if isinstance(val, (float, int)):
                row.append(f"{val:.6f}")
            else:
                row.append(str(val))
        # Print rows with the format
        print(row_format.format(*row))
    print()


def ft_describe(df: pd.DataFrame, advanced: bool = False) -> None:
    """
    Calculate statistics for numerical columns in the DataFrame.

    Args:
        df: Input pandas DataFrame
        advanced: If True, includes missing, unique, and iqr statistics
    """
    if df.empty:
        print("Error: No dataframe")
        return

    numerical_df = df.select_dtypes(include=["number"])

    if numerical_df.empty:
        print("Error: No numerical columns found in dataframe")
        return

    columns, stat_names, stats = calculate_stats(numerical_df, advanced)
    print_stats(columns, stat_names, stats)
