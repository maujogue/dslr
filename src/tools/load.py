import pandas as pd


def load(path: str) -> pd.DataFrame:
    """
    Load a CSV file using pandas and print its dimensions.

    Args:
        path (str): Path to the CSV file

    Returns:
        pd.DataFrame or None: The loaded DataFrame if successful,
                             None if there's an error
    """
    try:
        df = pd.read_csv(path)

        print(f"Loading dataset of dimensions {df.shape}")

        return df

    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: File '{path}' has bad format and cannot be parsed.")
        return None
    except PermissionError:
        print(f"Error: Permission denied to read file '{path}'.")
        return None
    except Exception as e:
        print(
            f"Error: An unexpected error occurred while loading '{path}': {e}"
        )
        return None
