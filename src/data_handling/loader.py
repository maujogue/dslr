import logging
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)

        logger.info(f"Loading dataset of dimensions {df.shape}")

        return df

    except FileNotFoundError:
        logger.error(f"File '{path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"File '{path}' is empty.")
        return None
    except pd.errors.ParserError:
        logger.error(f"File '{path}' has bad format and cannot be parsed.")
        return None
    except PermissionError:
        logger.error(f"Permission denied to read file '{path}'.")
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading '{path}': {e}"
        )
        return None
