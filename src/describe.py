import sys
from describe_utils.statistics import ft_describe
from tools.load import load
from tools.constants import BLUE, GREEN

if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser(description="Describe a dataset")
        parser.add_argument(
            "file",
            type=str,
            help="The file to describe",
            default="datasets/dataset_train.csv",
            nargs="?",
        )
        parser.add_argument(
            "--advanced",
            "-a",
            action="store_true",
            help="Include advanced statistics (missing, unique, iqr)",
        )

        try:
            args = parser.parse_args()
        except SystemExit as e:
            print("Error: Invalid command line arguments.")
            sys.exit(e.code)

        df = load(args.file)
        if df is None:
            sys.exit(1)

        print(f"\n{BLUE}Original describe function:")

        print(df.describe())

        print(f"\n{GREEN}Custom describe function:")
        ft_describe(df, args.advanced)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
