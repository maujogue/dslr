import argparse
from data_handling.loader import load
from ml.data_engineering import pre_process_test
from ml.predict import predict_all
from ml.predict_bonus import predict_all_bonus


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Predict the logistic regression model"
        )
        parser.add_argument(
            "file",
            type=str,
            help="The file to predict the result on",
            default="datasets/dataset_test.csv",
            nargs="?",
        )
        parser.add_argument(
            "--bonus",
            action="store_true",
            help="Predict with multiple models and compare performance",
        )
        args = parser.parse_args()

        df = load(args.file)
        if df is None:
            exit(1)

        X, labels = pre_process_test(df)

        if args.bonus:
            predict_all_bonus(X, labels)
        else:
            predict_all(X, labels)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)
