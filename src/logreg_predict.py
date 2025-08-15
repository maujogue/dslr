import argparse
from data_handling.loader import load
from ml.data_engineering import pre_process_test
from ml.predict import (
    load_weights,
    predict,
    save_predictions,
    print_precision,
)


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
        args = parser.parse_args()

        df = load(args.file)
        if df is None:
            exit(1)

        X, labels = pre_process_test(df)

        weights = load_weights()
        predictions = predict(X, weights)
        if labels is not None:
            save_predictions(predictions)
            print_precision(predictions, labels)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)
