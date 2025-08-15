import argparse
from data_handling.loader import load
from ml.data_engineering import pre_process, split_train_validation
from ml.train import train_all

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Train the logistic regression model"
        )
        parser.add_argument(
            "file",
            type=str,
            help="The file to train the model on",
            default="datasets/dataset_train.csv",
            nargs="?",
        )
        parser.add_argument(
            "--split",
            type=float,
            help="The split ratio for the training and validation sets",
            const=0.2,
            nargs="?",
        )
        args = parser.parse_args()

        df = load(args.file)
        if df is None:
            exit(1)

        if args.split is not None:
            df_train, df_val = split_train_validation(df, args.split)
        else:
            df_train = df
        X, Y_dict = pre_process(df_train)

        train_all(X, Y_dict, 10000, 0.001)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
