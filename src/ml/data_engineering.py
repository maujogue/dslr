import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from data_handling.CustomStandardScaler import CustomStandardScaler
from data_handling.constants import FEATURE_COLUMNS, HOUSES


def save_scaler(scaler, filepath="weights/scaler.pkl"):
    weights_dir = os.path.dirname(filepath)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {filepath}")


def load_scaler(filepath="weights/scaler.pkl"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scaler file not found: {filepath}")

    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {filepath}")
    return scaler


def pre_process(df):
    # Remove rows with NaN in features or target
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ["Hogwarts House"]).copy()

    scaler = CustomStandardScaler()
    X_standardized = scaler.fit_transform(df_clean[FEATURE_COLUMNS])

    # Save the fitted scaler for use during testing
    save_scaler(scaler)

    # Add bias term (column of ones)
    X = np.column_stack([np.ones(X_standardized.shape[0]), X_standardized])

    # Extract and encode target variable for one-versus-all classification
    Y_dict = {}

    for house in HOUSES:
        # Create binary target: 1 if student is in this house, 0 otherwise
        Y_dict[house] = (
            (df_clean["Hogwarts House"] == house)
            .astype(int)
            .values.reshape(-1, 1)
        )

    return X, Y_dict


def pre_process_test(df):
    # Remove rows with NaN in features
    df_clean = df.dropna(subset=FEATURE_COLUMNS).copy()
    labels = None
    if (
        "Hogwarts House" in df_clean.columns
        and not df_clean["Hogwarts House"].isnull().all()
    ):
        labels = df_clean["Hogwarts House"]

    # Load the scaler fitted during training
    scaler = load_scaler()
    X_standardized = scaler.transform(df_clean[FEATURE_COLUMNS])

    # Add bias term (column of ones)
    X = np.column_stack([np.ones(X_standardized.shape[0]), X_standardized])
    print("Dataset after pre-processing: ", X.shape)
    return X, labels


def split_train_validation(df, split):
    """
    Splits the input CSV into training and validation sets and saves them as
    new CSV files.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=split,
        random_state=42,
        shuffle=True,
        stratify=(
            df["Hogwarts House"] if "Hogwarts House" in df.columns else None
        ),
    )
    train_df.to_csv("datasets/Training_houses.csv", index=False)
    val_df.to_csv("datasets/Validation_houses.csv", index=False)

    return train_df, val_df
