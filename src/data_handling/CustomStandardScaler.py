import pandas as pd


class CustomStandardScaler:
    """
    Custom standard scaler that scales the data to have a mean of 0 and a
    standard deviation of 1.
    """

    def fit(self, X: pd.DataFrame):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)

    def transform(self, X: pd.DataFrame):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: pd.DataFrame):
        return X * self.scale_ + self.mean_
