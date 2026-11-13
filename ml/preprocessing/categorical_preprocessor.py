
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        X = pd.DataFrame(X)

        self.frequency_maps_ = {}

        for col in X.columns:

            values = X[col].astype(str)

            self.frequency_maps_[col] = (
                values.value_counts(normalize=True).to_dict()
            )

        return self

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        for col in X.columns:

            values = X[col].astype(str)

            X[col] = values.map(
                self.frequency_maps_[col]
            ).fillna(0)

        return X.to_numpy(dtype=float)


def build_categorical_pipeline(
    encoding,
    ordinal_categories=None
):

    if encoding == "onehot":

        encoder = OneHotEncoder(
            handle_unknown="ignore"
        )

    elif encoding == "ordinal":

        encoder = OrdinalEncoder(
            categories=(
                [ordinal_categories]
                if ordinal_categories is not None
                else "auto"
            ),
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    elif encoding == "frequency":

        encoder = FrequencyEncoder()

    else:

        raise ValueError(
            f"Unsupported categorical encoding: {encoding}"
        )

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            encoder
        )
    ])
