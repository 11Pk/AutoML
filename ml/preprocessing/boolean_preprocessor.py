
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



# custom scikit-learn transformer to convert boolean values to numeric
class BooleanConverter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        mapping = {
            True: 1,
            False: 0,
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0
        }

        for col in X.columns:

            X[col] = (
                X[col]
                .map(
                    lambda v: mapping.get(
                        v if isinstance(v, bool)
                        else str(v).strip().lower(),
                        np.nan
                    )
                )
            )

        return X.to_numpy(dtype=float)


def build_boolean_pipeline():

    return Pipeline([
        ("convert", BooleanConverter()),
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        )
    ])
