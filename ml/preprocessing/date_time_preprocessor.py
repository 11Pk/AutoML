
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# custom scikit-learn transformer to extract date and time features
class DateTimeExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = pd.DataFrame(X)

        result = []

        for col in X.columns:

            dates = pd.to_datetime(
                X[col],
                errors="coerce"
            )

            result.extend([
                dates.dt.year,
                dates.dt.month,
                dates.dt.day,
                dates.dt.dayofweek
            ])

        return pd.concat(
            result,
            axis=1
        ).to_numpy(dtype=float)


def build_datetime_pipeline():

    return Pipeline([
        ("extract", DateTimeExtractor()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
