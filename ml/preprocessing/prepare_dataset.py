
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ml.preprocessing.column_selector import (
    separate_columns
)

from ml.preprocessing.preprocessing_pipeline import (
    build_preprocessing_pipeline
)


def prepare_dataset(
    df,
    column_config,
    problem_type,
    test_size=0.2,
    random_state=42
):

    # 1. Validate column configuration
    if set(column_config.keys()) != set(df.columns):
        raise ValueError(
            "Configuration must contain every dataset column exactly once."
        )

    for col, config in column_config.items():

        if config["role"] == "feature" and config["feature_type"] == "identifier":
            raise ValueError(
                f"Identifier column '{col}' cannot be used as a feature."
            )

    # 2. Separate X and y
    separated = separate_columns(
        df,
        column_config
    )

    X = separated["X"]
    y = separated["y"]

    # 3. Remove rows with missing target
    valid_rows = y.notna()

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    # 4. Validate target
    if problem_type == "regression":

        y = pd.to_numeric(y, errors="coerce")

        valid_rows = y.notna()

        X = X.loc[valid_rows].copy()
        y = y.loc[valid_rows].copy()

        if not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError(
                "Regression target contains infinite values."
            )

        if y.nunique() < 2:
            raise ValueError(
                "Regression requires at least two distinct target values."
            )

    elif problem_type == "classification":

        y = y.astype(str)

        if y.nunique() < 2:
            raise ValueError(
                "Classification requires at least two classes."
            )

    else:

        raise ValueError("Invalid problem type.")

    if len(X) < 2:
        raise ValueError("Not enough valid rows to split.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    # 5. Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        X,
        column_config
    )

    # 6. Train-test split
    stratify = None

    if problem_type == "classification":

        class_counts = y.value_counts()

        n_classes = len(class_counts)
        n_test = int(np.ceil(len(y) * test_size))
        n_train = len(y) - n_test

        if class_counts.min() < 2:
            raise ValueError(
                "Every class must have at least 2 samples "
                "for train-test splitting."
            )

        if n_test < n_classes or n_train < n_classes:
            raise ValueError(
                "Not enough samples for a stratified split. "
                "Increase dataset size or adjust test_size."
            )

        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # 7. Encode classification target separately
    label_encoder = None

    if problem_type == "classification":

        label_encoder = LabelEncoder()

        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

    # 8. Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # 9. Transform testing data
    X_test_processed = preprocessor.transform(
        X_test
    )

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,

        "preprocessor": preprocessor,
        "label_encoder": label_encoder,

        "target_column": separated["target_column"],
        "feature_columns": separated["feature_columns"],
        "dropped_columns": separated["dropped_columns"],

        "problem_type": problem_type,

        "train_rows": X_train_processed.shape[0],
        "test_rows": X_test_processed.shape[0],
        "processed_features": X_train_processed.shape[1]
    }
