
from sklearn.compose import ColumnTransformer

from ml.preprocessing.numerical_processor import (
    build_numerical_pipeline
)

from ml.preprocessing.categorical_preprocessor import (
    build_categorical_pipeline
)

from ml.preprocessing.text_preprocessor import (
    build_text_pipeline
)

from ml.preprocessing.date_time_preprocessor import (
    build_datetime_pipeline
)

from ml.preprocessing.boolean_preprocessor import (
    build_boolean_pipeline
)


def build_preprocessing_pipeline(X, column_config):

    transformers = []

    numerical_cols = []
    boolean_cols = []
    datetime_cols = []

    for col in X.columns:

        config = column_config[col]

        feature_type = config["feature_type"]

        # Numerical features
        if feature_type == "numerical":
            numerical_cols.append(col)

        # Boolean features
        elif feature_type == "boolean":
            boolean_cols.append(col)

        # Datetime features
        elif feature_type == "datetime":
            datetime_cols.append(col)

        # Categorical features
        elif feature_type == "categorical":

            encoding = config.get("encoding")

            if not encoding:
                raise ValueError(
                    f"Encoding not selected for {col}"
                )

            pipeline = build_categorical_pipeline(
                encoding=encoding,
                ordinal_categories=config.get(
                    "ordinal_categories"
                )
            )

            transformers.append(
                (f"{col}_categorical", pipeline, [col])
            )

        # Text features
        elif feature_type == "text":

            method = config.get("text_method")

            if not method:
                raise ValueError(
                    f"Text processing method not selected for {col}"
                )

            pipeline = build_text_pipeline(method)

            transformers.append(
                (f"{col}_text", pipeline, [col])
            )

        else:

            raise ValueError(
                f"Unsupported feature type for {col}: {feature_type}"
            )

    # Add numerical transformer
    if numerical_cols:

        transformers.append((
            "numerical",
            build_numerical_pipeline(),
            numerical_cols
        ))

    # Add boolean transformer
    if boolean_cols:

        transformers.append((
            "boolean",
            build_boolean_pipeline(),
            boolean_cols
        ))

    # Add datetime transformer
    if datetime_cols:

        transformers.append((
            "datetime",
            build_datetime_pipeline(),
            datetime_cols
        ))

    if not transformers:
        raise ValueError("No valid preprocessing transformations found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True
    )
