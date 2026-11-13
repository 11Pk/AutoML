
from pydantic import BaseModel
from typing import Literal


class ColumnConfig(BaseModel):

    # What should happen to this column?
    role: Literal["feature", "target", "drop"]

    # Validated by the user; may differ from detected type
    feature_type: Literal[
        "numerical",
        "categorical",
        "boolean",
        "datetime",
        "text",
        "identifier"
    ]

    # Categorical preprocessing
    encoding: Literal[
        "onehot",
        "ordinal",
        "frequency"
    ] | None = None

    # Text preprocessing
    text_method: Literal[
        "tfidf",
        "count",
        "text_stats"
    ] | None = None

    # Required when ordinal encoding needs a custom order
    ordinal_categories: list[str] | None = None

    class Config:
        extra = "forbid"


class DatasetConfigRequest(BaseModel):

    problem_type: Literal[
        "classification",
        "regression"
    ]

    # Key = column name, Value = its configuration
    columns: dict[str, ColumnConfig]

    class Config:
        extra = "forbid"
