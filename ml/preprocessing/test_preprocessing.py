
import pandas as pd

from ml.preprocessing.prepare_dataset import (
    prepare_dataset
)


df = pd.DataFrame({

    "Age": [
        23, 31, 25, 40, 35, 29,
        45, 28, 33, 38, 26, 42
    ],

    "Gender": [
        "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male"
    ],

    "Review": [
        "Great product",
        "Poor quality",
        "Worth buying",
        "Not recommended",
        "Amazing experience",
        "Bad experience",
        "Very useful",
        "Waste of money",
        "Excellent",
        "Terrible",
        "Good product",
        "Not worth it"
    ],

    "Salary": [
        45000, 60000, 52000, 70000,
        48000, 55000, 80000, 51000,
        62000, 68000, 47000, 75000
    ],

    "Purchased": [
        "Yes", "No", "Yes", "No",
        "Yes", "No", "Yes", "No",
        "Yes", "No", "Yes", "No"
    ]

})


column_config = {

    "Age": {
        "role": "feature",
        "feature_type": "numerical"
    },

    "Gender": {
        "role": "feature",
        "feature_type": "categorical",
        "encoding": "onehot"
    },

    "Review": {
        "role": "feature",
        "feature_type": "text",
        "text_method": "tfidf"
    },

    "Salary": {
        "role": "drop",
        "feature_type": "numerical"
    },

    "Purchased": {
        "role": "target",
        "feature_type": "categorical"
    }

}


result = prepare_dataset(
    df=df,
    column_config=column_config,
    problem_type="classification"
)


print("Training shape:", result["X_train"].shape)
print("Testing shape:", result["X_test"].shape)

print("Training labels:", result["y_train"])
print("Testing labels:", result["y_test"])

print("Dropped columns:", result["dropped_columns"])

print("Processed features:", result["processed_features"])
