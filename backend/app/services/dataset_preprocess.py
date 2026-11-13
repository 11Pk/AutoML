
from ml.preprocessing.prepare_dataset import prepare_dataset
from ml.preprocessing.save_processed_dataset import (
    save_prepared_dataset
)


def prepare_uploaded_dataset(
    df,
    dataset_id,
    config
):

    # Convert Pydantic objects into dictionaries
    column_config = {
        col: details.model_dump(exclude_none=True)
        for col, details in config.columns.items()
    }

    # Call ML engine
    result = prepare_dataset(
        df=df,
        column_config=column_config,
        problem_type=config.problem_type
    )

    # Save prepared data and fitted transformers
    artifact_path = save_prepared_dataset(
        result,
        dataset_id
    )

    # Return metadata, not the actual arrays
    return {
        "dataset_id": dataset_id,
        "problem_type": result["problem_type"],
        "target_column": result["target_column"],
        "feature_columns": result["feature_columns"],
        "dropped_columns": result["dropped_columns"],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "processed_features": result["processed_features"],
        "artifact_path": artifact_path
    }
