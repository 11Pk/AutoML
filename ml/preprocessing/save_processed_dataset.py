
import joblib
from pathlib import Path


ARTIFACT_DIR = Path("storage/prepared_datasets")


def save_prepared_dataset(result, dataset_id):

    dataset_dir = ARTIFACT_DIR / dataset_id

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    artifact = {
        "X_train": result["X_train"],
        "X_test": result["X_test"],
        "y_train": result["y_train"],
        "y_test": result["y_test"],
        "preprocessor": result["preprocessor"],
        "label_encoder": result["label_encoder"],
        "problem_type": result["problem_type"],
        "target_column": result["target_column"],
        "feature_columns": result["feature_columns"],
        "dropped_columns": result["dropped_columns"]
    }

    artifact_path = dataset_dir / "prepared_data.pkl"

    joblib.dump(artifact, artifact_path)

    return str(artifact_path)


def load_prepared_dataset(dataset_id):

    artifact_path = (
        ARTIFACT_DIR /
        dataset_id /
        "prepared_data.pkl"
    )

    if not artifact_path.exists():
        raise FileNotFoundError(
            "Prepared dataset not found."
        )

    return joblib.load(artifact_path)
