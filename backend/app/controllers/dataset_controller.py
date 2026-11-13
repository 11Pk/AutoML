from backend.app.services.dataset_analyze import process_dataset

from fastapi import HTTPException

from app.services.dataset_preprocess import (
    prepare_uploaded_dataset
)

from ml.data_analysis.data_loader import load_dataset

from pathlib import Path


STORAGE_PATH = Path("storage/datasets")


def prepare_dataset_controller(
    dataset_id,
    config
):

    # 1. Locate uploaded dataset
    file_path = STORAGE_PATH / f"{dataset_id}.csv"

    if not file_path.exists():
        raise FileNotFoundError("Uploaded dataset not found.")

    # 2. Load original dataset
    df = load_dataset(str(file_path))

    # 3. Call service
    result = prepare_uploaded_dataset(
        df=df,
        dataset_id=dataset_id,
        config=config
    )

    return result


def upload_dataset_controller(file_path: str):

    profile = process_dataset(file_path)

    return {
        "message": "Dataset uploaded and analyzed successfully",
        "profile": profile
    }