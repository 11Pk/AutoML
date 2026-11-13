
from fastapi import APIRouter, UploadFile, File, HTTPException

from pathlib import Path
from uuid import uuid4

from backend.app.controllers.dataset_controller import (
    upload_dataset_controller
)


router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_DIR = BASE_DIR / "storage" / "datasets"

DATASET_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

MAX_FILE_SIZE = 50 * 1024 * 1024


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):

    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only CSV and Excel files are supported."
        )

    dataset_id = str(uuid4())

    file_path = DATASET_DIR / f"{dataset_id}{extension}"

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds 50 MB."
            )

        file_path.write_bytes(content)

        result = upload_dataset_controller(
            str(file_path)
        )

        return {
            "dataset_id": dataset_id,
            **result
        }

    except HTTPException:
        raise

    except Exception as e:

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    finally:
        await file.close()
