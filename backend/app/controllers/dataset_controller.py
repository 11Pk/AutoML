from backend.app.services.dataset_analyze import process_dataset


def upload_dataset_controller(file_path: str):

    profile = process_dataset(file_path)

    return {
        "message": "Dataset uploaded and analyzed successfully",
        "profile": profile
    }