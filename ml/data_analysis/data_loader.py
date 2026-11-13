import pandas as pd
from pathlib import Path


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV or Excel dataset and returns a DataFrame.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError("Dataset file not found.")

    extension = file_path.suffix.lower()

    if extension == ".csv":
        df = pd.read_csv(file_path)

    elif extension in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    else:
        raise ValueError("Unsupported file format.")

    if df.empty:
        raise ValueError("Uploaded dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("Dataset contains no columns.")

    return df
