from ml.data_analysis.data_loader import load_dataset
from ml.data_analysis.data_profiler import analyze_dataset


def process_dataset(file_path: str):

    # 1. Load dataset
    df = load_dataset(file_path)

    # 2. Analyze dataset
    profile = analyze_dataset(df)

    # 3. Return results
    return profile