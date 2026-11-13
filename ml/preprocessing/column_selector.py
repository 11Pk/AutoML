
def separate_columns(df, column_config):

    target_columns = [
        col for col, config in column_config.items()
        if config["role"] == "target"
    ]

    feature_columns = [
        col for col, config in column_config.items()
        if config["role"] == "feature"
    ]

    dropped_columns = [
        col for col, config in column_config.items()
        if config["role"] == "drop"
    ]

    # Exactly one target is required
    if len(target_columns) != 1:
        raise ValueError(
            "Exactly one target column must be selected."
        )

    if not feature_columns:
        raise ValueError(
            "At least one input feature is required."
        )

    target_column = target_columns[0]

    # Separate X and y
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    return {
        "X": X,
        "y": y,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "dropped_columns": dropped_columns
    }
