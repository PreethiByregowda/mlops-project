import pandas as pd

def process_data():
    """
    Load train and validation data from CSV files.
    Returns X_train, y_train, X_valid, y_valid.
    """
    train_df = pd.read_csv("data/train.csv")
    valid_df = pd.read_csv("data/valid.csv")

    X_train = train_df["text"]
    y_train = train_df["label"]

    X_valid = valid_df["text"]
    y_valid = valid_df["label"]

    # Wrap X as DataFrame for consistency in downstream pipelines
    return (
        pd.DataFrame(X_train, columns=["text"]),
        y_train.reset_index(drop=True),
        pd.DataFrame(X_valid, columns=["text"]),
        y_valid.reset_index(drop=True),
    )
