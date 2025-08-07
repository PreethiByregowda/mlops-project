# web_service/emotion_model/utils.py

from typing import Union
import pandas as pd

def prepare_features(input_data: Union[list[str], list[dict], pd.DataFrame]):
    """
    Prepare input text data for model prediction.
    Ensures consistent format for DictVectorizer-based pipeline.
    """
    if isinstance(input_data, list):
        if all(isinstance(x, str) for x in input_data):
            X = pd.DataFrame({"text": input_data})
        elif all(isinstance(x, dict) and "text" in x for x in input_data):
            X = pd.DataFrame(input_data)
        else:
            raise ValueError("Input list must contain either strings or dicts with 'text' key.")
    elif isinstance(input_data, pd.DataFrame):
        if "text" not in input_data.columns:
            raise ValueError("DataFrame must have a 'text' column.")
        X = input_data
    else:
        raise TypeError("Unsupported input type.")

    # Convert to dict format for DictVectorizer
    return X.to_dict(orient="records")


if __name__ == "__main__":
    sample_input = ["I'm really happy today!", "I feel terrible..."]
    X = prepare_features(sample_input)
    print(X)
