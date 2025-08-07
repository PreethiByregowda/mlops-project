import json
import requests

def test_single_text():
    sample_text = {
        "text": "I feel fantastic and everything is going great!"
    }
    host = "http://127.0.0.1:9696"
    url = f"{host}/predict"
    
    response = requests.post(url, json=sample_text)
    try:
        result = response.json()
        print("Single text prediction result:", result)
    except json.JSONDecodeError:
        print("Failed to decode JSON response. Status code:", response.status_code)
        print("Raw response:", response.text)

def test_batch_text():
    sample_texts = {
        "text": [
            "I feel miserable and alone",            # sadness
            "I'm so angry with what happened",       # anger
            "This is terrifying news",               # fear
            "I love spending time with my family",   # love
            "Wow! I wasn't expecting that at all!",  # surprise
        ]
    }
    host = "http://127.0.0.1:9696"
    url = f"{host}/predict"

    response = requests.post(url, json=sample_texts)
    try:
        result = response.json()
        print("Batch text prediction result:", result)
    except json.JSONDecodeError:
        print("Failed to decode JSON response. Status code:", response.status_code)
        print("Raw response:", response.text)

if __name__ == "__main__":
    test_single_text()
    test_batch_text()
