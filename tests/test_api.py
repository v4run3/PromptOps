import requests
import time

def test_api():
    url = "http://127.0.0.1:8000/api/summarize"
    payload = {
        "dialogue": "John: Hey, did you see the game?\nSarah: Yeah, it was crazy!",
        "num_beams": 3
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Verify fallback message
        assert "Model checkpoint not found" in response.json()["summary"]
    except Exception as e:
        print(f"API Test Failed: {e}")

if __name__ == "__main__":
    # Wait for server to be ready
    time.sleep(2)
    test_api()
