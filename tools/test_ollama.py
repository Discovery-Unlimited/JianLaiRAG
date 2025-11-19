import requests
import json
import sys

def test_ollama_connection(model_name="gpt-oss:120b-cloud"):
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    user_prompt = input("请输入提示词 (默认: '休谟是谁？'): ").strip()
    if not user_prompt:
        user_prompt = "休谟是谁？"

    data = {
        "model": model_name,
        "prompt": user_prompt,
        "stream": False
    }

    print(f"Testing connection to Ollama at {url} with model '{model_name}'...")

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        print("\nSuccess! Response from Ollama:")
        print("-" * 50)
        print(result.get("response", "No response text found"))
        print("-" * 50)
        print(f"Total duration: {result.get('total_duration', 0) / 1e9:.2f} seconds")
        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to Ollama. Is it running on http://localhost:11434?")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\nHTTP Error: {e}")
        print(f"Response content: {response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_ollama_connection()
