import os
from dotenv import load_dotenv
from google import genai

def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not configured.")
    client = genai.Client(api_key=api_key)
    for model in client.models.list():
        print(model.name)


if __name__ == "__main__":
    main()
