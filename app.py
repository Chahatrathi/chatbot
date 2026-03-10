import os
from google import genai
from google.genai import types

# 1. Initialize the Client correctly
# Use 'v1' for the stable production endpoint
client = genai.Client(
    api_key="YOUR_GEMINI_API_KEY",
    http_options={'api_version': 'v1'}
)

def get_equity_definition():
    try:
        # 2. Use a supported 2026 model name
        # Avoid the 'models/' prefix as the SDK adds it automatically
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents="What is equity in finance? Explain simply."
        )
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print(get_equity_definition())
