import os
from openai import OpenAI

# 🔑 Replace this with your Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key")

client = OpenAI(api_key=GEMINI_API_KEY)

try:
    # ✅ Simple test — list available models
    models = client.models.list()
    print("✅ API Key is valid!")
    print("Here are some available models:")
    for m in models.data[:5]:
        print("-", m.id)

    # ✅ Test a quick completion
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from OpenAI API test!'"}
        ]
    )
    print("\nAI Response:", response.choices[0].message.content)

except Exception as e:
    print("❌ API Key test failed!")
    print("Error details:", e)
