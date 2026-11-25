from openai import OpenAI

# OllamaのOpenAI互換API（ローカル）
client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",  # 必須だがローカルでは無視される
)

resp = client.chat.completions.create(
    model="qwen3:4b-instruct",
    messages=[
        {"role": "user", "content": "こんにちは。日本語で自己紹介してください。"},
    ],
    temperature=0.7,
)

print(resp.choices[0].message.content)
