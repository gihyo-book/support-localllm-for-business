from openai import OpenAI

# OllamaのOpenAI互換API（ローカル）
client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",  # 必須だがローカルでは無視される
)

resp = client.chat.completions.create(
    model="qwen3:4b-instruct",
    messages=[
        {
            "role": "system",
            "content": """あなたは日本語のビジネスメール作成アシスタントです。
以下のルールを必ず守ってください。
- 返信本文のみを出力する（前置きや補足説明は不要）
- 件名から書き始める
- 署名欄に雛形（[あなたの名前]など）を入れない"""
        },
        {
            "role": "user",
            "content": """以下の内容でビジネスメールを作成してください。
    ・来週火曜15時の打ち合わせ日程を変更したい
    ・こちらの都合で変更
    ・来週中に再調整希望"""
        }
    ],
    temperature=0.2,
)
print(resp.choices[0].message.content)
