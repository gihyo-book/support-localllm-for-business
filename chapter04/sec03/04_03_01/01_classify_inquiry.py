from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy")
def classify_email(email_text: str) -> str:
    """メールテキストをカテゴリに分類する"""
    response = client.chat.completions.create(
        model="qwen3:8b",
        messages=[
            {
                "role": "system",
                "content": "あなたはメール分類の専門家です。メールの内容を以下のカテゴリのいずれかに分類してください：製品不具合、使い方の質問、解約希望、その他"
            },
            {
                "role": "user",
                "content": email_text
            }
        ]
    )
    return response.choices[0].message.content
def process_emails(emails: list[str]) -> list[dict]:
    """メールのリストを一括処理する"""
    results = []
    for i, email in enumerate(emails):
        try:
            category = classify_email(email)
            results.append({"index": i, "category": category, "status": "success"})
        except Exception as e:
            results.append({"index": i, "category": None, "status": "error", "error": str(e)})
    return results
