def extract_invoice_info(invoice_text: str) -> dict:
    """請求書テキストから情報を抽出してJSONで返す"""
    response = client.chat.completions.create(
        model="qwen3:8b",
        messages=[
            {
                "role": "system",
                "content": """請求書のテキストから以下の情報を抽出し、必ずJSON形式で返してください。
{
  "company_name": "会社名",
  "invoice_date": "請求日（YYYY-MM-DD形式）",
  "amount": "請求金額（数値のみ）",
  "due_date": "支払期限（YYYY-MM-DD形式）",
  "items": ["品目のリスト"]
}
情報が見つからない場合はnullを設定してください。"""
            },
            {
                "role": "user",
                "content": invoice_text
            }
        ]
    )
    import json
    result_text = response.choices[0].message.content
    return json.loads(result_text)
