import base64
import json
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:11434/v1",  # セルフホスティングしたマルチモーダルモデルのエンドポイント
    api_key="your-api-key"
)
def encode_image(image_path: str) -> str:
    """画像ファイルをBase64エンコードする"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
def extract_receipt_info(image_path: str) -> dict:
    """領収書画像から情報を抽出する"""
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="your-multimodal-model",
        messages=[
            {
                "role": "system",
                "content": """領収書の画像から情報を抽出し、必ずJSON形式で返してください。
{
  "store_name": "店舗名",
  "date": "日付（YYYY-MM-DD形式）",
  "total_amount": "合計金額（数値のみ）",
  "tax_amount": "消費税額（数値のみ）",
  "items": [
    {"name": "品目名", "price": "金額（数値のみ）"}
  ]
}
情報が読み取れない場合はnullを設定してください。"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "この領収書から情報を抽出してください。"
                    }
                ]
            }
        ]
    )
    return json.loads(response.choices[0].message.content)
