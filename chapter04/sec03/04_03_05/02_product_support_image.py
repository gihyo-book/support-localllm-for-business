def analyze_product_issue(
    image_path: str,
    user_description: str
) -> str:
    """製品の不具合画像とユーザーの説明から対応策を提案する"""
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="your-multimodal-model",
        messages=[
            {
                "role": "system",
                "content": """あなたは製品サポートの専門家です。
ユーザーから送られてきた製品の画像と説明をもとに、
考えられる原因と対応策を日本語で説明してください。
推測が含まれる場合はその旨を明記してください。"""
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
                        "text": f"製品の状態について説明します：{user_description}"
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content
