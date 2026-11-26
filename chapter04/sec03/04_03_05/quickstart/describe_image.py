import argparse
import base64
from pathlib import Path

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="画像の内容を日本語で説明します。")
    parser.add_argument("image", type=Path, help="PNGまたはJPEG画像")
    parser.add_argument("--model", default="gemma3:4b")
    args = parser.parse_args()

    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_types.get(args.image.suffix.lower())
    if mime_type is None:
        parser.error("PNGまたはJPEG画像を指定してください。")
    try:
        encoded = base64.b64encode(args.image.read_bytes()).decode("ascii")
    except OSError as error:
        parser.error(str(error))

    client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "この画像に写っている内容を日本語で説明してください。"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                ],
            }
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
