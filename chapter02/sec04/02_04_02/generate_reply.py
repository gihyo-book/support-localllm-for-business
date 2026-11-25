#!/usr/bin/env python3
"""
2.4.2 返信メール生成（baseline）

シンプルなプロンプトでメール返信案を生成します。
"""
import argparse
import json
import sys
from typing import Any

from openai import OpenAI


# ── プロンプト ───────────────────────────────────────────

SYSTEM_PROMPT = "あなたは日本語のビジネスメール返信アシスタントです。返信本文のみを出力してください。"


def build_user_prompt(mail: dict[str, Any]) -> str:
    return f"""以下の受信メールに返信してください。

[件名]
{mail.get("subject", "")}

[本文]
{mail.get("body", "")}
"""


# ── 入出力 ───────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="返信メール生成（baseline）: シンプルなプロンプトで返信案を生成します。"
    )
    parser.add_argument("--input", "-i", help="入力ファイルのパス（JSON/JSONL）。省略時はstdinから読み込みます。")
    parser.add_argument(
        "--input-format", choices=["auto", "json", "jsonl"], default="auto",
        help="入力フォーマット。autoの場合はファイル拡張子で判定します。",
    )
    parser.add_argument(
        "--output-format", choices=["json", "jsonl"], default="json",
        help="出力フォーマット。",
    )
    parser.add_argument("--model", "-m", default="qwen3:4b-instruct", help="使用するモデル名。")
    parser.add_argument(
        "--base-url", default="http://localhost:11434/v1/",
        help="OllamaのOpenAI互換APIのベースURL。",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="サンプリング温度。")
    return parser.parse_args()


def detect_input_format(path: str | None, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    if path and path.lower().endswith(".jsonl"):
        return "jsonl"
    return "json"


def load_records(path: str | None, input_format: str) -> list[dict[str, Any]]:
    fmt = detect_input_format(path, input_format)
    if fmt == "jsonl":
        src = open(path, "r", encoding="utf-8") if path else sys.stdin
        with src as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        src = open(path, "r", encoding="utf-8") if path else sys.stdin
        with src as f:
            loaded = json.load(f)
        rows = loaded if isinstance(loaded, list) else [loaded]
    if not all(isinstance(r, dict) for r in rows):
        raise ValueError("入力はJSONオブジェクトまたはオブジェクトの配列/JSONLである必要があります。")
    return rows


def output_results(rows: list[dict[str, Any]], output_format: str) -> None:
    if output_format == "jsonl":
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
    else:
        print(json.dumps(rows, ensure_ascii=False, indent=2))


# ── メイン ───────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key="ollama")
    mails = load_records(args.input, args.input_format)

    results = []
    for mail in mails:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(mail)},
            ],
            temperature=args.temperature,
        )
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("モデルが空のレスポンスを返しました。")
        results.append({"id": mail.get("id"), "reply_draft": content.strip()})

    output_results(results, args.output_format)


if __name__ == "__main__":
    main()