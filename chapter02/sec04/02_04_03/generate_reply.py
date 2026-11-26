#!/usr/bin/env python3
"""
2.4.3 返信メール生成（改善版）

--mode structured_prompt : プロンプトを丁寧に構造化して1回のリクエストで生成します。
--mode two_step          : 下書き生成→自己レビューの2ステップで生成します。
"""
import argparse
import json
import re
import sys
from typing import Any

from openai import OpenAI


# ── プロンプト（structured_prompt） ──────────────────────

STRUCTURED_SYSTEM = """あなたは日本語のビジネスメール担当者です。
以下のルールを必ず守ってください。

- 日本語のみで応答する
- 返信本文のみを出力する（件名・説明・補足は不要）
- 「お世話になっております。」で始める
- 受信メールの依頼・確認・共有事項をすべて拾う
- 自然な文章で書く（箇条書き不可）
- 3〜7文程度にまとめる
- 結びは「何かご不明点がございましたらお知らせください。」で締める"""


def build_structured_user(mail: dict[str, Any]) -> str:
    return f"""以下の受信メールに対する返信を作成してください。

手順：
1. 受信メールの依頼・確認・共有事項を整理する
2. それをすべてカバーする返信本文を作成する

[件名]
{mail.get("subject", "")}

[本文]
{mail.get("body", "")}

[署名]
{mail.get("signature", "")}
"""


# ── プロンプト（two_step） ────────────────────────────────

# STEP1: シンプルな指示で下書きを生成（baselineと同じ）
DRAFT_SYSTEM = "あなたは日本語のビジネスメール返信アシスタントです。返信本文のみを出力してください。"


def build_draft_user(mail: dict[str, Any]) -> str:
    return f"""以下の受信メールに対する返信を作成してください。

[件名]
{mail.get("subject", "")}

[本文]
{mail.get("body", "")}
"""


# STEP2: 下書きを元メールと照らし合わせて論点の漏れを補完する
REVIEW_SYSTEM = "あなたは日本語のビジネスメール担当者です。返信本文のみを出力してください。"


def build_review_user(body: str, draft: str) -> str:
    return f"""以下の受信メールへの返信下書きを改善してください。
論点の漏れがあれば補完し、なければそのまま出力してください。

【受信メール】
{body}

【下書き】
{draft}"""


# ── ユーティリティ ────────────────────────────────────────

def chat(client: OpenAI, model: str, system: str, user: str, temperature: float) -> str:
    """1回のチャットリクエストを送信してレスポンスを返します。"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("モデルが空のレスポンスを返しました。")
    return content.strip()


def strip_fences(text: str) -> str:
    """コードブロックのフェンス（```）を除去します。"""
    text = re.sub(r"^```[^\n]*\n?", "", text).strip()
    text = re.sub(r"\n?```$", "", text).strip()
    return text


# ── 生成処理 ─────────────────────────────────────────────

def generate_structured(
    client: OpenAI, model: str, temperature: float, mail: dict[str, Any]
) -> str:
    """structured_prompt: 詳細なルールを持つプロンプトで1回生成します。"""
    return strip_fences(chat(
        client, model, STRUCTURED_SYSTEM,
        build_structured_user(mail), temperature,
    ))


def generate_two_step(
    client: OpenAI,
    model_draft: str,
    model_review: str,
    temp_draft: float,
    temp_review: float,
    mail: dict[str, Any],
    include_draft: bool = False,
) -> dict[str, Any]:
    """
    two_step: 2ステップで返信を生成します。
    STEP1: シンプルなプロンプトで下書きを生成
    STEP2: 下書きを元メールと照らし合わせて論点の漏れを補完
    """
    # STEP1: 下書き生成
    draft = strip_fences(chat(
        client, model_draft, DRAFT_SYSTEM,
        build_draft_user(mail), temp_draft,
    ))

    # STEP2: 自己レビュー＆補完
    final = strip_fences(chat(
        client, model_review, REVIEW_SYSTEM,
        build_review_user(mail.get("body", ""), draft), temp_review,
    ))

    result: dict[str, Any] = {"reply_draft": final}
    if include_draft:
        result["draft"] = draft  # 中間出力（下書き）を含める場合
    return result


# ── 入出力 ───────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="返信メール生成（改善版）: structured_promptまたはtwo_stepで返信案を生成します。"
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
    parser.add_argument(
        "--mode", choices=["structured_prompt", "two_step"], required=True,
        help="生成モード。structured_prompt: 詳細なプロンプトで1回生成。two_step: 下書き→レビューの2ステップ生成。",
    )
    parser.add_argument("--model", default="qwen3:4b-instruct", help="使用するモデル名。")
    parser.add_argument("--model-draft",  help="two_stepのSTEP1（下書き）に使うモデル。省略時は--modelを使用。")
    parser.add_argument("--model-review", help="two_stepのSTEP2（レビュー）に使うモデル。省略時は--modelを使用。")
    parser.add_argument(
        "--base-url", default="http://localhost:11434/v1/",
        help="OllamaのOpenAI互換APIのベースURL。",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="サンプリング温度（structured_prompt用）。")
    parser.add_argument("--temp-draft",  type=float, default=0.2, help="STEP1（下書き）のサンプリング温度。")
    parser.add_argument("--temp-review", type=float, default=0.1, help="STEP2（レビュー）のサンプリング温度。")
    parser.add_argument(
        "--include-draft", action="store_true",
        help="two_stepモードでSTEP1の下書きを出力に含めます。",
    )
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
    model_draft  = args.model_draft  or args.model
    model_review = args.model_review or args.model
    client = OpenAI(base_url=args.base_url, api_key="ollama")
    mails = load_records(args.input, args.input_format)

    results: list[dict[str, Any]] = []
    for mail in mails:
        row_id = mail.get("id")
        try:
            if args.mode == "structured_prompt":
                reply = generate_structured(
                    client=client, model=args.model,
                    temperature=args.temperature, mail=mail,
                )
                results.append({"id": row_id, "reply_draft": reply})

            else:  # two_step
                result = generate_two_step(
                    client=client,
                    model_draft=model_draft,
                    model_review=model_review,
                    temp_draft=args.temp_draft,
                    temp_review=args.temp_review,
                    mail=mail,
                    include_draft=args.include_draft,
                )
                results.append({"id": row_id, **result})

        except Exception as e:
            results.append({"id": row_id, "error": str(e)})

    output_results(results, args.output_format)


if __name__ == "__main__":
    main()