#!/usr/bin/env python3
import argparse
import json
from typing import Any

from openai import OpenAI

JUDGE_SYSTEM_PROMPT = """あなたは業務メールの返信案を評価する専門家です。
以下の評価軸に従って、返信案を1〜5点で採点し、理由を述べてください。

## 評価軸

### 1. 論点網羅性（1〜5点）
元メールで依頼・確認・共有されている論点をすべて拾えているか。
- 5: すべての論点に対応している
- 3: 主要な論点は対応しているが一部漏れがある
- 1: 重要な論点が複数抜けている

### 2. 指示遵守（1〜5点）
与えられた返信条件（調査中・影響不明など）を正確に守っているか。
- 5: すべての条件を正確に守っている
- 3: おおむね守っているが一部解釈が違う
- 1: 条件に反する内容が含まれている

### 3. ビジネスメールとしての自然さ（1〜5点）
実際の業務で送れるクオリティか。
- 5: そのまま送れる
- 3: 少し修正すれば送れる
- 1: 大幅な修正が必要

## 出力形式（JSON）
{
  "coverage": {"score": 1〜5, "reason": "理由"},
  "compliance": {"score": 1〜5, "reason": "理由"},
  "naturalness": {"score": 1〜5, "reason": "理由"},
  "total": 合計点（3〜15）,
  "summary": "全体的なコメント"
}
JSONのみを出力してください。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge email reply drafts with Ollama LLM-as-a-judge."
    )
    parser.add_argument("--mails", required=True, help="Path to mail JSON/JSONL input.")
    parser.add_argument(
        "--replies", required=True, help="Path to reply JSON/JSONL generated output."
    )
    parser.add_argument(
        "--mails-format",
        choices=["auto", "json", "jsonl"],
        default="auto",
        help="Mail input format. auto detects by extension.",
    )
    parser.add_argument(
        "--replies-format",
        choices=["auto", "json", "jsonl"],
        default="auto",
        help="Reply input format. auto detects by extension.",
    )
    parser.add_argument(
        "--reply-field",
        default="reply_draft",
        help='Reply text field in replies file (default: "reply_draft").',
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl"],
        default="json",
        help="Output format for judge results.",
    )
    parser.add_argument(
        "--include-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include aggregated average summary in output (default: true).",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="qwen3:4b",
        help='Model name on Ollama (e.g. "qwen3:4b").',
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1/",
        help='Ollama OpenAI-compatible base URL (default: "http://localhost:11434/v1/").',
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for judge model.",
    )
    return parser.parse_args()


def detect_input_format(path: str, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    if path.lower().endswith(".jsonl"):
        return "jsonl"
    return "json"


def load_records(path: str, input_format: str) -> list[dict[str, Any]]:
    fmt = detect_input_format(path, input_format)
    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        rows = loaded if isinstance(loaded, list) else [loaded]

    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid record in {path}: all items must be objects.")
    return rows


def to_index(records: list[dict[str, Any]], source_name: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in records:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{source_name} contains record with missing/invalid id.")
        index[row_id] = row
    return index


def build_original_mail_text(mail: dict[str, Any]) -> str:
    payload = {
        "id": mail.get("id"),
        "subject": mail.get("subject", ""),
        "body": mail.get("body", ""),
        "signature": mail.get("signature", ""),
        "plan": mail.get("plan", {}),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_judge_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Judge output does not contain valid JSON.")
        return json.loads(content[start : end + 1])


def judge_one(
    client: OpenAI,
    model: str,
    temperature: float,
    original_mail_text: str,
    reply_text: str,
) -> dict[str, Any]:
    user_prompt = f"""## 元メール
{original_mail_text}

## 評価対象の返信案
{reply_text}

上記の返信案を評価してください。"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("Judge model returned empty content.")
    return parse_judge_json(content)


def get_score(judge: dict[str, Any], axis: str) -> float:
    axis_obj = judge.get(axis, {})
    if isinstance(axis_obj, dict):
        score = axis_obj.get("score")
        if isinstance(score, (int, float)):
            return float(score)
    return 0.0


def get_total(judge: dict[str, Any]) -> float:
    total = judge.get("total")
    if isinstance(total, (int, float)):
        return float(total)
    return 0.0


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "count": 0,
            "coverage_avg": 0.0,
            "compliance_avg": 0.0,
            "naturalness_avg": 0.0,
            "total_avg": 0.0,
            "total_min": 0.0,
            "total_max": 0.0,
        }

    coverage_scores = [get_score(row["judge"], "coverage") for row in results]
    compliance_scores = [get_score(row["judge"], "compliance") for row in results]
    naturalness_scores = [get_score(row["judge"], "naturalness") for row in results]
    totals = [get_total(row["judge"]) for row in results]

    def r2(v: float) -> float:
        return round(v, 2)

    return {
        "count": n,
        "coverage_avg": r2(sum(coverage_scores) / n),
        "compliance_avg": r2(sum(compliance_scores) / n),
        "naturalness_avg": r2(sum(naturalness_scores) / n),
        "total_avg": r2(sum(totals) / n),
        "total_min": r2(min(totals)),
        "total_max": r2(max(totals)),
    }


def main() -> None:
    args = parse_args()
    mails = load_records(args.mails, args.mails_format)
    replies = load_records(args.replies, args.replies_format)

    mail_index = to_index(mails, "mails")
    reply_index = to_index(replies, "replies")
    common_ids = sorted(set(mail_index.keys()) & set(reply_index.keys()))
    if not common_ids:
        raise ValueError("No matching ids found between mails and replies.")

    client = OpenAI(base_url=args.base_url, api_key="ollama")
    results = []
    for row_id in common_ids:
        mail = mail_index[row_id]
        reply_row = reply_index[row_id]
        reply_text = reply_row.get(args.reply_field, "")
        if not isinstance(reply_text, str) or not reply_text.strip():
            raise ValueError(
                f'Reply record "{row_id}" does not contain non-empty "{args.reply_field}".'
            )
        judge = judge_one(
            client=client,
            model=args.model,
            temperature=args.temperature,
            original_mail_text=build_original_mail_text(mail),
            reply_text=reply_text,
        )
        results.append({"id": row_id, "judge": judge})

    summary = build_summary(results)

    if args.output_format == "jsonl":
        for row in results:
            print(json.dumps(row, ensure_ascii=False))
        if args.include_summary:
            print(json.dumps({"type": "summary", "summary": summary}, ensure_ascii=False))
        return
    if args.include_summary:
        print(
            json.dumps(
                {"results": results, "summary": summary},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
