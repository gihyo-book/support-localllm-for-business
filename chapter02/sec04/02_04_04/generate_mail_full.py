#!/usr/bin/env python3
"""
generate_mail_full.py

ダミーメール生成スクリプト（LLMに全部任せる版）。
各Stageでプロンプトの設計を変えながら、崩れ方の違いを記録します。

Stage 1: 一発生成（条件を整理してtype候補を明示）
Stage 2: 一発生成（Stage1 + 出力例の例示を追加）
Stage 3: ステップ分割（Plan→件名→本文→PII→マスキング候補を個別に生成）

Usage:
    python generate_mail_full.py --count 3 --model qwen3:4b-instruct --stage 1
    python generate_mail_full.py --count 3 --model gemma3:12b --stage 2
    python generate_mail_full.py --count 3 --model gpt-oss:20b --stage 3
"""

import argparse
import json
import random
import re
import sys
from typing import Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# Stage 1: 一発生成（条件を整理してtype候補を明示）
# ---------------------------------------------------------------------------
STAGE1_PROMPT = """日本語のビジネスメールを1件生成してください。
以下の条件をすべて守ってください。

条件:
- 実在しない架空の人物・企業・住所を使うこと
- 氏名・会社名・電話番号・住所は典型的な例（山田太郎、080-1234-5678、東京都港区赤坂など）を使わないこと
- メールアドレスのドメインは必ず example.invalid にすること
- 電話番号は日本の形式（例: 070-XXXX-XXXX）にすること
- 請求書番号は INV-YYYY-NNNN 形式にすること
- 口座番号は ACCT-XXXX-XXXX 形式にすること
- 個人情報（氏名・会社名・メールアドレス・電話番号・住所・請求書番号・口座番号）を masking_candidates としてリストアップすること
- masking_candidates の各要素は type / value / reason の3フィールドを持つこと
- type は name / company / email / phone / address / invoice_id / account_number / contract_id のいずれかにすること
- value は必ず本文または署名に実際に含まれる文字列と完全一致すること
- reason は日本語で記述すること

以下のJSON形式で出力してください。

{{
  "plan": {{
    "topic": "請求関連 / 契約関連 / 納期関連 / 発注関連 のいずれか",
    "intent": "依頼 / 確認 / 報告 のいずれか",
    "issue": "問題の種類（例: 誤記・遅延・数量相違など）",
    "action": "必要なアクション（例: 再送依頼・進捗確認など）",
    "urgency": "高 / 通常 / 低 のいずれか",
    "tone": "丁寧 / 柔らかめ / フォーマル のいずれか",
    "relationship": "新規 / 継続 のいずれか"
  }},
  "subject": "件名",
  "body": "本文",
  "signature": "署名（氏名・会社名・メールアドレス・電話番号・住所を含む）",
  "masking_candidates": [
    {{"type": "name", "value": "氏名", "reason": "理由"}},
    ...
  ]
}}

JSONのみ出力してください。説明は不要です。"""


# ---------------------------------------------------------------------------
# Stage 1（強調版）: Stage 1の各条件に【重要】【必須】【絶対遵守】を付与
# 「強調するほど守ってくれる」という直感が本当に効くかを検証するためのバリアント。
# ---------------------------------------------------------------------------
STAGE1_EMPHASIS_PROMPT = """日本語のビジネスメールを1件生成してください。
以下の条件をすべて守ってください。【厳守】

条件:
- 【重要】実在しない架空の人物・企業・住所を使うこと
- 【必須】氏名・会社名・電話番号・住所は典型的な例（山田太郎、080-1234-5678、東京都港区赤坂など）を使わないこと
- 【絶対遵守】メールアドレスのドメインは必ず example.invalid にすること
- 【必須】電話番号は日本の形式（例: 070-XXXX-XXXX）にすること
- 【重要】請求書番号は INV-YYYY-NNNN 形式にすること
- 【重要】口座番号は ACCT-XXXX-XXXX 形式にすること
- 【必須】個人情報（氏名・会社名・メールアドレス・電話番号・住所・請求書番号・口座番号）を masking_candidates としてリストアップすること
- 【必須】masking_candidates の各要素は type / value / reason の3フィールドを持つこと
- 【絶対遵守】type は name / company / email / phone / address / invoice_id / account_number / contract_id のいずれかにすること
- 【絶対遵守】value は必ず本文または署名に実際に含まれる文字列と完全一致すること
- 【必須】reason は日本語で記述すること

★★★ 上記の条件はすべて重要です。1つでも違反した出力は無効です。 ★★★

以下のJSON形式で出力してください。

{{
  "plan": {{
    "topic": "請求関連 / 契約関連 / 納期関連 / 発注関連 のいずれか",
    "intent": "依頼 / 確認 / 報告 のいずれか",
    "issue": "問題の種類（例: 誤記・遅延・数量相違など）",
    "action": "必要なアクション（例: 再送依頼・進捗確認など）",
    "urgency": "高 / 通常 / 低 のいずれか",
    "tone": "丁寧 / 柔らかめ / フォーマル のいずれか",
    "relationship": "新規 / 継続 のいずれか"
  }},
  "subject": "件名",
  "body": "本文",
  "signature": "署名（氏名・会社名・メールアドレス・電話番号・住所を含む）",
  "masking_candidates": [
    {{"type": "name", "value": "氏名", "reason": "理由"}},
    ...
  ]
}}

【重要】JSONのみ出力してください。説明は不要です。"""


def generate_stage1(client: OpenAI, model: str, temperature: float, emphasis: bool = False) -> dict[str, Any]:
    prompt = STAGE1_EMPHASIS_PROMPT if emphasis else STAGE1_PROMPT
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Model returned empty content.")
    text = content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in output:\n{text}")
    return json.loads(text[start:end + 1])


# ---------------------------------------------------------------------------
# Stage 2: 一発生成（Stage1 + 出力例の例示を追加）
# ---------------------------------------------------------------------------
STAGE2_EXAMPLE = """{
  "plan": {
    "topic": "請求関連",
    "intent": "依頼",
    "issue": "誤記",
    "action": "再送依頼",
    "urgency": "高",
    "tone": "柔らかめ",
    "relationship": "継続"
  },
  "subject": "請求書の誤記についてご確認ください",
  "body": "いつもお世話になっております。先日お送りした請求書番号 INV-2026-4521 に誤記がございました。お振込先口座（ACCT-1234-5678）についても誤りがございましたので、修正版を再送いたします。ご確認いただけますと幸いです。",
  "signature": "株式会社テスト商事\\n営業部\\n鈴木 一郎\\nizumi.ichiro@example.invalid\\n090-0000-1234\\n東京都千代田区架空町1-2-3",
  "masking_candidates": [
    {"type": "name",           "value": "鈴木 一郎",                    "reason": "個人を特定できる氏名"},
    {"type": "company",        "value": "株式会社テスト商事",            "reason": "取引先を特定できる法人名"},
    {"type": "email",          "value": "izumi.ichiro@example.invalid", "reason": "連絡先メールアドレス"},
    {"type": "phone",          "value": "090-0000-1234",                "reason": "連絡先電話番号"},
    {"type": "address",        "value": "東京都千代田区架空町1-2-3",     "reason": "所在地住所"},
    {"type": "invoice_id",     "value": "INV-2026-4521",                "reason": "請求識別子"},
    {"type": "account_number", "value": "ACCT-1234-5678",               "reason": "口座識別子"}
  ]
}"""

STAGE2_PROMPT = f"""日本語のビジネスメールを1件生成してください。
以下の条件をすべて守ってください。

条件:
- 実在しない架空の人物・企業・住所を使うこと
- 氏名・会社名・電話番号・住所は典型的な例（山田太郎、080-1234-5678、東京都港区赤坂など）を使わないこと
- メールアドレスのドメインは必ず example.invalid にすること
- 電話番号は日本の形式（例: 070-XXXX-XXXX）にすること
- 請求書番号は INV-YYYY-NNNN 形式にすること
- 口座番号は ACCT-XXXX-XXXX 形式にすること
- 個人情報（氏名・会社名・メールアドレス・電話番号・住所・請求書番号・口座番号）を masking_candidates としてリストアップすること
- masking_candidates の各要素は type / value / reason の3フィールドを持つこと
- type は name / company / email / phone / address / invoice_id / account_number / contract_id のいずれかにすること
- value は必ず本文または署名に実際に含まれる文字列と完全一致すること
- reason は日本語で記述すること
- plan.topic は 請求関連 / 契約関連 / 納期関連 / 発注関連 のいずれかにすること
- plan.intent は 依頼 / 確認 / 報告 のいずれかにすること
- plan.urgency は 高 / 通常 / 低 のいずれかにすること

出力例:
{STAGE2_EXAMPLE}

上記の例と同じJSON形式で、内容が異なるメールを1件生成してください。
JSONのみ出力してください。説明は不要です。"""


def generate_stage2(client: OpenAI, model: str, temperature: float) -> dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": STAGE2_PROMPT}],
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Model returned empty content.")
    text = content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in output:\n{text}")
    return json.loads(text[start:end + 1])


# ---------------------------------------------------------------------------
# Stage 3: ステップ分割（Plan→PII→本文→件名→マスキング候補を個別に生成）
#
# 各ステップをシンプルな指示に絞ることで品質が上がるか検証するパターン。
# PII（STEP2）を先に生成し、本文生成（STEP3）にPIIを渡すことで
# 本文とmasking_candidatesの値が一致するよう設計している。
# ただし「渡されたPIIを本文に正しく組み込む」という新しい難しさが生じる。
# ---------------------------------------------------------------------------

# hybridと同じラベル定義
STAGE3_TOPICS    = ["請求関連", "納期関連", "契約関連", "発注関連"]
STAGE3_INTENTS   = ["確認", "依頼", "報告"]
STAGE3_URGENCIES = ["低", "通常", "高"]


def stage3_step1_plan(client: OpenAI, model: str, temperature: float) -> dict[str, Any]:
    """STEP1: Planを生成する。topic/intent/urgencyはコードでランダム選択してモデルに渡す。"""
    topic   = random.choice(STAGE3_TOPICS)
    intent  = random.choice(STAGE3_INTENTS)
    urgency = random.choice(STAGE3_URGENCIES)

    system = "あなたはビジネスメールの設計アシスタントです。JSONのみ出力してください。"
    user = f"""以下の属性を持つビジネスメールのPlanを生成してください。

topic: {topic}
intent: {intent}
urgency: {urgency}

以下のJSON形式で出力してください。
{{
  "topic": "{topic}",
  "intent": "{intent}",
  "urgency": "{urgency}",
  "issue": "具体的な問題・課題（1フレーズ）",
  "action": "相手に求めるアクション（1フレーズ）",
  "tone": "丁寧 / 柔らかめ / フォーマル のいずれか",
  "relationship": "新規 / 継続 のいずれか"
}}"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    text = content.strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"STEP1: No JSON found:\n{text}")
    return json.loads(text[s:e + 1])


def stage3_step2_pii(client: OpenAI, model: str, temperature: float) -> dict[str, str]:
    """STEP2: PIIを先に生成する。後続の本文生成でこのPIIを使う。"""
    system = "あなたは架空の個人情報を生成するアシスタントです。JSONのみ出力してください。"
    user = """架空のビジネスパーソンの個人情報を生成してください。

条件:
- 実在しない架空の人物・企業・住所を使うこと
- メールアドレスのドメインは必ず example.invalid にすること
- 電話番号は日本の形式（例: 070-XXXX-XXXX）にすること
- 請求書番号は INV-2026-NNNN 形式にすること
- 口座番号は ACCT-XXXX-XXXX 形式にすること

{
  "name": "氏名",
  "company": "会社名",
  "department": "部署名",
  "email": "メールアドレス",
  "phone": "電話番号",
  "address": "住所",
  "invoice_id": "請求書番号",
  "account_number": "口座番号"
}"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    text = content.strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"STEP2: No JSON found:\n{text}")
    return json.loads(text[s:e + 1])


def stage3_step3_body(
    client: OpenAI, model: str, plan: dict, pii: dict, temperature: float
) -> str:
    """STEP3: 本文を生成する。PlanとPIIを渡して本文にPIIを組み込ませる。
    LLMが渡されたPIIを正しく使うかどうかが課題になる。"""
    user = f"""以下のPlanと個人情報に従い、ビジネスメール本文を生成してください。
挨拶・本題・締め文を含む自然な日本語のメール本文のみ出力してください。
本文中に以下の個人情報（請求書番号・口座番号など）を自然に組み込んでください。

Plan: {json.dumps(plan, ensure_ascii=False)}

使用する個人情報:
- 送信者名: {pii.get('name', '')}
- 会社名: {pii.get('company', '')}
- 請求書番号: {pii.get('invoice_id', '')}
- 口座番号: {pii.get('account_number', '')}"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def stage3_step4_subject(
    client: OpenAI, model: str, plan: dict, body: str, temperature: float
) -> str:
    """STEP4: 件名を生成する。Planと本文を渡して内容に合った件名を作る。"""
    user = f"""以下のPlanと本文に対応するビジネスメールの件名を1行だけ出力してください。
件名のテキストのみ出力。説明不要。

Plan: {json.dumps(plan, ensure_ascii=False)}
本文（先頭100文字）: {body[:100]}"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip().strip('"').strip("「」")


def stage3_step5_masking(
    client: OpenAI, model: str, body: str, signature: str, temperature: float
) -> list[dict]:
    """STEP5: masking_candidatesを生成する。本文・署名からPIIを抽出させる。"""
    system = "あなたは個人情報マスキングの専門家です。JSONのみ出力してください。"
    user = f"""以下のメール本文と署名から、マスキングすべき個人情報をすべて抽出してください。

本文:
{body}

署名:
{signature}

以下のJSON形式で出力してください。
{{
  "masking_candidates": [
    {{"type": "name/company/email/phone/address/invoice_id/account_number/contract_id のいずれか", "value": "実際の文字列", "reason": "理由（日本語）"}}
  ]
}}"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    text = content.strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"STEP5: No JSON found:\n{text}")
    parsed = json.loads(text[s:e + 1])
    return parsed.get("masking_candidates", [])


def generate_stage3(client: OpenAI, model: str, temperature: float) -> dict[str, Any]:
    # STEP1: Plan生成（topic/intent/urgencyはコードでランダム選択）
    plan = stage3_step1_plan(client, model, temperature)
    print(f"    STEP1 plan: {plan.get('topic')} / {plan.get('intent')} / {plan.get('urgency')}", file=sys.stderr)

    # STEP2: PII生成（本文生成より先に確定させる）
    pii = stage3_step2_pii(client, model, temperature)
    print(f"    STEP2 pii: name={pii.get('name')} email={pii.get('email')}", file=sys.stderr)

    # 署名をPIIから組み立て
    signature = (
        f"{pii.get('company', '')}\n"
        f"{pii.get('department', '')}\n"
        f"{pii.get('name', '')}\n"
        f"{pii.get('email', '')}\n"
        f"{pii.get('phone', '')}\n"
        f"{pii.get('address', '')}"
    )

    # STEP3: 本文生成（PlanとPIIを渡して本文にPIIを組み込ませる）
    body = stage3_step3_body(client, model, plan, pii, temperature)
    print(f"    STEP3 body: {body[:40]}...", file=sys.stderr)

    # STEP4: 件名生成（Planと本文を渡す）
    subject = stage3_step4_subject(client, model, plan, body, temperature)
    print(f"    STEP4 subject: {subject}", file=sys.stderr)

    # STEP5: masking_candidates生成（本文・署名からPIIを抽出）
    candidates = stage3_step5_masking(client, model, body, signature, temperature)
    print(f"    STEP5 candidates: {len(candidates)} items", file=sys.stderr)

    return {
        "plan":               plan,
        "subject":            subject,
        "body":               body,
        "signature":          signature,
        "masking_candidates": candidates,
        "_pii":               pii,  # デバッグ用：生成したPIIを記録しておく
    }


# ---------------------------------------------------------------------------
# 出力検証（リトライはしない・崩れの記録が目的）
# ---------------------------------------------------------------------------

def check_output(record: dict[str, Any]) -> list[str]:
    """出力の問題点をリストアップする。"""
    issues = []
    body = record.get("body", "")
    sig  = record.get("signature", "")
    full_text = body + sig

    if not record.get("subject", "").strip():
        issues.append("subject is empty")
    if not body.strip():
        issues.append("body is empty")
    if not sig.strip():
        issues.append("signature is empty")

    candidates = record.get("masking_candidates")
    if not candidates:
        issues.append("masking_candidates missing")
    else:
        valid_types = {"name", "company", "email", "phone", "address", "invoice_id", "account_number", "contract_id"}
        for c in candidates:
            t = c.get("type", "")
            v = c.get("value", "")
            if t not in valid_types:
                issues.append(f"invalid type: {t}")
            if v and v not in full_text:
                issues.append(f"value not in text: '{v}'")
            for field in ("type", "value", "reason"):
                if not c.get(field):
                    issues.append(f"missing field: {field}")

    emails = re.findall(r'[\w.+-]+@[\w.-]+', full_text)
    for email in emails:
        if not email.endswith("example.invalid"):
            issues.append(f"invalid email domain: {email}")

    return issues


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ダミーメール生成（LLMに全部任せる版）。各Stageで崩れ方の違いを記録します。"
    )
    parser.add_argument("--count",       "-n", type=int, default=3,                 help="生成件数")
    parser.add_argument("--model",       "-m", default="qwen3:4b-instruct",         help="モデル名")
    parser.add_argument("--base-url",          default="http://localhost:11434/v1/", help="OllamaのベースURL")
    parser.add_argument("--output",      "-o", default=None,                        help="出力先ファイル（省略時はstdout）")
    parser.add_argument("--temperature",       type=float, default=0.7,             help="サンプリング温度")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="生成Stage（1: 条件整理版, 2: 例示追加版, 3: ステップ分割版）")
    parser.add_argument("--emphasis", action="store_true",
                        help="Stage 1のみ有効。各条件行に【重要】【必須】等の強調を付与した版を使う")
    parser.add_argument("--id-prefix",         default="mail",                      help="IDのプレフィックス")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key="ollama")

    records: list[dict[str, Any]] = []
    for i in range(1, args.count + 1):
        mail_id = f"{args.id_prefix}_{i}"
        emphasis_label = " [emphasis]" if (args.emphasis and args.stage == 1) else ""
        print(f"[{i}/{args.count}] Generating {mail_id} (stage={args.stage}{emphasis_label}) ...", file=sys.stderr)
        try:
            if args.stage == 1:
                result = generate_stage1(client, args.model, args.temperature, emphasis=args.emphasis)
            elif args.stage == 2:
                if args.emphasis:
                    print("  [WARN] --emphasis is only applied to stage 1; ignored for stage 2.", file=sys.stderr)
                result = generate_stage2(client, args.model, args.temperature)
            elif args.stage == 3:
                if args.emphasis:
                    print("  [WARN] --emphasis is only applied to stage 1; ignored for stage 3.", file=sys.stderr)
                result = generate_stage3(client, args.model, args.temperature)

            result["id"]    = mail_id
            result["model"] = args.model
            result["stage"] = args.stage
            if args.emphasis and args.stage == 1:
                result["variant"] = "emphasis"

            issues = check_output(result)
            if issues:
                print(f"  [ISSUES] {issues}", file=sys.stderr)
            else:
                print(f"  [OK]", file=sys.stderr)

            result["_issues"] = issues
            records.append(result)

        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)

    lines = []
    for r in records:
        r.pop("_issues", None)
        r.pop("_pii", None)  # デバッグ用フィールドは出力から除く
        lines.append(json.dumps(r, ensure_ascii=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote {len(records)} records to {args.output}", file=sys.stderr)
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()