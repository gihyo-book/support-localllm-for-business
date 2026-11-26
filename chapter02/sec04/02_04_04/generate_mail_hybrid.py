#!/usr/bin/env python3
"""
generate_mail_hybrid_v2.py

ハイブリッド版ダミーメール生成スクリプト（LLM負荷軽減版）。

v1 からの変更点:
  1. SYSTEM_PROMPT を簡素化（禁止事項列挙 → 「書くこと」だけ指定）
  2. PIIプレースホルダをLLMに渡さない（生成後にコードで文中挿入）
  3. 件名生成をテンプレートに切り替え（LLM呼び出し廃止）

v2.1 からの変更点:
  4. 「問い合わせ」カテゴリを削除（ラベル境界が曖昧なため）

Usage:
    python generate_mail_hybrid_v2.py --count 5 --model gemma3:4b
    python generate_mail_hybrid_v2.py --count 10 --model gpt-oss:20b --output mails.jsonl
"""

import argparse
import json
import random
import re
import sys
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Plan プリセット
# ---------------------------------------------------------------------------
PLAN_PRESETS: list[dict[str, Any]] = [
    {"topic": "請求関連", "intent": "確認", "issue": "金額相違",   "action": "確認依頼",     "urgency": "高",   "tone": "かため",   "relationship": "継続"},
    {"topic": "請求関連", "intent": "依頼", "issue": "誤記",       "action": "再送依頼",     "urgency": "高",   "tone": "柔らかめ", "relationship": "継続"},
    {"topic": "請求関連", "intent": "依頼", "issue": "入金遅延",   "action": "入金確認依頼", "urgency": "高",   "tone": "かため",   "relationship": "継続"},
    {"topic": "請求関連", "intent": "確認", "issue": "振込先相違", "action": "口座確認依頼", "urgency": "高",   "tone": "普通",     "relationship": "継続"},
    {"topic": "納期関連", "intent": "提案", "issue": "遅延リスク", "action": "日程調整",     "urgency": "高",   "tone": "普通",     "relationship": "継続"},
    {"topic": "納期関連", "intent": "確認", "issue": "未返答",     "action": "確認依頼",     "urgency": "通常", "tone": "柔らかめ", "relationship": "新規"},
    {"topic": "契約関連", "intent": "依頼", "issue": "条件変更",   "action": "承認依頼",     "urgency": "通常", "tone": "かため",   "relationship": "継続"},
    {"topic": "契約関連", "intent": "報告", "issue": "期日超過",   "action": "対応確認",     "urgency": "高",   "tone": "かため",   "relationship": "長期"},
    {"topic": "発注関連", "intent": "確認", "issue": "数量不一致", "action": "確認依頼",     "urgency": "通常", "tone": "普通",     "relationship": "継続"},
    {"topic": "発注関連", "intent": "依頼", "issue": "仕様変更",   "action": "変更依頼",     "urgency": "通常", "tone": "柔らかめ", "relationship": "新規"},
]

CONTRACT_CHANGE_DETAILS = [
    "支払期限を30日から45日へ延長",
    "自動更新条項の削除",
    "サービス提供範囲の一部縮小",
    "月額料金の改定（現行比10%増）",
    "納品物の仕様定義条項の追記",
    "違約金条項の上限額引き下げ",
]
CONTRACT_OVERDUE_REASONS = [
    "担当者の異動による引き継ぎ遅延",
    "社内決裁フローの変更に伴う遅れ",
    "関連部署との調整に想定以上の時間を要したため",
    "必要書類の収集に遅れが生じたため",
    "システム障害による処理停止",
]
ORDER_SPEC_CHANGES = [
    "材質をアルミから樹脂に変更",
    "カラーをホワイトからブラックに変更",
    "サイズをA4からA3に変更",
    "パッケージを個別包装から一括包装に変更",
    "表面加工をマットからグロスに変更",
    "数量を50個から30個に減少",
]
ORDER_QUANTITY_GAPS = [
    {"ordered": 20, "delivered": 18, "unit": "個"},
    {"ordered": 50, "delivered": 45, "unit": "個"},
    {"ordered": 10, "delivered": 12, "unit": "箱"},
    {"ordered": 100, "delivered": 95, "unit": "枚"},
    {"ordered": 30, "delivered": 28, "unit": "セット"},
]


def enrich_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """topicに応じてplanにダミー詳細フィールドを追加する。"""
    p = {**plan}
    topic, issue = p.get("topic", ""), p.get("issue", "")
    if topic == "契約関連":
        if issue == "条件変更":
            p["change_detail"] = random.choice(CONTRACT_CHANGE_DETAILS)
        elif issue == "期日超過":
            p["overdue_reason"] = random.choice(CONTRACT_OVERDUE_REASONS)
    elif topic == "発注関連":
        if issue == "仕様変更":
            p["spec_change"] = random.choice(ORDER_SPEC_CHANGES)
        elif issue == "数量不一致":
            gap = random.choice(ORDER_QUANTITY_GAPS)
            p["ordered_qty"]   = gap["ordered"]
            p["delivered_qty"] = gap["delivered"]
            p["qty_unit"]      = gap["unit"]
    return p

# topicごとに使うPIIキーと、本文への挿入フレーズテンプレート
TOPIC_PII_CONFIG: dict[str, dict[str, Any]] = {
    "請求関連": {
        "keys": ["invoice_id", "account_number"],
        "inserts": {
            "invoice_id":     ["請求書番号{id}の件につきまして、", "ご請求書（{id}）に関しまして、"],
            "account_number": ["お振込先口座（{id}）について、", "口座番号{id}宛のお振込みについて、"],
        },
    },
    "納期関連": {
        "keys": ["order_id"],
        "inserts": {
            "order_id": ["注文番号{id}の件につきまして、", "ご注文（{id}）の納期について、"],
        },
    },
    "契約関連": {
        "keys": ["contract_id"],
        "inserts": {
            "contract_id": ["契約番号{id}に関しまして、", "ご契約（{id}）の件につきまして、"],
        },
    },
    "発注関連": {
        "keys": ["order_id"],
        "inserts": {
            "order_id": ["注文番号{id}の件につきまして、", "発注番号{id}に関しまして、"],
        },
    },
}

# ---------------------------------------------------------------------------
# 件名テンプレート（LLM不要）
# ---------------------------------------------------------------------------
SUBJECT_TEMPLATES: dict[tuple[str, str], list[str]] = {
    ("請求関連", "確認"): ["【ご確認】請求金額の相違について", "請求金額の確認依頼"],
    ("請求関連", "依頼"): ["請求書誤記に関する再送のお願い", "【お願い】入金確認のご依頼", "【至急】入金状況のご確認"],
    ("納期関連", "提案"): ["納期遅延リスクへの日程調整のご提案", "【重要】納期調整についてのご相談"],
    ("納期関連", "確認"): ["納期確認のお願い", "【ご確認】納期についてのご回答のお願い"],
    ("契約関連", "依頼"): ["契約条件変更の承認依頼", "【お願い】契約条件変更についてご確認ください"],
    ("契約関連", "報告"): ["【重要】契約期日超過に関するご報告", "契約履行遅延に関する対応確認のお願い"],
    ("発注関連", "確認"): ["発注数量の相違についてご確認のお願い", "【確認】発注内容の数量不一致について"],
    ("発注関連", "依頼"): ["発注仕様変更のご依頼", "【お願い】仕様変更についてご検討ください"],
}

def generate_subject(plan: dict[str, Any]) -> str:
    key = (plan["topic"], plan["intent"])
    candidates = SUBJECT_TEMPLATES.get(key, [f"{plan['topic']}に関するご連絡"])
    return random.choice(candidates)

# ---------------------------------------------------------------------------
# PII プール
# ---------------------------------------------------------------------------
LAST_NAMES  = ["佐藤", "鈴木", "田中", "渡辺", "伊藤", "山本", "中村", "小林", "加藤", "吉田"]
FIRST_NAMES = ["太一", "花子", "誠", "裕子", "健太", "美咲", "拓也", "由美", "翔", "奈々"]
COMPANIES   = [
    "合同会社ノースブリッジ", "株式会社東西商事", "有限会社南北物産",
    "株式会社アオヤマ製作所", "合同会社ミドリ総研", "株式会社フジワラ工業",
]
DEPARTMENTS = ["調達管理課", "営業部", "経理課", "購買部", "総務課", "業務推進部"]
EMAIL_SUFFIXES = ["204", "101", "305", "077", "412", "088", "203", "501", "99", "310"]
ADDRESS_PREFS = [
    "大阪府大阪市仮想区", "東京都千代田区架空町", "愛知県名古屋市仮想区",
    "福岡県福岡市架空区", "神奈川県横浜市仮想区", "北海道札幌市架空区",
]
ADDRESS_NUMS = ["1-2-3", "2-3-8", "3-5-11", "4-1-7", "5-8-2", "6-2-14"]
PHONE_NUMBERS = [
    "080-0000-1234", "080-0000-5678", "090-0000-2345", "090-0000-6789",
    "070-0000-3456", "070-0000-7890", "080-0000-4321", "090-0000-8765",
    "070-0000-1111", "080-0000-2222",
]
LAST_NAME_ROMAN  = {"佐藤": "sato",  "鈴木": "suzuki", "田中": "tanaka",  "渡辺": "watanabe",
                    "伊藤": "ito",   "山本": "yamamoto","中村": "nakamura","小林": "kobayashi",
                    "加藤": "kato",  "吉田": "yoshida"}
FIRST_NAME_ROMAN = {"太一": "taichi","花子": "hanako", "誠":   "makoto",  "裕子": "yuko",
                    "健太": "kenta", "美咲": "misaki",  "拓也": "takuya",  "由美": "yumi",
                    "翔":   "sho",   "奈々": "nana"}

def _format_name(last: str, first: str) -> tuple[str, list[str]]:
    roman_last  = LAST_NAME_ROMAN.get(last, last.lower())
    roman_first = FIRST_NAME_ROMAN.get(first, first.lower())
    style = random.choice(["ja", "roman_upper", "mixed"])
    if style == "ja":
        display = f"{last} {first}"
    elif style == "roman_upper":
        display = f"{roman_first.capitalize()} {roman_last.capitalize()}"
    else:
        ja = f"{last} {first}"
        roman = f"{roman_first.capitalize()} {roman_last.capitalize()}"
        display = f"{ja} / {roman}"
    return display, [display]

def _inv_id() -> str:  return f"INV-{2026}-{random.randint(1000,9999)}"
def _acct_id() -> str: return f"ACCT-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
def _ord_id() -> str:  return f"ORD-{2026}-{random.randint(1000,9999)}"
def _ctr_id() -> str:  return f"CTR-{2026}-{random.randint(1000,9999)}"

def build_pii() -> dict[str, Any]:
    last, first = random.choice(LAST_NAMES), random.choice(FIRST_NAMES)
    name_display, name_targets = _format_name(last, first)
    roman_last  = LAST_NAME_ROMAN.get(last,  last.lower())
    roman_first = FIRST_NAME_ROMAN.get(first, first.lower())
    suffix = random.choice(EMAIL_SUFFIXES)
    style  = random.choice(["last_num", "first_last", "last_first_initial"])
    if style == "last_num":         user = f"{roman_last}{suffix}"
    elif style == "first_last":     user = f"{roman_first}.{roman_last}"
    else:                           user = f"{roman_last}.{roman_first[0]}"
    return {
        "name":           name_display,
        "name_targets":   name_targets,
        "company":        random.choice(COMPANIES),
        "department":     random.choice(DEPARTMENTS),
        "email":          f"{user}.test@example.invalid",
        "phone":          random.choice(PHONE_NUMBERS),
        "address":        f"{random.choice(ADDRESS_PREFS)}{random.choice(ADDRESS_NUMS)}",
        "invoice_id":     _inv_id(),
        "account_number": _acct_id(),
        "order_id":       _ord_id(),
        "contract_id":    _ctr_id(),
    }

# ---------------------------------------------------------------------------
# 挨拶・締めテンプレート
# ---------------------------------------------------------------------------
GREETINGS = [
    "いつも大変お世話になっております。",
    "平素より格別のご高配を賜り、誠にありがとうございます。",
    "お世話になっております。",
    "いつもお世話になっております。",
]
CLOSINGS = [
    "ご確認のほど、よろしくお願いいたします。",
    "お手数をおかけしますが、どうぞよろしくお願いいたします。",
    "ご対応いただけますと幸いです。よろしくお願い申し上げます。",
    "引き続きよろしくお願いいたします。",
]

# ---------------------------------------------------------------------------
# LLM呼び出し（本題のみ・PII情報なし）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """ビジネスメール本文の状況説明と依頼・確認の部分を書いてください。

条件:
- 状況の説明と依頼・確認の内容のみ書く
- 4〜6文
- 日本語・丁寧語
- テキストのみ出力する
- 具体的な日付・期日・日時は一切書かない
- 「お世話になっております」などの挨拶文は書かない（挨拶はシステムが付加する）
- 契約・承諾・承認を想起させる表現は避け、依頼・確認の表現にとどめる"""

def generate_body_core(
    client: OpenAI,
    model: str,
    plan: dict[str, Any],
    temperature: float,
) -> str:
    """本題のみLLMで生成。PIIは一切渡さない。"""
    user = f"""以下のPlanに従い、ビジネスメール本文の状況説明と依頼・確認部分を書いてください。
挨拶・締め・署名は不要です。

Plan:
{json.dumps(plan, ensure_ascii=False)}"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ],
        extra_body={"options": {"num_ctx": 2048}},
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Model returned empty body.")
    core = content.strip()
    core = re.sub(r'[ \t]*\n[ \t]*', '', core)
    return core


def inject_pii_into_core(core: str, plan: dict[str, Any], pii: dict[str, Any]) -> str:
    topic = plan.get("topic", "")
    config = TOPIC_PII_CONFIG.get(topic, {"keys": [], "inserts": {}})

    prefixes = []
    for key in config["keys"]:
        templates = config["inserts"].get(key, [])
        if not templates:
            continue
        tmpl = random.choice(templates)
        prefixes.append(tmpl.replace("{id}", pii[key]))

    if not prefixes:
        return core

    prefix_str = "".join(prefixes)
    core_cleaned = re.sub(r'^(拝啓|敬具|謹啓)[、。\s]*', '', core).strip()
    return prefix_str + core_cleaned


def assemble_body(greeting: str, core: str, closing: str) -> str:
    return f"{greeting}{core}{closing}"


def build_signature(pii: dict[str, Any]) -> str:
    return (
        f"{pii['company']}\n{pii['department']}\n{pii['name']}\n"
        f"{pii['email']}\n{pii['phone']}\n{pii['address']}"
    )


def build_masking_candidates(pii: dict[str, Any], topic: str) -> list[dict[str, str]]:
    candidates = [
        *[{"type": "name",    "value": v, "reason": "個人を特定できる氏名"} for v in pii["name_targets"]],
        {"type": "company",   "value": pii["company"],  "reason": "取引先を特定できる法人名"},
        {"type": "email",     "value": pii["email"],    "reason": "連絡先メールアドレス"},
        {"type": "phone",     "value": pii["phone"],    "reason": "連絡先電話番号"},
        {"type": "address",   "value": pii["address"],  "reason": "所在地住所"},
    ]
    key_meta = {
        "invoice_id":     ("invoice_id",     "請求識別子"),
        "account_number": ("account_number", "口座識別子"),
        "order_id":       ("order_id",       "注文識別子"),
        "contract_id":    ("contract_id",    "契約識別子"),
    }
    for key in TOPIC_PII_CONFIG.get(topic, {}).get("keys", []):
        type_, reason = key_meta[key]
        candidates.append({"type": type_, "value": pii[key], "reason": reason})
    return candidates

# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------
STRAY_PLACEHOLDER = re.compile(
    r'\{[A-Z_]+\}'
    r'|\[[^\]]{1,20}\]'
    r'|[〇○]{2,}'
    r'|[〇○](?:（[^）]*）|「[^」]*」)'
    r'|(?<![A-Za-z])[Xx]{2,}(?![A-Za-z0-9])'
)

def has_unfilled_placeholder(text: str) -> bool:
    return bool(STRAY_PLACEHOLDER.search(text))


def validate(record: dict[str, Any], pii: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    body = record.get("body", "")
    sig  = record.get("signature", "")
    full_text = body + sig
    topic = record.get("plan", {}).get("topic", "")

    for key in TOPIC_PII_CONFIG.get(topic, {}).get("keys", []):
        if pii[key] not in full_text:
            errors.append(f"PII missing: {key}={pii[key]}")
    for v in pii["name_targets"]:
        if v not in full_text:
            errors.append(f"signature PII missing: name={v}")
    for key in ("company", "email", "phone", "address"):
        if pii[key] not in full_text:
            errors.append(f"signature PII missing: {key}={pii[key]}")
    for cand in record.get("masking_candidates", []):
        v = cand.get("value", "")
        if v and v not in full_text:
            errors.append(f"masking_candidate not found: {v}")
    if not record.get("subject", "").strip():
        errors.append("subject is empty")
    return errors

# ---------------------------------------------------------------------------
# メイン生成
# ---------------------------------------------------------------------------
def generate_one(
    client: OpenAI,
    model: str,
    mail_id: str,
    temperature: float,
    max_retries: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = enrich_plan(random.choice(PLAN_PRESETS))
    pii  = build_pii()

    for attempt in range(1, max_retries + 1):
        core     = generate_body_core(client, model, plan, temperature)
        core     = inject_pii_into_core(core, plan, pii)
        greeting = random.choice(GREETINGS)
        closing  = random.choice(CLOSINGS)
        body     = assemble_body(greeting, core, closing)

        if has_unfilled_placeholder(body):
            print(f"  [RETRY {attempt}/{max_retries}] placeholder detected.", file=sys.stderr)
            if attempt < max_retries:
                continue
            print(f"  [WARN] giving up after {max_retries} retries.", file=sys.stderr)
        break

    subject    = generate_subject(plan)
    sig        = build_signature(pii)
    candidates = build_masking_candidates(pii, plan["topic"])

    record: dict[str, Any] = {
        "id":                 mail_id,
        "model":              model,
        "plan":               plan,
        "subject":            subject,
        "body":               body,
        "signature":          sig,
        "masking_candidates": candidates,
    }
    errors = validate(record, pii)
    record["_errors"] = errors
    return record, pii


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ハイブリッド版ダミーメール生成 v2")
    p.add_argument("--count",       "-n", type=int,   default=5)
    p.add_argument("--model",       "-m",             default="gemma3:4b")
    p.add_argument("--base-url",          default="http://localhost:11434/v1/")
    p.add_argument("--output",      "-o", default=None)
    p.add_argument("--temperature",       type=float, default=0.7)
    p.add_argument("--seed",              type=int,   default=None)
    p.add_argument("--id-prefix",         default="mail")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    client  = OpenAI(base_url=args.base_url, api_key="ollama")
    records: list[dict[str, Any]] = []

    for i in range(1, args.count + 1):
        mail_id = f"{args.id_prefix}_{i}"
        print(f"[{i}/{args.count}] Generating {mail_id} ...", file=sys.stderr)
        try:
            record, pii = generate_one(client, args.model, mail_id, args.temperature)
            errors = record.pop("_errors", [])
            pii.pop("name_targets", None)
            if errors:
                print(f"  [WARN] {errors}", file=sys.stderr)
            else:
                print(f"  [OK]", file=sys.stderr)
            records.append(record)
        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)

    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote {len(records)} records to {args.output}", file=sys.stderr)
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()