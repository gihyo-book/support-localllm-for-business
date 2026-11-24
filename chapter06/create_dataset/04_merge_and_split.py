"""アノテーション済みデータの統合・分割・ChatML形式変換.

- JSQuAD / JEMHopQA のアノテーション済みデータを読み込み
- アノテーション失敗レコードを除外
- train / dev / test_jsquad / test_jemhopqa に分割
- ChatML形式（messages配列）に変換して出力

分割ルール:
  train        : JSQuAD train
  dev          : JSQuAD validation（test_jsquad分を除く）
  test_jsquad  : JSQuAD validationからランダム抽出 (~1,000件)
  test_jemhopqa: JEMHopQA 全件

JEMHopQAのcontext切り詰め:
  そこで JEMHopQA テストデータのみ、tokenizer ベースで「推論時の chat
  template 後の総トークン数」が 110_000 トークン以下になるよう context を
  事前に切り詰める。Qwen3.5-4B の max_position_embeddings (131,072) に対し
  生成分・余裕を確保した値。質問文と generation prompt は必ず保持する。
"""

import copy
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSQUAD_PATH = DATA_DIR / "jsquad_annotated.jsonl"
JEMHOPQA_PATH = DATA_DIR / "jemhopqa_annotated.jsonl"

SEED = 42
JSQUAD_TEST_SIZE = 1000

# JEMHopQAテストデータの context 切り詰め用パラメータ
TOKENIZER_NAME = "Qwen/Qwen3.5-4B"
MAX_PROMPT_TOKENS = 110000
QUESTION_MARKER = "\n\n質問:"

SYSTEM_PROMPT = (
    "あなたは文書読解アシスタントです。与えられた文脈と質問に基づき、"
    "回答を以下のJSON形式で出力してください。\n"
    "- answer: 質問に対する回答\n"
    "- evidence: 回答の根拠となる文脈中の文\n"
    "- related_entities: 回答に関連する固有表現のリスト "
    '(各要素は {"name": "...", "type": "人物|地名|組織|日時|作品|数値|概念|物質|製品|生物|事象|言語|その他", "relation": "..."})\n'
    "- answer_type: 回答の種類 (person|location|organization|date|number|other)"
)


def load_jsonl(path: Path) -> list[dict]:
    """JSONLファイルを読み込み、アノテーション失敗レコードを除外."""
    records = []
    failed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("_annotation_failed"):
                failed += 1
                continue
            records.append(rec)
    if failed:
        print(
            f"  Excluded {failed} failed annotations from {path.name}", file=sys.stderr
        )
    return records


def build_user_message(context: str, question: str) -> str:
    """userメッセージを構築."""
    return f"以下の文脈を読み、質問に答えてください。\n\n文脈:\n{context}\n\n質問: {question}"


def build_assistant_message(rec: dict) -> str:
    """assistantメッセージ（構造化JSON）を構築."""
    output = {
        "answer": rec["answer"],
        "evidence": rec["evidence"],
        "related_entities": rec["related_entities"],
        "answer_type": rec["answer_type"],
    }
    return json.dumps(output, ensure_ascii=False)


def to_chatml(rec: dict) -> dict:
    """レコードをChatML形式に変換."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_message(rec["context"], rec["question"]),
            },
            {"role": "assistant", "content": build_assistant_message(rec)},
        ]
    }


def truncate_context_in_chatml(
    chatml_rec: dict, tokenizer, max_prompt_tokens: int
) -> tuple[dict, int]:
    """ChatMLレコードの user メッセージ内 context をトークン単位で切り詰める.

    推論時のモデル入力は system + user + generation prompt（assistantは未生成）。
    その総トークン数が max_prompt_tokens 以下になるよう、user 内の context
    部分の末尾だけを切り詰める。質問文・generation prompt は保持。

    Returns:
        (切り詰め後のレコード, 元の context トークン数 - 切り詰め後トークン数)
    """
    msgs = copy.deepcopy(chatml_rec["messages"])
    user_idx = next(i for i, m in enumerate(msgs) if m["role"] == "user")
    user_content = msgs[user_idx]["content"]

    marker_pos = user_content.rfind(QUESTION_MARKER)
    context_part = user_content[:marker_pos]
    question_part = user_content[marker_pos:]

    # context 以外の chat template 込みトークン数を計測する
    # （推論時形を再現するため assistant は除外し generation prompt を付与）
    shell_msgs = [m for m in msgs if m["role"] != "assistant"]
    shell_msgs[
        next(i for i, m in enumerate(shell_msgs) if m["role"] == "user")
    ]["content"] = question_part
    shell_text = tokenizer.apply_chat_template(
        shell_msgs, tokenize=False, add_generation_prompt=True
    )
    shell_tokens = len(tokenizer.encode(shell_text, add_special_tokens=False))

    context_budget = max_prompt_tokens - shell_tokens
    context_ids = tokenizer.encode(context_part, add_special_tokens=False)
    if len(context_ids) <= context_budget:
        return chatml_rec, 0

    truncated_ids = context_ids[:context_budget]
    truncated_context = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    msgs[user_idx]["content"] = truncated_context + question_part
    return {"messages": msgs}, len(context_ids) - context_budget


def write_jsonl(records: list[dict], path: Path):
    """JSONL形式で書き出し."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(records):,} records", file=sys.stderr)


def main():
    print("Loading annotated data...", file=sys.stderr)
    jsquad = load_jsonl(JSQUAD_PATH)
    jemhopqa = load_jsonl(JEMHOPQA_PATH)

    # --- JSQuAD分割 ---
    jsquad_train = [r for r in jsquad if r["split"] == "train"]
    jsquad_val = [r for r in jsquad if r["split"] == "validation"]

    # validationからtest_jsquadを抽出
    random.seed(SEED)
    random.shuffle(jsquad_val)
    jsquad_test = jsquad_val[:JSQUAD_TEST_SIZE]
    jsquad_dev = jsquad_val[JSQUAD_TEST_SIZE:]

    print("\n=== Split sizes ===", file=sys.stderr)
    print(f"  train:          {len(jsquad_train):,} (JSQuAD train)", file=sys.stderr)
    print(
        f"  dev:            {len(jsquad_dev):,} (JSQuAD validation remainder)",
        file=sys.stderr,
    )
    print(
        f"  test_jsquad:    {len(jsquad_test):,} (JSQuAD validation holdout)",
        file=sys.stderr,
    )
    print(f"  test_jemhopqa:  {len(jemhopqa):,} (JEMHopQA all)", file=sys.stderr)

    # --- ChatML変換 ---
    print("\nConverting to ChatML format...", file=sys.stderr)
    train_chatml = [to_chatml(r) for r in jsquad_train]
    dev_chatml = [to_chatml(r) for r in jsquad_dev]
    test_jsquad_chatml = [to_chatml(r) for r in jsquad_test]
    test_jemhopqa_chatml = [to_chatml(r) for r in jemhopqa]

    # JEMHopQA テストデータのみ tokenizer で MAX_PROMPT_TOKENS 以下に切り詰める
    print(
        f"\nLoading tokenizer ({TOKENIZER_NAME}) for JEMHopQA truncation...",
        file=sys.stderr,
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(
        f"Truncating test_jemhopqa context so total prompt <= {MAX_PROMPT_TOKENS:,} tokens...",
        file=sys.stderr,
    )
    truncated_count = 0
    truncated_records = []
    for rec in test_jemhopqa_chatml:
        new_rec, removed = truncate_context_in_chatml(rec, tokenizer, MAX_PROMPT_TOKENS)
        if removed > 0:
            truncated_count += 1
        truncated_records.append(new_rec)
    test_jemhopqa_chatml = truncated_records
    print(
        f"  Truncated {truncated_count}/{len(test_jemhopqa_chatml)} records",
        file=sys.stderr,
    )

    # --- 書き出し ---
    print(f"\nWriting to {OUTPUT_DIR}/...", file=sys.stderr)
    write_jsonl(train_chatml, OUTPUT_DIR / "train.jsonl")
    write_jsonl(dev_chatml, OUTPUT_DIR / "dev.jsonl")
    write_jsonl(test_jsquad_chatml, OUTPUT_DIR / "test_jsquad.jsonl")
    write_jsonl(test_jemhopqa_chatml, OUTPUT_DIR / "test_jemhopqa.jsonl")

    # --- 統計 ---
    print("\n=== Token length estimates (chars) ===", file=sys.stderr)
    for name, data in [
        ("train", train_chatml),
        ("dev", dev_chatml),
        ("test_jsquad", test_jsquad_chatml),
        ("test_jemhopqa", test_jemhopqa_chatml),
    ]:
        lengths = [sum(len(m["content"]) for m in r["messages"]) for r in data]
        avg = sum(lengths) / len(lengths) if lengths else 0
        mx = max(lengths) if lengths else 0
        print(
            f"  {name:20s}: avg {avg:,.0f} chars, max {mx:,.0f} chars", file=sys.stderr
        )

    # --- サンプル表示 ---
    print("\n=== Sample (train[0]) ===", file=sys.stderr)
    sample = train_chatml[0]
    for msg in sample["messages"]:
        role = msg["role"]
        content = msg["content"][:200]
        print(f"  [{role}] {content}...", file=sys.stderr)

    print(f"\nDone! Output: {OUTPUT_DIR}/", file=sys.stderr)


if __name__ == "__main__":
    main()
