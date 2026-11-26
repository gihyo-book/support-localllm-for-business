"""JSQuADデータセットの取得・整形・evidence自動抽出.

- HuggingFace sbintuitions/JSQuAD を取得
- context の [SEP] を整形
- answer_start を利用して evidence（回答を含む文）をルールベースで抽出
- data/jsquad_base.jsonl に出力
"""

import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "jsquad_base.jsonl"


# ============================================================
# Context 整形
# ============================================================
def format_context(raw_context: str) -> tuple[str, str, str]:
    """「タイトル [SEP] 本文」を分離・整形する.

    Returns:
        (formatted_context, title, body)
    """
    sep = " [SEP] "
    idx = raw_context.find(sep)
    if idx != -1:
        title = raw_context[:idx].strip()
        body = raw_context[idx + len(sep) :].strip()
    else:
        title = ""
        body = raw_context.strip()
    formatted = f"「{title}」\n{body}" if title else body
    return formatted, title, body


# ============================================================
# 文分割
# ============================================================
def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """テキストを句点「。」で文分割し、(start, end, sentence) のリストを返す."""
    sentences = []
    start = 0
    for m in re.finditer(r"。", text):
        end = m.end()
        sentences.append((start, end, text[start:end]))
        start = end
    # 末尾に句点がない残り部分
    if start < len(text):
        remaining = text[start:].strip()
        if remaining:
            sentences.append((start, len(text), remaining))
    return sentences


# ============================================================
# Evidence 抽出
# ============================================================
def extract_evidence(
    raw_context: str,
    body: str,
    question: str,
    answer_text: str,
    answer_start: int,
) -> tuple[str, str]:
    """回答を含む文を抽出してevidenceとする.

    Returns:
        (evidence, method) — method は抽出方法の記録用
    """
    sep = " [SEP] "
    sep_idx = raw_context.find(sep)
    body_offset = sep_idx + len(sep) if sep_idx != -1 else 0

    sentences = split_sentences(body)
    if not sentences:
        return body.strip(), "fallback_no_sentences"

    # Case 1: answer_start が本文内を指す場合
    ans_pos_in_body = answer_start - body_offset
    if ans_pos_in_body >= 0:
        for s_start, s_end, sent in sentences:
            if s_start <= ans_pos_in_body < s_end:
                return sent.strip(), "answer_start_in_body"

    # Case 2: answer_start がタイトル領域だが、本文にも回答テキストがある場合
    for s_start, s_end, sent in sentences:
        if answer_text in sent:
            return sent.strip(), "answer_text_found_in_body"

    # Case 3: 質問のキーワードを含む文を探す
    # 質問から助詞・疑問詞等を除いた内容語でマッチ
    q_clean = re.sub(r"[はがのをにへでとかもやっていうた、。？?\s]", " ", question)
    q_keywords = [w for w in q_clean.split() if len(w) >= 2]

    best_sent = None
    best_score = 0
    for _s_start, _s_end, sent in sentences:
        score = sum(1 for kw in q_keywords if kw in sent)
        if score > best_score:
            best_score = score
            best_sent = sent
    if best_sent and best_score > 0:
        return best_sent.strip(), "keyword_match_in_body"

    # Case 4: 最終フォールバック → 先頭文
    return sentences[0][2].strip(), "fallback_first_sentence"


# ============================================================
# メイン処理
# ============================================================
def main():
    print("Loading JSQuAD from HuggingFace...", file=sys.stderr)
    ds = load_dataset("sbintuitions/JSQuAD")

    records = []
    method_counts: dict[str, int] = {}

    for split_name in ["train", "validation"]:
        for item in ds[split_name]:
            raw_context = item["context"]
            question = item["question"]
            answers = item["answers"]
            answer_text = answers["text"][0] if answers["text"] else ""
            answer_start = answers["answer_start"][0] if answers["answer_start"] else -1

            formatted_context, title, body = format_context(raw_context)

            evidence, method = extract_evidence(
                raw_context, body, question, answer_text, answer_start
            )
            method_counts[method] = method_counts.get(method, 0) + 1

            records.append(
                {
                    "split": split_name,
                    "context": formatted_context,
                    "title": title,
                    "question": question,
                    "answer": answer_text,
                    "evidence": evidence,
                    "evidence_method": method,
                }
            )

    # 出力
    print(f"Writing {len(records)} records to {OUTPUT_PATH}...", file=sys.stderr)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 統計レポート
    print("\n=== Evidence Extraction Stats ===", file=sys.stderr)
    total = len(records)
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count:,} ({count / total * 100:.1f}%)", file=sys.stderr)
    print(f"  Total: {total:,}", file=sys.stderr)

    # 各メソッドのサンプルを表示
    print("\n=== Samples by method ===", file=sys.stderr)
    shown: set[str] = set()
    for rec in records:
        m = rec["evidence_method"]
        if m not in shown:
            shown.add(m)
            print(f"\n[{m}]", file=sys.stderr)
            print(f"  Q: {rec['question']}", file=sys.stderr)
            print(f"  A: {rec['answer']}", file=sys.stderr)
            print(f"  Evidence: {rec['evidence'][:100]}", file=sys.stderr)
        if len(shown) == len(method_counts):
            break

    print(f"\nDone! Output: {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
