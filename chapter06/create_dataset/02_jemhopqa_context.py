"""JEMHopQAデータセットの取得・Wikipediaコンテキスト付与・evidence抽出.

- HuggingFace sbintuitions/JEMHopQA v1.2 を取得
- page_ids から日本語Wikipedia APIでページ全文を取得（キャッシュ付き）
- derivations を利用して evidence を抽出
- data/jemhopqa_base.jsonl に出力
"""

import json
import sys
import time
from pathlib import Path

import requests
from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "jemhopqa_base.jsonl"

CACHE_DIR = OUTPUT_DIR / "wiki_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "pages.json"

API_URL = "https://ja.wikipedia.org/w/api.php"
USER_AGENT = "FineTuningResearch/1.0 (academic research; contact: research@example.com)"
REQUEST_INTERVAL = 0.5  # リクエスト間隔(秒)
SAVE_EVERY = 100  # N件ごとにキャッシュ保存


# ============================================================
# Wikipedia API
# ============================================================
def load_cache() -> dict[str, dict]:
    """キャッシュ済みのWikipediaページを読み込む."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict[str, dict]):
    """キャッシュを保存."""
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


def fetch_wikipedia_pages(
    page_ids: list[str], cache: dict[str, dict]
) -> dict[str, dict]:
    """Wikipedia APIでページを1件ずつ取得する.

    extracts APIはレスポンスサイズ制限のため、バッチリクエストでは
    1ページ分しかextractが返らない場合がある。確実に全文を取得するため
    個別リクエストとする。

    Returns:
        {page_id: {"title": str, "extract": str}} の辞書
    """
    # キャッシュにないID、またはextractが空のIDを取得対象とする
    missing = [
        pid for pid in page_ids if pid not in cache or not cache[pid].get("extract", "")
    ]
    if not missing:
        return cache

    print(
        f"  Fetching {len(missing)} pages from Wikipedia API "
        f"(cached: {len(page_ids) - len(missing)})...",
        file=sys.stderr,
    )

    for i, pid in enumerate(missing):
        resp = requests.get(
            API_URL,
            params={
                "action": "query",
                "pageids": pid,
                "prop": "extracts",
                "explaintext": True,
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        page_data = data.get("query", {}).get("pages", {}).get(pid, {})
        title = page_data.get("title", "")
        extract = page_data.get("extract", "")
        cache[pid] = {"title": title, "extract": extract}

        if (i + 1) % SAVE_EVERY == 0:
            save_cache(cache)
            print(f"    {i + 1}/{len(missing)} fetched", file=sys.stderr)

        # レート制限対応
        if i + 1 < len(missing):
            time.sleep(REQUEST_INTERVAL)

    print(f"    {len(missing)}/{len(missing)} fetched", file=sys.stderr)

    # キャッシュを保存
    save_cache(cache)
    return cache


# ============================================================
# Context / Evidence 構成
# ============================================================
def build_context(page_ids: list[str], cache: dict[str, dict]) -> str:
    """2つのWikipediaページを結合してcontextを構成する."""
    parts = []
    for pid in page_ids:
        page = cache.get(pid, {})
        title = page.get("title", f"(page_id={pid})")
        extract = page.get("extract", "")
        parts.append(f"「{title}」\n{extract}")
    return "\n\n".join(parts)


def extract_evidence(
    context: str,
    derivations: list[dict],
) -> tuple[str, str]:
    """derivationsの値をcontextから検索してevidenceを抽出する.

    Returns:
        (evidence, method)
    """
    evidence_parts = []

    for deriv in derivations:
        subject = deriv.get("0", "")
        values = deriv.get("2", [])

        # 値を含む文をcontextから探す
        found = False
        for value in values:
            # contextを。で分割して該当文を探す
            sentences = [s.strip() for s in context.split("。") if s.strip()]
            for sent in sentences:
                if value in sent:
                    evidence_parts.append(sent + "。")
                    found = True
                    break
            if found:
                break

        if not found:
            # 値が見つからない場合、主語を含む文をフォールバックとして探す
            sentences = [s.strip() for s in context.split("。") if s.strip()]
            for sent in sentences:
                if subject in sent:
                    evidence_parts.append(sent + "。")
                    break

    if evidence_parts:
        method = "derivation_value_match"
        return " ".join(dict.fromkeys(evidence_parts)), method  # 重複排除しつつ順序保持

    return "", "no_evidence_found"


# ============================================================
# メイン処理
# ============================================================
def main():
    print("Loading JEMHopQA from HuggingFace...", file=sys.stderr)
    ds = load_dataset("sbintuitions/JEMHopQA", "v1.2")

    # 全page_idsを収集
    all_page_ids = set()
    for split in ds:
        for item in ds[split]:
            for pid in item["page_ids"]:
                all_page_ids.add(pid)

    print(f"Unique Wikipedia pages to fetch: {len(all_page_ids)}", file=sys.stderr)

    # Wikipedia取得（キャッシュ付き）
    cache = load_cache()
    cache = fetch_wikipedia_pages(list(all_page_ids), cache)

    # データセット構築
    records = []
    method_counts: dict[str, int] = {}
    empty_extract_count = 0

    for split_name in ds:
        for item in ds[split_name]:
            page_ids = item["page_ids"]
            context = build_context(page_ids, cache)

            # extractが空のページがないかチェック
            for pid in page_ids:
                if not cache.get(pid, {}).get("extract", ""):
                    empty_extract_count += 1

            derivations = item.get("derivations", [])
            evidence, method = extract_evidence(context, derivations)
            method_counts[method] = method_counts.get(method, 0) + 1

            records.append(
                {
                    "split": split_name,
                    "qid": item["qid"],
                    "type": item.get("type", ""),
                    "context": context,
                    "question": item["question"],
                    "answer": item["answer"],
                    "evidence": evidence,
                    "evidence_method": method,
                    "derivations": derivations,
                    "page_ids": page_ids,
                    "time_dependent": item.get("time_dependent", False),
                }
            )

    # 出力
    print(f"\nWriting {len(records)} records to {OUTPUT_PATH}...", file=sys.stderr)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 統計
    print("\n=== Evidence Extraction Stats ===", file=sys.stderr)
    total = len(records)
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count:,} ({count / total * 100:.1f}%)", file=sys.stderr)
    print(f"  Total: {total:,}", file=sys.stderr)

    if empty_extract_count > 0:
        print(
            f"\n  WARNING: {empty_extract_count} page references had empty extracts",
            file=sys.stderr,
        )

    # context長の統計
    ctx_lens = [len(r["context"]) for r in records]
    print("\n=== Context Length Stats ===", file=sys.stderr)
    print(f"  Mean: {sum(ctx_lens) / len(ctx_lens):.0f} chars", file=sys.stderr)
    print(f"  Min: {min(ctx_lens)}, Max: {max(ctx_lens)}", file=sys.stderr)

    # サンプル表示
    print("\n=== Sample ===", file=sys.stderr)
    rec = records[0]
    print(f"  Q: {rec['question']}", file=sys.stderr)
    print(f"  A: {rec['answer']}", file=sys.stderr)
    print(f"  Evidence: {rec['evidence'][:200]}", file=sys.stderr)
    print(f"  Context[:200]: {rec['context'][:200]}", file=sys.stderr)

    print(f"\nDone! Output: {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
