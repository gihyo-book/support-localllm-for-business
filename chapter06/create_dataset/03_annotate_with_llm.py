"""LLMアノテーション: related_entities と answer_type を生成.

- data/jsquad_base.jsonl と data/jemhopqa_base.jsonl を読み込み
- OpenAI互換APIで related_entities と answer_type を一括生成
- data/jsquad_annotated.jsonl と data/jemhopqa_annotated.jsonl に出力
- チェックポイント機能で中断・再開可能
- LLM_RECORDS_PER_PROMPT > 1 でバッチプロンプティング（1回のAPI呼び出しで複数件処理）
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# .envから環境変数を読み込み
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

# ============================================================
# 設定
# ============================================================
BASE_URL = os.environ["LLM_BASE_URL"]
API_KEY = os.environ["LLM_KEY"]
MODEL = os.environ["LLM_MODEL"]

CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "5"))
RECORDS_PER_PROMPT = int(os.environ.get("LLM_RECORDS_PER_PROMPT", "1"))
WRITE_EVERY = int(os.environ.get("LLM_WRITE_EVERY", "50"))
MAX_RETRIES = 3

# パス
DATA_DIR = Path(__file__).parent / "data"
PROMPT_PATH = Path(__file__).parent / "prompts" / "annotate.txt"

DATASETS = {
    "jemhopqa": {
        "input": DATA_DIR / "jemhopqa_base.jsonl",
        "output": DATA_DIR / "jemhopqa_annotated.jsonl",
    },
    "jsquad": {
        "input": DATA_DIR / "jsquad_base.jsonl",
        "output": DATA_DIR / "jsquad_annotated.jsonl",
    },
}

# バリデーション用定数
VALID_ANSWER_TYPES = {"date", "person", "location", "organization", "number", "other"}
VALID_ENTITY_TYPES = {
    "人物",
    "地名",
    "組織",
    "日時",
    "作品",
    "数値",
    "概念",
    "物質",
    "製品",
    "生物",
    "事象",
    "言語",
    "その他",
}


# ============================================================
# プロンプト
# ============================================================
SYSTEM_PROMPT = (
    "あなたは日本語QAデータの構造化アノテーターです。"
    "指示に従い、JSON形式のみで回答してください。説明文や補足は一切不要です。"
)

SYSTEM_PROMPT_BATCH = (
    "あなたは日本語QAデータの構造化アノテーターです。"
    "複数のQAペアを一度にアノテーションします。"
    "JSON配列形式のみで回答してください。説明文は不要です。"
)


def load_user_prompt_template() -> str:
    """プロンプトテンプレートをファイルから読み込む."""
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def build_batch_prompt(records: list[dict]) -> str:
    """複数レコード用のプロンプトを構築する."""
    parts = []
    for i, rec in enumerate(records):
        parts.append(
            f"[{i}]\n"
            f"質問: {rec['question']}\n"
            f"回答: {rec['answer']}\n"
            f"根拠文: {rec.get('evidence', '')}"
        )
    items_text = "\n\n".join(parts)
    n = len(records)
    return (
        f"以下の{n}件のQAペアについて、それぞれのrelated_entitiesと"
        f"answer_typeを抽出してください。\n\n"
        f"{items_text}\n\n"
        f"## ルール\n"
        f"related_entities: 回答に関連する固有表現を1〜5個。回答自体も含む。\n"
        f"- name: エンティティ名（テキスト中の表記のまま）\n"
        f"- type: 人物 / 地名 / 組織 / 日時 / 作品 / 数値 / 概念 / 物質 / 製品 / 生物 / 事象 / 言語 / その他\n"
        f"- relation: 回答との関係を簡潔に\n\n"
        f"answer_type: person / location / organization / date / number / other\n\n"
        f"## 出力（{n}要素のJSON配列のみ）\n"
        f'[{{"related_entities": [...], "answer_type": "..."}}, ...]'
    )


# ============================================================
# JSONL I/O
# ============================================================
def read_jsonl(path: Path) -> list[dict]:
    """JSONLファイルを読み込む."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def count_valid_lines(path: Path) -> int:
    """出力ファイルの有効なJSONL行数を数える（レジューム用）."""
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    break  # 不完全な行で停止
    return count


def truncate_to_valid(path: Path, valid_count: int):
    """出力ファイルをvalid_count行に切り詰める（不完全な末尾を除去）."""
    if not path.exists():
        return
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    json.loads(line)
                    lines.append(line)
                except json.JSONDecodeError:
                    break
            if len(lines) >= valid_count:
                break
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# ============================================================
# JSON抽出・バリデーション
# ============================================================
def extract_json_object(text: str) -> dict | None:
    """LLMレスポンスからJSONオブジェクトを抽出する."""
    # ```json ... ``` ブロック
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    # 直接JSON
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    # { ... } を抽出
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return None


def extract_json_array(text: str) -> list | None:
    """LLMレスポンスからJSON配列を抽出する."""
    # ```json ... ``` ブロック
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    # 直接JSON
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # [ ... ] を抽出
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None


def validate_annotation(data: dict) -> dict | None:
    """アノテーション結果のバリデーションと正規化."""
    if not isinstance(data, dict):
        return None

    answer_type = data.get("answer_type", "other")
    if answer_type not in VALID_ANSWER_TYPES:
        answer_type = "other"

    entities = data.get("related_entities", [])
    if not isinstance(entities, list):
        entities = []

    validated = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        name = ent.get("name", "")
        if not name:
            continue
        etype = ent.get("type", "その他")
        if etype not in VALID_ENTITY_TYPES:
            etype = "その他"
        relation = ent.get("relation", "")
        validated.append({"name": name, "type": etype, "relation": relation})

    if not validated:
        return None

    return {"related_entities": validated, "answer_type": answer_type}


FAILED_ANNOTATION = {
    "related_entities": [],
    "answer_type": "other",
    "_annotation_failed": True,
}


# ============================================================
# LLM呼び出し（単一レコード）
# ============================================================
async def annotate_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    record: dict,
    template: str,
) -> list[dict]:
    """1レコードをLLMでアノテーション. 返り値はリスト（統一インターフェース）."""
    prompt = (
        template.replace("{question}", record["question"])
        .replace("{answer}", record["answer"])
        .replace("{evidence}", record.get("evidence", ""))
    )

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=16384,
                )
            text = resp.choices[0].message.content or ""
            parsed = extract_json_object(text)
            if parsed:
                validated = validate_annotation(parsed)
                if validated:
                    return [validated]
            if attempt < MAX_RETRIES - 1:
                print(
                    f"    Validation failed (attempt {attempt + 1}), "
                    f"retrying... raw: {text[:80]}",
                    file=sys.stderr,
                )
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = 2 ** (attempt + 1)
                print(
                    f"    API error (attempt {attempt + 1}): {e}, "
                    f"retrying in {delay}s...",
                    file=sys.stderr,
                )
                await asyncio.sleep(delay)
            else:
                print(f"    API error (final): {e}", file=sys.stderr)

    return [FAILED_ANNOTATION]


# ============================================================
# LLM呼び出し（バッチ＝複数レコード/1プロンプト）
# ============================================================
async def annotate_multi(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    records: list[dict],
) -> list[dict]:
    """複数レコードを1つのプロンプトでアノテーション."""
    n = len(records)
    prompt = build_batch_prompt(records)

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_BATCH},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=16384,
                )
            text = resp.choices[0].message.content or ""
            parsed = extract_json_array(text)
            if parsed and len(parsed) == n:
                results = []
                for item in parsed:
                    validated = validate_annotation(item)
                    results.append(validated if validated else FAILED_ANNOTATION)
                return results
            if attempt < MAX_RETRIES - 1:
                arr_len = len(parsed) if parsed else 0
                print(
                    f"    Batch parse failed (attempt {attempt + 1}), "
                    f"expected {n} got {arr_len}, retrying...",
                    file=sys.stderr,
                )
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = 2 ** (attempt + 1)
                print(
                    f"    Batch API error (attempt {attempt + 1}): {e}, "
                    f"retrying in {delay}s...",
                    file=sys.stderr,
                )
                await asyncio.sleep(delay)
            else:
                print(f"    Batch API error (final): {e}", file=sys.stderr)

    return [FAILED_ANNOTATION] * n


# ============================================================
# メイン処理
# ============================================================
async def process_dataset(name: str, input_path: Path, output_path: Path):
    """1つのデータセットを処理."""
    records = read_jsonl(input_path)
    total = len(records)

    # レジューム: 既存出力ファイルの有効行数から再開位置を決定
    completed = count_valid_lines(output_path)
    if completed > 0:
        truncate_to_valid(output_path, completed)

    if completed >= total:
        print(f"[{name}] Already completed ({total} records)", file=sys.stderr)
        return

    if completed > 0:
        print(f"[{name}] Resuming from {completed}/{total}", file=sys.stderr)
    else:
        print(f"[{name}] Processing {total} records", file=sys.stderr)

    template = load_user_prompt_template()
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    failed_count = 0
    start_time = time.time()

    remaining = records[completed:]
    rpp = RECORDS_PER_PROMPT

    # レコードをプロンプトグループに分割
    groups = [remaining[i : i + rpp] for i in range(0, len(remaining), rpp)]

    # グループをディスク書き込み単位のウィンドウに分割
    groups_per_window = max(WRITE_EVERY // rpp, 1)

    for win_start in range(0, len(groups), groups_per_window):
        window = groups[win_start : win_start + groups_per_window]

        if rpp == 1:
            tasks = [
                annotate_one(client, semaphore, grp[0], template) for grp in window
            ]
        else:
            tasks = [annotate_multi(client, semaphore, grp) for grp in window]

        batch_results = await asyncio.gather(*tasks)

        # 結果をフラットにして追記
        with open(output_path, "a", encoding="utf-8") as f:
            for grp, annotations in zip(window, batch_results):
                for rec, ann in zip(grp, annotations):
                    out = {**rec, **ann}
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

        window_failed = sum(
            1
            for annotations in batch_results
            for a in annotations
            if a.get("_annotation_failed")
        )
        failed_count += window_failed
        done_now = completed + (win_start + len(window)) * rpp
        done_now = min(done_now, total)

        elapsed = time.time() - start_time
        processed = done_now - completed
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - done_now) / rate if rate > 0 else 0

        print(
            f"  [{name}] {done_now}/{total} ({done_now / total * 100:.1f}%) "
            f"| {rate:.1f} rec/s | ETA {eta:.0f}s | failed={failed_count}",
            file=sys.stderr,
        )

    elapsed_total = time.time() - start_time
    print(
        f"\n[{name}] Done! {total} records in {elapsed_total:.0f}s, "
        f"{failed_count} failed → {output_path}",
        file=sys.stderr,
    )


async def main_async(targets: list[str]):
    """指定されたデータセットを順番に処理."""
    for target in targets:
        cfg = DATASETS[target]
        await process_dataset(target, cfg["input"], cfg["output"])


def main():
    # コマンドライン引数: jsquad, jemhopqa, all (デフォルト)
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg == "all":
        targets = ["jemhopqa", "jsquad"]  # JEMHopQA(小)を先に処理
    elif arg in DATASETS:
        targets = [arg]
    else:
        print(f"Usage: {sys.argv[0]} [jsquad|jemhopqa|all]", file=sys.stderr)
        sys.exit(1)

    print("=== LLM Annotation ===", file=sys.stderr)
    print(f"  API: {BASE_URL}", file=sys.stderr)
    print(f"  Model: {MODEL}", file=sys.stderr)
    print(f"  Concurrency: {CONCURRENCY}", file=sys.stderr)
    print(f"  Records/prompt: {RECORDS_PER_PROMPT}", file=sys.stderr)
    print(f"  Targets: {targets}", file=sys.stderr)
    print(file=sys.stderr)

    asyncio.run(main_async(targets))


if __name__ == "__main__":
    main()
