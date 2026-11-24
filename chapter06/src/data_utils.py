"""データ読み込み・few-shot構築・サブサンプリング."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "create_dataset" / "data" / "final"


def load_jsonl(path: str | Path) -> list[dict]:
    """Load JSONL file and return list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_dataset(split: str) -> list[dict]:
    """Load a dataset split: train, dev, test_jsquad, test_jemhopqa."""
    path = DATA_DIR / f"{split}.jsonl"
    return load_jsonl(path)


def extract_ground_truth(record: dict) -> dict:
    """Extract ground truth from assistant message in ChatML format."""
    for msg in record["messages"]:
        if msg["role"] == "assistant":
            return json.loads(msg["content"])
    raise ValueError("No assistant message found")


def extract_messages_without_assistant(record: dict) -> list[dict]:
    """Return system + user messages (no assistant) for inference."""
    return [msg for msg in record["messages"] if msg["role"] != "assistant"]


def get_system_message(record: dict) -> str:
    """Extract system message content."""
    for msg in record["messages"]:
        if msg["role"] == "system":
            return msg["content"]
    return ""


def build_few_shot_messages(
    train_data: list[dict],
    n_shots: int,
    seed: int = 42,
) -> list[dict]:
    """Select n_shots examples from train ensuring answer_type diversity.

    Returns list of messages (user/assistant pairs) to prepend.
    """
    if n_shots == 0:
        return []

    import random

    rng = random.Random(seed)

    # Group by answer_type
    by_type: dict[str, list[dict]] = {}
    for rec in train_data:
        gt = extract_ground_truth(rec)
        atype = gt.get("answer_type", "other")
        by_type.setdefault(atype, []).append(rec)

    # Round-robin selection from each type
    selected = []
    types = sorted(by_type.keys())
    type_iters = {
        t: iter(rng.sample(recs, min(len(recs), n_shots)))
        for t, recs in by_type.items()
    }

    while len(selected) < n_shots:
        added = False
        for t in types:
            if len(selected) >= n_shots:
                break
            try:
                rec = next(type_iters[t])
                selected.append(rec)
                added = True
            except StopIteration:
                continue
        if not added:
            break

    # Build messages from selected examples
    few_shot_msgs = []
    for rec in selected:
        for msg in rec["messages"]:
            if msg["role"] in ("user", "assistant"):
                few_shot_msgs.append(msg)

    return few_shot_msgs


def build_inference_messages(
    record: dict,
    few_shot_msgs: list[dict] | None = None,
) -> list[dict]:
    """Build full message list for inference.

    [system] + [few_shot user/assistant pairs] + [test user]
    """
    messages = []

    # System message
    for msg in record["messages"]:
        if msg["role"] == "system":
            messages.append(msg)
            break

    # Few-shot examples
    if few_shot_msgs:
        messages.extend(few_shot_msgs)

    # Test query (user message)
    for msg in record["messages"]:
        if msg["role"] == "user":
            messages.append(msg)
            break

    return messages


def stratified_subsample(
    data: list[dict],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Stratified subsampling by answer_type, preserving distribution."""
    import random

    rng = random.Random(seed)

    if n >= len(data):
        return data

    # Group by answer_type
    by_type: dict[str, list[int]] = {}
    for i, rec in enumerate(data):
        gt = extract_ground_truth(rec)
        atype = gt.get("answer_type", "other")
        by_type.setdefault(atype, []).append(i)

    # Calculate per-type counts proportionally
    total = len(data)
    selected_indices = []
    remaining = n

    types_sorted = sorted(by_type.keys())
    for i, t in enumerate(types_sorted):
        indices = by_type[t]
        if i == len(types_sorted) - 1:
            count = remaining
        else:
            count = max(1, round(len(indices) / total * n))
            count = min(count, remaining, len(indices))
        rng.shuffle(indices)
        selected_indices.extend(indices[:count])
        remaining -= count
        if remaining <= 0:
            break

    selected_indices.sort()
    return [data[i] for i in selected_indices]
