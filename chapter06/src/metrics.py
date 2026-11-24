"""評価指標モジュール: JSON Parse, Schema, Answer EM/F1, Evidence ROUGE-L, Entity F1, Answer Type Acc."""

import json
import re
import unicodedata


# ---------------------------------------------------------------------------
# JSON Parse helpers
# ---------------------------------------------------------------------------


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks that may leak from Qwen3.5."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_json_robust(text: str) -> dict | None:
    """Try to parse JSON from model output with fallbacks.

    1. Direct parse
    2. Extract from ```json ... ``` code block
    3. Extract first {...} substring
    """
    text = strip_think_blocks(text)

    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "answer": str,
    "evidence": str,
    "related_entities": list,
    "answer_type": str,
}


def check_schema(parsed) -> bool:
    """Check that parsed dict has all required keys with correct types."""
    if not isinstance(parsed, dict):
        return False
    for key, expected_type in REQUIRED_KEYS.items():
        if key not in parsed:
            return False
        if not isinstance(parsed[key], expected_type):
            return False
    return True


# ---------------------------------------------------------------------------
# Text normalisation (for Answer EM / F1)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize text for comparison: NFKC, lower, strip whitespace/punctuation."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Remove common punctuation
    text = re.sub(
        r"[\s。、．，・「」『』（）()\[\]【】\-―ー〜~！!？?：:；;／/\\\"\']+", "", text
    )
    return text


# ---------------------------------------------------------------------------
# Answer EM / F1 (character-level)
# ---------------------------------------------------------------------------


def answer_em(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def answer_f1(pred: str, gold: str) -> float:
    """Character-level F1."""
    pred_chars = list(normalize_text(pred))
    gold_chars = list(normalize_text(gold))
    if not gold_chars:
        return float(not pred_chars)
    if not pred_chars:
        return 0.0
    common = sum((c in gold_chars) for c in pred_chars)
    # Use multiset intersection
    from collections import Counter

    pred_counter = Counter(pred_chars)
    gold_counter = Counter(gold_chars)
    common = sum((pred_counter & gold_counter).values())
    precision = common / len(pred_chars)
    recall = common / len(gold_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Evidence ROUGE-L (character-level tokenisation)
# ---------------------------------------------------------------------------


def _lcs_length(x: list, y: list) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimized LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def evidence_rouge_l(pred: str, gold: str) -> float:
    """Character-level ROUGE-L F1."""
    pred_chars = list(pred)
    gold_chars = list(gold)
    if not gold_chars:
        return float(not pred_chars)
    if not pred_chars:
        return 0.0
    lcs = _lcs_length(pred_chars, gold_chars)
    precision = lcs / len(pred_chars)
    recall = lcs / len(gold_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evidence_em(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


# ---------------------------------------------------------------------------
# Entity F1 — (name, type) pair matching
# ---------------------------------------------------------------------------


def entity_f1(pred_entities: list[dict], gold_entities: list[dict]) -> float:
    """F1 based on (name, type) pairs."""

    def to_pairs(entities):
        pairs = set()
        for e in entities:
            if isinstance(e, dict) and "name" in e and "type" in e:
                pairs.add(
                    (normalize_text(str(e["name"])), normalize_text(str(e["type"])))
                )
        return pairs

    pred_pairs = to_pairs(pred_entities)
    gold_pairs = to_pairs(gold_entities)

    if not gold_pairs and not pred_pairs:
        return 1.0
    if not gold_pairs or not pred_pairs:
        return 0.0

    common = len(pred_pairs & gold_pairs)
    precision = common / len(pred_pairs)
    recall = common / len(gold_pairs)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Answer Type Accuracy
# ---------------------------------------------------------------------------


def answer_type_accuracy(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


# ---------------------------------------------------------------------------
# Aggregate evaluation for a single sample
# ---------------------------------------------------------------------------


def evaluate_single(prediction_text: str, gold: dict) -> dict:
    """Evaluate a single prediction against gold.

    Args:
        prediction_text: Raw model output string
        gold: Dict with keys: answer, evidence, related_entities, answer_type

    Returns:
        Dict with all metric scores and parse status
    """
    result = {
        "json_parse": 0.0,
        "schema_valid": 0.0,
        "answer_em": 0.0,
        "answer_f1": 0.0,
        "evidence_em": 0.0,
        "evidence_rouge_l": 0.0,
        "entity_f1": 0.0,
        "answer_type_acc": 0.0,
    }

    parsed = parse_json_robust(prediction_text)
    if parsed is None:
        return result
    result["json_parse"] = 1.0

    if not check_schema(parsed):
        return result
    result["schema_valid"] = 1.0

    result["answer_em"] = answer_em(parsed["answer"], gold["answer"])
    result["answer_f1"] = answer_f1(parsed["answer"], gold["answer"])
    result["evidence_em"] = evidence_em(parsed["evidence"], gold["evidence"])
    result["evidence_rouge_l"] = evidence_rouge_l(parsed["evidence"], gold["evidence"])
    result["entity_f1"] = entity_f1(
        parsed["related_entities"], gold["related_entities"]
    )
    result["answer_type_acc"] = answer_type_accuracy(
        parsed["answer_type"], gold["answer_type"]
    )

    return result


def aggregate_metrics(results: list[dict]) -> dict:
    """Average metrics across samples."""
    if not results:
        return {}
    keys = results[0].keys()
    return {k: sum(r[k] for r in results) / len(results) for k in keys}
