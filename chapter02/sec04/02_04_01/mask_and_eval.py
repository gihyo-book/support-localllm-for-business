#!/usr/bin/env python3
import argparse
import json
import re
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from openai import OpenAI


@dataclass
class Config:
    types: List[str]
    redaction_tokens: Dict[str, str]
    normalize: Dict[str, bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masking + entity evaluation tool")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--config", required=True, help="masking_labels.yaml path")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--task-mode",
        choices=["detect_only", "detect_and_mask", "mask_only"],
        default="detect_and_mask",
        help=(
            "detect_only: entitiesのみ出力させる（検出のみ）/ "
            "detect_and_mask: entities + masked_text を両方出力させる（従来の挙動）/ "
            "mask_only: masked_text のみ出力させる（マスキングのみ）"
        ),
    )
    parser.add_argument(
        "--mask-strength",
        choices=["optional", "required"],
        default="required",
        help=(
            "detect_and_mask 時のmasked_text指示の強度。"
            "optional: 「含めてもよい」と任意指定 / "
            "required: 「必ず含める」と必須指定（デフォルト）。"
            "他のtask_modeでは無視される。"
        ),
    )
    # 旧引数 --with-masked-text は --task-mode への置き換え推奨だが互換のため残す
    parser.add_argument(
        "--with-masked-text",
        action="store_true",
        help="[非推奨] --task-mode detect_and_mask と同じ。--task-modeを使ってください",
    )
    return parser.parse_args()


def resolve_task_mode(args: argparse.Namespace) -> str:
    """旧引数との互換性を保ちつつtask_modeを決定する。"""
    if args.with_masked_text and args.task_mode == "detect_and_mask":
        # --with-masked-text 単独指定 → detect_and_mask（デフォルトと同じ）
        return "detect_and_mask"
    return args.task_mode


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    types = data.get("types", [])
    redaction_tokens = data.get("redaction_tokens", {})
    normalize = data.get("normalize", {})

    if not types:
        raise ValueError("config.types is required")
    for t in types:
        if t not in redaction_tokens:
            raise ValueError(f"redaction_tokens missing for type: {t}")

    return Config(types=types, redaction_tokens=redaction_tokens, normalize=normalize)


def build_text(row: Dict[str, Any]) -> str:
    subject = row.get("subject", "") or ""
    body = row.get("body", "") or ""
    signature = row.get("signature", "") or ""
    return f"{subject}\n\n{body}\n\n{signature}"


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_\-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_first_json_object(text: str) -> Optional[str]:
    cleaned = strip_code_fences(text)
    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(cleaned[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : idx + 1]

    return None


def normalize_value(value: str, typ: Optional[str], cfg: Config) -> str:
    out = value
    if cfg.normalize.get("normalize_hyphen", False):
        out = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u30fc\uff0d]", "-", out)
    if cfg.normalize.get("strip_whitespace", False):
        out = re.sub(r"\s+", "", out)
    if typ == "email" and cfg.normalize.get("casefold_email", False):
        out = out.casefold()
    return out


def sanitize_entities(raw_entities: Any, allowed_types: Set[str]) -> List[Dict[str, str]]:
    entities: List[Dict[str, str]] = []
    if not isinstance(raw_entities, list):
        return entities
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        typ = item.get("type")
        value = item.get("value")
        if not isinstance(typ, str) or not isinstance(value, str):
            continue
        typ = typ.strip()
        value = value.strip()
        if not typ or not value:
            continue
        if typ not in allowed_types:
            continue
        entities.append({"type": typ, "value": value})
    return entities


def as_set(entities: List[Dict[str, str]], cfg: Config) -> Set[Tuple[str, str]]:
    result: Set[Tuple[str, str]] = set()
    for ent in entities:
        result.add((ent["type"], normalize_value(ent["value"], ent["type"], cfg)))
    return result


def as_value_set(entities: List[Dict[str, str]], cfg: Config) -> Set[str]:
    return {normalize_value(ent["value"], ent["type"], cfg) for ent in entities}


def calc_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _normalize_items(
    entities: List[Dict[str, str]], cfg: Config
) -> List[Tuple[str, str]]:
    """(type, normalized_value) のリストを返す（重複除去なし）"""
    return [
        (ent["type"], normalize_value(ent["value"], ent["type"], cfg))
        for ent in entities
    ]


def _is_match(pred_val: str, gold_val: str) -> bool:
    """包含判定: どちらか一方が他方を含んでいればTrue"""
    return pred_val == gold_val or pred_val in gold_val or gold_val in pred_val


def compute_match(
    pred_entities: List[Dict[str, str]], gold_entities: List[Dict[str, str]], cfg: Config
) -> Dict[str, Any]:
    pred_items = _normalize_items(pred_entities, cfg)
    gold_items = _normalize_items(gold_entities, cfg)

    # --- strict (type + value 包含) ---
    gold_matched = [False] * len(gold_items)
    pred_matched = [False] * len(pred_items)

    tp_items = []
    for pi, (pt, pv) in enumerate(pred_items):
        for gi, (gt, gv) in enumerate(gold_items):
            if not gold_matched[gi] and pt == gt and _is_match(pv, gv):
                tp_items.append({"type": pt, "value": pv})
                pred_matched[pi] = True
                gold_matched[gi] = True
                break

    fp_items = [
        {"type": pt, "value": pv}
        for (pt, pv), matched in zip(pred_items, pred_matched)
        if not matched
    ]
    fn_items = [
        {"type": gt, "value": gv}
        for (gt, gv), matched in zip(gold_items, gold_matched)
        if not matched
    ]

    strict = {
        "tp_items": tp_items,
        "fp_items": fp_items,
        "fn_items": fn_items,
        **calc_prf(len(tp_items), len(fp_items), len(fn_items)),
    }

    # --- value_only (type無視 + value 包含) ---
    gold_val_matched = [False] * len(gold_items)
    pred_val_matched = [False] * len(pred_items)

    for pi, (_, pv) in enumerate(pred_items):
        for gi, (_, gv) in enumerate(gold_items):
            if not gold_val_matched[gi] and _is_match(pv, gv):
                pred_val_matched[pi] = True
                gold_val_matched[gi] = True
                break

    v_tp = sum(pred_val_matched)
    v_fp = sum(1 for m in pred_val_matched if not m)
    v_fn = sum(1 for m in gold_val_matched if not m)

    # --- mistyped: 値はマッチするがtypeが違う ---
    mistyped = []
    for pi, (pt, pv) in enumerate(pred_items):
        if pred_matched[pi]:
            continue
        gold_type_matches = [
            gt for (gt, gv) in gold_items if _is_match(pv, gv) and gt != pt
        ]
        if gold_type_matches:
            mistyped.append({
                "value": pv,
                "pred_type": pt,
                "gold_types": sorted(set(gold_type_matches)),
            })

    return {
        "strict": strict,
        "value_only": calc_prf(v_tp, v_fp, v_fn),
        "mistyped": mistyped,
    }


def apply_masking(
    text: str, entities: List[Dict[str, str]], redaction_tokens: Dict[str, str]
) -> Tuple[str, int]:
    masked = text
    count = 0

    uniq = {(ent["type"], ent["value"]) for ent in entities}
    ordered = sorted(uniq, key=lambda x: len(x[1]), reverse=True)

    for typ, value in ordered:
        token = redaction_tokens.get(typ, f"[{typ.upper()}]")
        occurrences = masked.count(value)
        if occurrences > 0:
            masked = masked.replace(value, token)
            count += occurrences

    return masked, count


def leak_check(masked_text: str, gold_entities: List[Dict[str, str]]) -> Dict[str, Any]:
    leaked = []
    for ent in gold_entities:
        v = ent.get("value", "")
        if v and v in masked_text:
            leaked.append({"type": ent.get("type"), "value": v})
    return {
        "leak": 1 if leaked else 0,
        "leaked_values": leaked,
    }


def build_system_prompt(
    types: List[str],
    task_mode: str,
    redaction_tokens: Dict[str, str],
    mask_strength: str = "required",
) -> str:
    """task_modeに応じたシステムプロンプトを生成する。

    mask_strength は task_mode == "detect_and_mask" のときのみ意味を持つ。
    - "optional": 「masked_text を含めてもよい」と任意指定
    - "required": 「masked_text を必ず含める」と必須指定
    """
    common_rules = (
        "あなたは個人情報マスキング抽出器です。\n"
        "以下を厳守してください。\n"
        "- 出力はJSONのみ。説明文やMarkdownは禁止。\n"
    )

    if task_mode == "detect_only":
        return (
            common_rules
            + f"- type は次の候補のみ: {', '.join(types)}\n"
            + "- entities.value は入力text内に実際に出現する文字列をそのまま返す。\n"
            + "- 推測で作らない。textに存在しない値は出力しない。\n"
            + "- JSONスキーマ: {\"id\": str, \"entities\": [{\"type\": str, \"value\": str}]}"
        )

    if task_mode == "mask_only":
        # マスキングのみ：entitiesを出力させず、masked_textのみ返させる
        token_pairs = ", ".join(f"{t}→{redaction_tokens[t]}" for t in types if t in redaction_tokens)
        return (
            common_rules
            + "- 入力textに含まれる個人情報（PII）を、種別に応じた固定トークンで置換したmasked_textを返す。\n"
            + f"- 置換ルール: {token_pairs}\n"
            + "- 入力textに存在しないPIIを推測で挿入してはいけない。\n"
            + "- JSONスキーマ: {\"id\": str, \"masked_text\": str}"
        )

    # task_mode == "detect_and_mask"
    token_pairs = ", ".join(f"{t}→{redaction_tokens[t]}" for t in types if t in redaction_tokens)
    if mask_strength == "optional":
        # 任意指定: 「含めてもよい」
        masked_clause = (
            "- masked_text を含めてもよい。含める場合は置換ルールに従う。\n"
            + f"- 置換ルール: {token_pairs}\n"
        )
        schema = '{"id": str, "entities": [{"type": str, "value": str}], "masked_text": str(任意)}'
    else:
        # 必須指定: 「必ず含める」
        masked_clause = (
            "- masked_text を必ず含める。entities に列挙した全 value を、対応するtypeのトークンで置換する。\n"
            + f"- 置換ルール: {token_pairs}\n"
        )
        schema = '{"id": str, "entities": [{"type": str, "value": str}], "masked_text": str}'

    return (
        common_rules
        + f"- type は次の候補のみ: {', '.join(types)}\n"
        + "- entities.value は入力text内に実際に出現する文字列をそのまま返す。\n"
        + "- 推測で作らない。textに存在しない値は出力しない。\n"
        + masked_clause
        + f"- JSONスキーマ: {schema}"
    )


def build_user_prompt(
    mail_id: str,
    text: str,
    types: List[str],
    task_mode: str,
    mask_strength: str = "required",
) -> str:
    if task_mode == "detect_only":
        schema_example = {
            "id": mail_id,
            "entities": [{"type": types[0], "value": "..."}],
        }
        return (
            f"types: {types}\n\n"
            f"input_id: {mail_id}\n"
            "input_text:\n"
            "-----\n"
            f"{text}\n"
            "-----\n\n"
            "次のスキーマでJSONのみ返答してください:\n"
            f"{json.dumps(schema_example, ensure_ascii=False, indent=2)}"
        )

    if task_mode == "mask_only":
        schema_example = {
            "id": mail_id,
            "masked_text": "...",
        }
        return (
            f"input_id: {mail_id}\n"
            "input_text:\n"
            "-----\n"
            f"{text}\n"
            "-----\n\n"
            "入力text内のPIIを置換ルールに従って置換し、masked_textを返してください。\n"
            "次のスキーマでJSONのみ返答してください:\n"
            f"{json.dumps(schema_example, ensure_ascii=False, indent=2)}"
        )

    # detect_and_mask
    if mask_strength == "optional":
        schema_example = {
            "id": mail_id,
            "entities": [{"type": types[0], "value": "..."}],
        }
        instruction = "entitiesでPIIを列挙してください。masked_textも返す場合は置換ルールに従ってください。\n"
    else:
        schema_example = {
            "id": mail_id,
            "entities": [{"type": types[0], "value": "..."}],
            "masked_text": "...",
        }
        instruction = "entitiesでPIIを列挙し、masked_textで置換ルールに従って置換してください。\n"

    return (
        f"types: {types}\n\n"
        f"input_id: {mail_id}\n"
        "input_text:\n"
        "-----\n"
        f"{text}\n"
        "-----\n\n"
        + instruction
        + "次のスキーマでJSONのみ返答してください:\n"
        f"{json.dumps(schema_example, ensure_ascii=False, indent=2)}"
    )


def call_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    timeout: Optional[float],
) -> Tuple[str, Dict[str, Any]]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    chat_api = client.with_options(timeout=timeout).chat.completions if timeout else client.chat.completions
    resp = chat_api.create(**kwargs)

    raw = resp.choices[0].message.content if resp.choices else ""
    usage = None
    response_model = None
    if hasattr(resp, "usage") and resp.usage is not None:
        usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
    if hasattr(resp, "model"):
        response_model = resp.model

    return raw or "", {"request_model": model, "response_model": response_model, "usage": usage}


def load_jsonl(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and i + 1 >= limit:
                break
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_misses_md(predictions: List[Dict[str, Any]]) -> str:
    lines = ["# Misses (FN中心)\n"]
    for p in predictions:
        mr = p.get("match_result", {})
        if not mr:
            continue
        strict = mr.get("strict", {})
        fn = strict.get("fn_items", [])
        fp = strict.get("fp_items", [])
        if not fn and not fp:
            continue

        subject = p.get("subject", "")
        raw = p.get("raw_model_output", "")
        raw_excerpt = raw[:400].replace("\n", " ")

        lines.append(f"## {p.get('id', 'unknown')}\n")
        lines.append(f"- subject: {subject}\n")
        lines.append(f"- FN: {json.dumps(fn, ensure_ascii=False)}\n")
        lines.append(f"- FP: {json.dumps(fp, ensure_ascii=False)}\n")
        lines.append(f"- mistyped: {json.dumps(mr.get('mistyped', []), ensure_ascii=False)}\n")
        lines.append(f"- raw_excerpt: {raw_excerpt}\n")

    if len(lines) == 1:
        lines.append("漏れ/過検出は見つかりませんでした。\n")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    task_mode = resolve_task_mode(args)

    input_path = Path(args.input)
    out_dir = Path(args.out)
    config_path = Path(args.config)

    ensure_dir(out_dir)
    cfg = load_config(config_path)
    rows = load_jsonl(input_path, args.limit)

    client_kwargs: dict[str, Any] = {}
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key
    client = OpenAI(**client_kwargs)

    predictions: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    overall_tp = overall_fp = overall_fn = 0
    value_tp = value_fp = value_fn = 0
    per_type_counts: Dict[str, Dict[str, int]] = {
        t: {"tp": 0, "fp": 0, "fn": 0} for t in cfg.types
    }

    leak_total = 0
    leak_hits = 0
    # detect_only時のハイブリッド構成（LLM検出 + コード置換）の集計
    hybrid_leak_total = 0
    hybrid_leak_hits = 0
    hybrid_replace_time_total = 0.0

    # task_modeごとに、entitiesの評価を行うかmasked_textの評価を行うかを決める
    needs_entity_eval = task_mode in ("detect_only", "detect_and_mask")
    needs_masked_eval = task_mode in ("mask_only", "detect_and_mask")
    # detect_onlyのときは、検出結果に対してコード置換でハイブリッド評価も行う
    eval_hybrid = task_mode == "detect_only"

    for row in rows:
        mail_start = time.perf_counter()
        mail_id = row.get("id", "")
        subject = row.get("subject", "")
        text = build_text(row)
        gold_entities = sanitize_entities(row.get("masking_candidates", []), set(cfg.types))

        status = "ok"
        raw_model_output = ""
        extracted_json_text = ""
        parsed: Dict[str, Any] = {}
        model_info: Dict[str, Any] = {}

        system_prompt = build_system_prompt(cfg.types, task_mode, cfg.redaction_tokens, args.mask_strength)
        user_prompt = build_user_prompt(mail_id, text, cfg.types, task_mode, args.mask_strength)

        for attempt in range(args.max_retries + 1):
            try:
                raw_model_output, model_info = call_model(
                    client=client,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )

                extracted_json_text = extract_first_json_object(raw_model_output) or ""
                if not extracted_json_text:
                    raise ValueError("JSON object not found")

                parsed = json.loads(extracted_json_text)
                break
            except Exception as e:  # noqa: BLE001
                if attempt < args.max_retries:
                    user_prompt = (
                        "前回出力はJSONとして解釈できませんでした。"
                        "説明を一切含めず、有効なJSONオブジェクトのみを再出力してください。\n\n"
                        f"前回出力:\n{raw_model_output}"
                    )
                    continue

                status = "failed"
                errors.append(
                    {
                        "id": mail_id,
                        "stage": "model_or_parse",
                        "error": str(e),
                        "attempts": args.max_retries + 1,
                        "raw_model_output": raw_model_output,
                    }
                )

        # entitiesの取り出し（detect_only, detect_and_mask のときのみ意味を持つ）
        pred_entities: List[Dict[str, str]] = []
        if needs_entity_eval and status == "ok":
            pred_entities = sanitize_entities(parsed.get("entities", []), set(cfg.types))

        # masked_textの取り出し
        masked_text = None
        masking_meta = None
        if needs_masked_eval:
            if status == "ok":
                masked_text_from_model = parsed.get("masked_text") if isinstance(parsed, dict) else None
                if isinstance(masked_text_from_model, str) and masked_text_from_model:
                    masked_text = masked_text_from_model
                    replacement_count = sum(
                        masked_text.count(tok) for tok in cfg.redaction_tokens.values()
                    )
                else:
                    # mask_only でmasked_textが取れなかったとき、
                    # detect_and_mask でmasked_textが空だったときのフォールバック
                    if pred_entities:
                        masked_text, replacement_count = apply_masking(
                            text, pred_entities, cfg.redaction_tokens
                        )
                    else:
                        masked_text = ""
                        replacement_count = 0
            else:
                masked_text = ""
                replacement_count = 0

            leak = leak_check(masked_text, gold_entities)
            leak_total += 1
            leak_hits += leak["leak"]
            overmask_warning = (
                replacement_count > max(10, len(pred_entities) * 3)
                if pred_entities
                else False
            )
            masking_meta = {
                "replacement_count": replacement_count,
                "overmask_warning": overmask_warning,
                **leak,
            }

        # 検出評価（needs_entity_evalのときのみ）
        match_result: Optional[Dict[str, Any]] = None
        if needs_entity_eval:
            match_result = compute_match(pred_entities, gold_entities, cfg)

            strict = match_result["strict"]
            overall_tp += strict["tp"]
            overall_fp += strict["fp"]
            overall_fn += strict["fn"]

            vo = match_result["value_only"]
            value_tp += vo["tp"]
            value_fp += vo["fp"]
            value_fn += vo["fn"]

            for item in strict["tp_items"]:
                t = item["type"]
                if t in per_type_counts:
                    per_type_counts[t]["tp"] += 1
            for item in strict["fp_items"]:
                t = item["type"]
                if t in per_type_counts:
                    per_type_counts[t]["fp"] += 1
            for item in strict["fn_items"]:
                t = item["type"]
                if t in per_type_counts:
                    per_type_counts[t]["fn"] += 1

        # ハイブリッド評価（detect_onlyのときのみ）：
        # 検出結果に対してコード側で置換を行い、leak_rateを計算する
        hybrid_meta = None
        if eval_hybrid:
            replace_start = time.perf_counter()
            if pred_entities:
                hybrid_masked_text, hybrid_replacement_count = apply_masking(
                    text, pred_entities, cfg.redaction_tokens
                )
            else:
                hybrid_masked_text = text
                hybrid_replacement_count = 0
            hybrid_replace_elapsed = time.perf_counter() - replace_start
            hybrid_replace_time_total += hybrid_replace_elapsed

            hybrid_leak = leak_check(hybrid_masked_text, gold_entities)
            hybrid_leak_total += 1
            hybrid_leak_hits += hybrid_leak["leak"]

            hybrid_meta = {
                "masked_text": hybrid_masked_text,
                "replacement_count": hybrid_replacement_count,
                "replace_elapsed_sec": round(hybrid_replace_elapsed, 6),
                **hybrid_leak,
            }

        prediction_row: Dict[str, Any] = {
            "id": mail_id,
            "subject": subject,
            "task_mode": task_mode,
            "prediction": {
                "entities": pred_entities if needs_entity_eval else None,
                "masked_text": masked_text,
            },
            "gold": {
                "masking_candidates": gold_entities,
            },
            "raw_model_output": raw_model_output,
            "extracted_json_text": extracted_json_text,
            "status": status,
            "elapsed_sec": round(time.perf_counter() - mail_start, 3),
            "model_info": model_info,
        }
        if match_result is not None:
            prediction_row["match_result"] = match_result
        if masking_meta is not None:
            prediction_row["masking_check"] = masking_meta
        if hybrid_meta is not None:
            prediction_row["hybrid_check"] = hybrid_meta
        predictions.append(prediction_row)

    total_elapsed = round(time.perf_counter() - total_start, 3)

    evaluation: Dict[str, Any] = {
        "task_mode": task_mode,
        "mask_strength": args.mask_strength if task_mode == "detect_and_mask" else None,
        "n_total": len(rows),
        "n_valid": sum(1 for p in predictions if p["status"] == "ok"),
        "n_failed": sum(1 for p in predictions if p["status"] != "ok"),
        "elapsed_sec": total_elapsed,
    }

    if needs_entity_eval:
        per_type_metrics = {
            t: calc_prf(c["tp"], c["fp"], c["fn"]) for t, c in per_type_counts.items()
        }
        evaluation["overall_micro"] = calc_prf(overall_tp, overall_fp, overall_fn)
        evaluation["value_only_micro"] = calc_prf(value_tp, value_fp, value_fn)
        evaluation["per_type"] = per_type_metrics

    if needs_masked_eval:
        evaluation["leak_rate"] = (leak_hits / leak_total) if leak_total else 0.0

    if eval_hybrid:
        # detect_only時のハイブリッド構成（LLM検出 + コード置換）の集計
        evaluation["hybrid"] = {
            "leak_rate": (hybrid_leak_hits / hybrid_leak_total) if hybrid_leak_total else 0.0,
            "code_replace_total_sec": round(hybrid_replace_time_total, 6),
            "total_sec_estimate": round(total_elapsed + hybrid_replace_time_total, 3),
        }

    write_jsonl(out_dir / "predictions.jsonl", predictions)
    write_jsonl(out_dir / "errors.jsonl", errors)
    with (out_dir / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    with (out_dir / "misses.md").open("w", encoding="utf-8") as f:
        f.write(build_misses_md(predictions))


if __name__ == "__main__":
    main()