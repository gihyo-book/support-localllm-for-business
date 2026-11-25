#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPT_VERSION = "v1.0"
SYSTEM_PROMPT = (
    "あなたはメール分類器です。\n"
    "出力はJSONのみで返してください。説明文、補足、コードブロックは禁止です。\n"
    "必ず次のキーのみを含むJSONオブジェクトを返してください: topic\n"
    "値は必ず指定されたラベル一覧の中から選んでください。"
)

JSON_FIX_PROMPT = "出力はJSONのみです。前回の出力をJSONとして修正してください。"


@dataclass
class LabelsConfig:
    labels: dict[str, list[str]]
    normalize_map: dict[str, dict[str, str]]


class ModelRunner:
    def __init__(self, model_name: str, temperature: float, max_tokens: int | None, timeout: int | None):
        from openai import OpenAI

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        client_kwargs: dict[str, Any] = {}
        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("OPENAI_API_KEY", "ollama")
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        self.client = OpenAI(**client_kwargs)

    def generate(self, prompt: str) -> tuple[str, dict]:
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            request_kwargs["timeout"] = self.timeout

        print(f"Requesting with kwargs: {request_kwargs}")

        response = self.client.chat.completions.create(**request_kwargs)
        raw_text = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if (hasattr(response, "usage") and hasattr(response.usage, "model_dump")) else None
        model_info = {
            "model": getattr(response, "model", self.model_name),
            "usage": usage,
        }
        return raw_text, model_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local LLM mail classification + evaluation")
    parser.add_argument("--input", required=True, help="Input mails jsonl")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--config", required=True, help="labels yaml")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_config(path: Path) -> LabelsConfig:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    labels = raw.get("labels") or {}
    normalize_map = raw.get("normalize_map") or {}
    return LabelsConfig(labels=labels, normalize_map=normalize_map)


def build_user_prompt(mail: dict[str, Any], labels: dict[str, list[str]]) -> str:
    labels_text = json.dumps({"topic": labels["topic"]}, ensure_ascii=False, indent=2)
    subject = mail.get("subject", "")
    body = mail.get("body", "")
    example = '{"topic":"契約関連"}'
    return (
        "次のメールを分類してください。\n"
        "ラベル一覧は以下のとおりです。必ずこの中から選択してください。\n"
        f"{labels_text}\n\n"
        "出力形式は厳密に次のJSONのみ:\n"
        f"{example}\n\n"
        "入力メール:\n"
        f"subject: {subject}\n"
        f"body: {body}\n"
    )


def extract_json_text(raw_text: str) -> str | None:
    text = (raw_text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                return part
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def normalize_label(field: str, value: Any, normalize_map: dict[str, dict[str, str]]) -> Any:
    if not isinstance(value, str):
        return value
    field_map = normalize_map.get(field, {})
    return field_map.get(value, value)


def parse_and_validate(raw_text: str, labels: dict[str, list[str]], normalize_map: dict[str, dict[str, str]]) -> tuple[dict[str, str] | None, str | None]:
    json_text = extract_json_text(raw_text)
    if not json_text:
        return None, "json_not_found"
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        return None, "json_decode_error"

    required_keys = ["topic"]  # topic のみ
    if set(obj.keys()) != set(required_keys):
        return None, "invalid_keys"

    out: dict[str, str] = {}
    for k in required_keys:
        v = normalize_label(k, obj.get(k), normalize_map)
        if not isinstance(v, str):
            return None, f"invalid_type_{k}"
        if v not in labels.get(k, []):
            return None, f"invalid_label_{k}:{v}"
        out[k] = v
    return out, None

def classify_with_repair(
    runner: ModelRunner,
    first_prompt: str,
    labels: dict[str, list[str]],
    normalize_map: dict[str, dict[str, str]],
    max_retries: int,
) -> tuple[dict[str, str] | None, str, dict, str | None]:
    prompt = first_prompt
    last_raw = ""
    last_model_info: dict[str, Any] = {}
    last_error = None

    for attempt in range(max_retries + 1):
        raw, model_info = runner.generate(prompt)
        last_raw = raw
        last_model_info = model_info

        parsed, err = parse_and_validate(raw, labels, normalize_map)
        if parsed is not None:
            return parsed, last_raw, last_model_info, None

        last_error = err
        if attempt < max_retries:
            prompt = (
                f"{JSON_FIX_PROMPT}\n\n"
                f"前回出力:\n{raw}\n\n"
                "正しいJSONのみを返してください。"
            )

    return None, last_raw, last_model_info, last_error


def calc_metrics(golds: list[str], preds: list[str], label_set: list[str]) -> dict[str, Any]:
    assert len(golds) == len(preds)
    n = len(golds)
    correct = sum(1 for g, p in zip(golds, preds) if g == p)
    accuracy = (correct / n) if n else 0.0

    per_label = {}
    f1_sum = 0.0
    for label in label_set:
        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_sum += f1
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(1 for g in golds if g == label),
        }

    macro_f1 = (f1_sum / len(label_set)) if label_set else 0.0
    return {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "n": n,
        "per_label": per_label,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out)
    config_path = Path(args.config)

    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    rows = load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    runner = ModelRunner(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    pred_path = out_dir / "predictions.jsonl"
    err_path = out_dir / "errors.jsonl"
    eval_path = out_dir / "evaluation.json"

    valid_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    with pred_path.open("w", encoding="utf-8") as pred_f, err_path.open("w", encoding="utf-8") as err_f:
        for row in rows:
            mail_id = row.get("id")
            ts = dt.datetime.now(dt.timezone.utc).isoformat()
            prompt = build_user_prompt(row, config.labels)
            mail_start = time.perf_counter()

            pred, raw_out, model_info, err = classify_with_repair(
                runner=runner,
                first_prompt=prompt,
                labels=config.labels,
                normalize_map=config.normalize_map,
                max_retries=args.max_retries,
            )

            
            gold = {
                "topic": normalize_label("topic", row.get("plan", {}).get("topic"), config.normalize_map),
                # intent・urgency はデータに残るが評価対象外
            }

            rec = {
                "id": mail_id,
                "model": args.model,
                "temperature": args.temperature,
                "prompt_version": PROMPT_VERSION,
                "timestamp": ts,
                "elapsed_sec": round(time.perf_counter() - mail_start, 3),
                "raw_model_output": raw_out,
                "prediction": pred,
                "gold": gold,
                "status": "valid" if pred is not None else "failed",
                "model_info": model_info,
            }
            pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if pred is None:
                err_rec = {
                    "id": mail_id,
                    "error": err,
                    "raw_model_output": raw_out,
                    "timestamp": ts,
                    "model_info": model_info,
                }
                err_f.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                failed_records.append(rec)
            else:
                valid_records.append(rec)

    
    metrics: dict[str, Any] = {}
    for field in ["topic"]:  # topic のみ評価
        golds = [r["gold"][field] for r in valid_records if r.get("prediction")]
        preds = [r["prediction"][field] for r in valid_records if r.get("prediction")]
        metrics[field] = calc_metrics(golds, preds, config.labels.get(field, []))

    total_elapsed = round(time.perf_counter() - total_start, 3)

    evaluation = {
        "model": args.model,
        "temperature": args.temperature,
        "prompt_version": PROMPT_VERSION,
        "n_total": len(rows),
        "n_valid": len(valid_records),
        "n_failed": len(failed_records),
        "elapsed_sec": total_elapsed,
        "metrics": metrics,
    }

    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    print(json.dumps({"out": str(out_dir), "n_total": len(rows), "n_valid": len(valid_records), "n_failed": len(failed_records), "elapsed_sec": total_elapsed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
