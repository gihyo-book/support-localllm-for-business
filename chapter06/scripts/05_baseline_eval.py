"""Step 5: ベースライン評価 — Zero-shot / 1-shot / 3-shot"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_utils import (
    build_few_shot_messages,
    build_inference_messages,
    extract_ground_truth,
    load_dataset,
)
from src.metrics import aggregate_metrics, evaluate_single
from src.model_utils import generate_batch, load_model_for_inference


def run_evaluation(
    model,
    tokenizer,
    test_data: list[dict],
    few_shot_msgs: list[dict],
    batch_size: int = 1,
    max_new_tokens: int = 512,
    max_input_tokens: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run evaluation and return (predictions, metric_results)."""
    # Build all inference messages
    all_messages = []
    all_golds = []
    for rec in test_data:
        msgs = build_inference_messages(rec, few_shot_msgs)
        gold = extract_ground_truth(rec)
        all_messages.append(msgs)
        all_golds.append(gold)

    # Generate in batches
    print(f"  Generating {len(all_messages)} predictions (batch_size={batch_size})...")
    all_outputs = generate_batch(
        model,
        tokenizer,
        all_messages,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
    )

    # Evaluate
    predictions = []
    results = []
    for i, (output, gold) in enumerate(zip(all_outputs, all_golds)):
        metrics = evaluate_single(output, gold)
        results.append(metrics)
        predictions.append(
            {
                "index": i,
                "prediction": output,
                "gold": gold,
                "metrics": metrics,
            }
        )

    return predictions, results


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 1, 3, 5])
    parser.add_argument(
        "--test-sets", nargs="+", default=["test_jsquad", "test_jemhopqa"]
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=262144,
        help="モデルの最大シーケンス長",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="バッチサイズ。長文コンテキストでのOOM回避のため既定1",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=110000,
        help="推論時の総トークン数上限。超過時は context 末尾を切り詰め",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default="results/baseline")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_for_inference(
        args.model, max_seq_length=args.max_seq_length
    )

    # Load train data for few-shot examples
    train_data = load_dataset("train")
    print(f"Train data loaded: {len(train_data)} records")

    # Summary table
    summary = {}

    for test_set in args.test_sets:
        print(f"\n{'=' * 60}")
        print(f"Test set: {test_set}")
        print(f"{'=' * 60}")

        test_data = load_dataset(test_set)
        print(f"Test data loaded: {len(test_data)} records")

        for n_shots in args.shots:
            print(f"\n--- {n_shots}-shot ---")
            t0 = time.time()

            few_shot_msgs = build_few_shot_messages(train_data, n_shots)

            predictions, results = run_evaluation(
                model,
                tokenizer,
                test_data,
                few_shot_msgs,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
            )

            metrics = aggregate_metrics(results)
            elapsed = time.time() - t0

            # Print metrics
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            print(f"  Time: {elapsed:.1f}s")

            # Save predictions
            pred_path = output_dir / f"{test_set}_{n_shots}shot_predictions.jsonl"
            with open(pred_path, "w", encoding="utf-8") as f:
                for p in predictions:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

            summary[f"{test_set}/{n_shots}-shot"] = {**metrics, "time_seconds": elapsed}

    # Write summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Write markdown summary
    md_path = output_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        for test_set in args.test_sets:
            f.write(f"## {test_set}\n\n")
            f.write(
                "| Metric | " + " | ".join(f"{n}-shot" for n in args.shots) + " |\n"
            )
            f.write("|--------|" + "|".join("--------" for _ in args.shots) + "|\n")

            metric_names = [
                "json_parse",
                "schema_valid",
                "answer_em",
                "answer_f1",
                "evidence_em",
                "evidence_rouge_l",
                "entity_f1",
                "answer_type_acc",
            ]
            for metric_name in metric_names:
                row = f"| {metric_name} |"
                for n in args.shots:
                    key = f"{test_set}/{n}-shot"
                    val = summary.get(key, {}).get(metric_name, 0)
                    row += f" {val:.4f} |"
                f.write(row + "\n")
            f.write("\n")

    print(f"\nSummary saved to {summary_path} and {md_path}")


if __name__ == "__main__":
    main()
