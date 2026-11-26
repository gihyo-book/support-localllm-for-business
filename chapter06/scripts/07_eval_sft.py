"""Step 7: SFT後の評価 — ベースラインとの比較表生成"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_utils import (
    build_inference_messages,
    extract_ground_truth,
    load_dataset,
)
from src.metrics import aggregate_metrics, evaluate_single
from src.model_utils import generate_batch, load_model_for_inference


def main():
    parser = argparse.ArgumentParser(description="SFT Model Evaluation")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-path", default="results/sft/final_adapter")
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
    parser.add_argument("--output-dir", default="results/sft_eval")
    parser.add_argument("--baseline-dir", default="results/baseline")
    parser.add_argument(
        "--baseline-shots",
        nargs="+",
        type=int,
        default=[0, 1, 3],
        help="比較表に並べるベースラインの shot 数",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SFT model
    print(f"Loading SFT model: {args.model} + {args.adapter_path}")
    model, tokenizer = load_model_for_inference(
        args.model,
        adapter_path=args.adapter_path,
        max_seq_length=args.max_seq_length,
    )

    sft_results = {}

    for test_set in args.test_sets:
        print(f"\n{'=' * 60}")
        print(f"Test set: {test_set}")
        print(f"{'=' * 60}")

        test_data = load_dataset(test_set)
        print(f"Test data: {len(test_data)} records")

        t0 = time.time()

        # Build messages (no few-shot for SFT)
        all_messages = []
        all_golds = []
        for rec in test_data:
            msgs = build_inference_messages(rec)
            gold = extract_ground_truth(rec)
            all_messages.append(msgs)
            all_golds.append(gold)

        # Generate
        print(f"  Generating {len(all_messages)} predictions...")
        all_outputs = generate_batch(
            model,
            tokenizer,
            all_messages,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
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

        metrics = aggregate_metrics(results)
        elapsed = time.time() - t0

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # Save predictions
        pred_path = output_dir / f"{test_set}_sft_predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        sft_results[test_set] = {**metrics, "time_seconds": elapsed}

    # Save SFT results
    with open(output_dir / "sft_results.json", "w", encoding="utf-8") as f:
        json.dump(sft_results, f, ensure_ascii=False, indent=2)

    # Generate comparison table
    baseline_summary_path = Path(args.baseline_dir) / "summary.json"
    if baseline_summary_path.exists():
        with open(baseline_summary_path) as f:
            baseline_summary = json.load(f)

        md_path = output_dir / "comparison.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# SFT vs Baseline Comparison\n\n")

            shots = args.baseline_shots
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

            for test_set in args.test_sets:
                f.write(f"## {test_set}\n\n")
                header = (
                    "| Metric | "
                    + " | ".join(f"{n}-shot" for n in shots)
                    + " | SFT |\n"
                )
                sep = (
                    "|--------|" + "|".join("--------" for _ in shots) + "|--------|\n"
                )
                f.write(header)
                f.write(sep)

                for metric_name in metric_names:
                    row = f"| {metric_name} |"
                    for n in shots:
                        key = f"{test_set}/{n}-shot"
                        val = baseline_summary.get(key, {}).get(metric_name, 0)
                        row += f" {val:.4f} |"
                    sft_val = sft_results.get(test_set, {}).get(metric_name, 0)
                    row += f" **{sft_val:.4f}** |"
                    f.write(row + "\n")
                f.write("\n")

        print(f"\nComparison table saved to {md_path}")
    else:
        print(
            f"\nBaseline summary not found at {baseline_summary_path}, skipping comparison"
        )


if __name__ == "__main__":
    main()
