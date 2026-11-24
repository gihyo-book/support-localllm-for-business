"""Step 6: LoRA SFT学習 with Unsloth + TRL SFTTrainer."""

import argparse
import sys
from pathlib import Path

import yaml
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_utils import load_dataset
from src.model_utils import (
    apply_chat_template,
    ensure_chat_template,
    load_model_for_training,
)


def tokenize_example(example, tokenizer, text_tokenizer, max_seq_length):
    """Apply chat template and tokenize with labels masking.

    Only compute loss on assistant response tokens (JSON content + <|im_end|>).
    System/user messages and template boilerplate are masked with -100.
    """
    # Full text: system + user + assistant (including response)
    text = apply_chat_template(
        tokenizer,
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = text_tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    # Prefix: system + user + generation prompt
    prefix_messages = [m for m in example["messages"] if m["role"] != "assistant"]
    prefix_text = apply_chat_template(
        tokenizer,
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_tokenized = text_tokenizer(
        prefix_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    prefix_len = len(prefix_tokenized["input_ids"])

    # Labels: -100 for prefix (no loss), real token ids for response + <|im_end|>
    labels = tokenized["input_ids"].copy()
    for i in range(min(prefix_len, len(labels))):
        labels[i] = -100

    tokenized["labels"] = labels
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="SFT Training with LoRA")
    parser.add_argument("--config", default="configs/sft_default.yaml")
    parser.add_argument("--output-dir", default="experiments/20250328/results/sft")
    parser.add_argument(
        "--train-subset",
        type=int,
        default=None,
        help="Use subset of training data (for learning curve)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint directory",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_seq_length = config["model"]["max_seq_length"]

    # Load model with LoRA
    print("Loading model with LoRA...")
    model, tokenizer = load_model_for_training(
        model_name=config["model"]["name"],
        max_seq_length=max_seq_length,
        lora_r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
    )

    ensure_chat_template(tokenizer)

    # Get text tokenizer (Qwen3.5 returns VL processor)
    text_tokenizer = (
        tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Load data
    print("Loading training data...")
    train_data = load_dataset("train")
    dev_data = load_dataset("dev")

    if args.train_subset:
        from src.data_utils import stratified_subsample

        train_data = stratified_subsample(train_data, args.train_subset)
        print(f"Using subset: {len(train_data)} training samples")

    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}")

    # Create HF datasets and tokenize
    train_ds = Dataset.from_list(train_data)
    dev_ds = Dataset.from_list(dev_data)

    train_ds = train_ds.map(
        lambda ex: tokenize_example(ex, tokenizer, text_tokenizer, max_seq_length),
        remove_columns=train_ds.column_names,
    )
    dev_ds = dev_ds.map(
        lambda ex: tokenize_example(ex, tokenizer, text_tokenizer, max_seq_length),
        remove_columns=dev_ds.column_names,
    )

    # Training arguments
    from transformers import TrainingArguments
    from trl import SFTTrainer

    tc = config["training"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        bf16=tc["bf16"],
        gradient_checkpointing=tc["gradient_checkpointing"],
        logging_steps=tc["logging_steps"],
        eval_strategy="steps",
        eval_steps=tc["eval_steps"],
        save_steps=tc["save_steps"],
        save_total_limit=tc["save_total_limit"],
        load_best_model_at_end=tc["load_best_model_at_end"],
        metric_for_best_model=tc["metric_for_best_model"],
        greater_is_better=tc["greater_is_better"],
        seed=tc["seed"],
        dataloader_num_workers=tc["dataloader_num_workers"],
        report_to="none",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=text_tokenizer,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        args=training_args,
    )

    # Train
    if args.resume_from_checkpoint:
        print(f"Resuming training from {args.resume_from_checkpoint}...")
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        print("Starting training...")
        train_result = trainer.train()

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save final adapter
    final_adapter_dir = output_dir / "final_adapter"
    print(f"Saving adapter to {final_adapter_dir}")
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    print("Training complete!")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Train runtime: {metrics.get('train_runtime', 'N/A'):.1f}s")


if __name__ == "__main__":
    main()
