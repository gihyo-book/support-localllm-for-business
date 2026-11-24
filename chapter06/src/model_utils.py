"""モデル読み込み・推論ユーティリティ (Unsloth + Qwen3.5)."""

import re

import torch

MODEL_NAME = "Qwen/Qwen3.5-4B"

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)


def load_model_for_inference(
    model_name: str = MODEL_NAME,
    adapter_path: str | None = None,
    max_seq_length: int = 4096,
):
    """Load model and tokenizer using Unsloth for inference.

    Returns (model, tokenizer).
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    if _is_base_model(tokenizer):
        ensure_chat_template(tokenizer)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_model_for_training(
    model_name: str = MODEL_NAME,
    max_seq_length: int = 2048,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
):
    """Load model with LoRA for training using Unsloth.

    Returns (model, tokenizer).
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _get_text_tokenizer(tokenizer):
    """Get the underlying text tokenizer from a processor or tokenizer."""
    # Qwen3.5 may return a VL processor; extract the text tokenizer
    if hasattr(tokenizer, "tokenizer"):
        return tokenizer.tokenizer
    return tokenizer


def _is_base_model(tokenizer):
    """Check if the tokenizer belongs to a base (non-instruct) model."""
    text_tok = _get_text_tokenizer(tokenizer)
    template = getattr(text_tok, "chat_template", None) or ""
    return "enable_thinking" not in template


def ensure_chat_template(tokenizer):
    """Force-set ChatML template to avoid Qwen3.5's built-in <think> blocks."""
    text_tok = _get_text_tokenizer(tokenizer)
    text_tok.chat_template = CHATML_TEMPLATE
    if tokenizer is not text_tok:
        tokenizer.chat_template = CHATML_TEMPLATE


def apply_chat_template(tokenizer, messages, **kwargs):
    """Apply chat template."""
    return tokenizer.apply_chat_template(messages, **kwargs)


def _get_max_position_embeddings(config) -> int:
    """Extract max_position_embeddings from model config, handling VL nesting."""
    if hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    if hasattr(config, "text_config") and hasattr(
        config.text_config, "max_position_embeddings"
    ):
        return config.text_config.max_position_embeddings
    return 4096


QUESTION_MARKER = "\n\n質問:"


def _truncate_context(msgs, text_tokenizer, tokenizer, max_input_tokens):
    """Truncate context in user message to fit within max_input_tokens."""
    import copy

    template_kwargs = {}
    if not _is_base_model(tokenizer):
        template_kwargs["enable_thinking"] = False

    user_idx = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
    if user_idx is None:
        return msgs

    user_content = msgs[user_idx]["content"]
    marker_pos = user_content.rfind(QUESTION_MARKER)
    if marker_pos == -1:
        return msgs

    context_part = user_content[:marker_pos]
    question_part = user_content[marker_pos:]

    msgs_no_context = copy.deepcopy(msgs)
    msgs_no_context[user_idx]["content"] = question_part
    shell_text = apply_chat_template(
        tokenizer,
        msgs_no_context,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    shell_tokens = len(text_tokenizer.encode(shell_text))

    context_budget = max_input_tokens - shell_tokens
    context_token_ids = text_tokenizer.encode(context_part)
    if len(context_token_ids) <= context_budget:
        return msgs

    truncated_context = text_tokenizer.decode(
        context_token_ids[:context_budget], skip_special_tokens=True
    )
    truncated_msgs = copy.deepcopy(msgs)
    truncated_msgs[user_idx]["content"] = truncated_context + question_part
    return truncated_msgs


def generate_batch(
    model,
    tokenizer,
    messages_list: list[list[dict]],
    max_new_tokens: int = 512,
    batch_size: int = 8,
    max_input_tokens: int | None = None,
) -> list[str]:
    """Generate responses for a batch of message lists.

    Args:
        model: The language model
        tokenizer: Tokenizer or Processor
        messages_list: List of conversation message lists
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for generation
        max_input_tokens: If set, truncate context in user messages to fit

    Returns:
        List of generated text strings (assistant responses only)
    """
    all_outputs = []
    text_tokenizer = _get_text_tokenizer(tokenizer)

    # Ensure padding token
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    n_batches = (len(messages_list) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(messages_list), batch_size)):
        batch_msgs = messages_list[i : i + batch_size]
        print(
            f"  Batch {batch_idx + 1}/{n_batches} ({i + len(batch_msgs)}/{len(messages_list)})",
            flush=True,
        )

        # Truncate context if max_input_tokens is set
        if max_input_tokens is not None:
            batch_msgs = [
                _truncate_context(msgs, text_tokenizer, tokenizer, max_input_tokens)
                for msgs in batch_msgs
            ]

        # Tokenize with chat template
        template_kwargs = {}
        if not _is_base_model(tokenizer):
            template_kwargs["enable_thinking"] = False
        texts = []
        for msgs in batch_msgs:
            text = apply_chat_template(
                tokenizer,
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            texts.append(text)

        # Tokenize with left padding for batch generation (use text tokenizer)
        text_tokenizer.padding_side = "left"
        max_len = _get_max_position_embeddings(model.config)
        inputs = text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(model.device)

        im_end_id = text_tokenizer.convert_tokens_to_ids("<|im_end|>")
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        }
        if isinstance(im_end_id, int) and im_end_id != text_tokenizer.unk_token_id:
            gen_kwargs["eos_token_id"] = im_end_id

        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            # Decode only the generated part (skip full input including padding)
            input_len = inputs["input_ids"].shape[1]
            for j, output in enumerate(outputs):
                generated = output[input_len:]
                text = text_tokenizer.decode(generated, skip_special_tokens=True)
                text = strip_think_blocks(text)
                all_outputs.append(text)
        except torch.OutOfMemoryError:
            seq_len = inputs["input_ids"].shape[1]
            print(
                f"  OOM at batch {batch_idx + 1} (seq_len={seq_len}), skipping",
                flush=True,
            )
            for _ in batch_msgs:
                all_outputs.append("")
            torch.cuda.empty_cache()

    return all_outputs
