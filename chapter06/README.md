# 第6章 ファインチューニングの実践 — サンプルコード

本ディレクトリには、書籍第6章（特に6.4節「ファインチューニングの実践」）で扱う、Qwen3.5-4B-Base に対するLoRA SFTの実行コード一式を収録しています。

- 学習対象モデル: `Qwen/Qwen3.5-4B-Base`
- 学習手法: LoRA SFT（bf16, Unsloth + TRL `SFTTrainer`）
- データセット: JSQuAD（学習・in-domain評価）/ JEMHopQA（out-of-domain評価）

## ディレクトリ構成

```
chapter06/
├── pyproject.toml                # 依存ライブラリ定義（uv管理）
├── configs/
│   └── sft_base.yaml             # SFT学習用ハイパーパラメータ
├── src/                          # 共通モジュール
│   ├── data_utils.py             # JSONL読み込み・Few-shot構築・サブサンプリング
│   ├── model_utils.py            # モデル/トークナイザ読み込み・ChatMLテンプレート・バッチ推論
│   └── metrics.py                # 6種の評価指標（JSON Parse / Schema / Answer EM・F1 / Evidence EM・ROUGE-L / Entity F1 / Answer Type Acc）
├── create_dataset/               # データセット構築パイプライン
│   ├── 01_jsquad_base.py         # JSQuAD取得・evidenceルールベース抽出
│   ├── 02_jemhopqa_context.py    # JEMHopQA取得・Wikipediaコンテキスト付与
│   ├── 03_annotate_with_llm.py   # related_entities / answer_type のLLMアノテーション
│   ├── 04_merge_and_split.py     # 統合・train/dev/test分割・ChatML変換
│   └── prompts/
│       └── annotate.txt          # アノテーション用プロンプトテンプレート
└── scripts/                      # 学習・評価スクリプト
    ├── 05_baseline_eval.py       # プロンプトのみのベースライン評価（0/1/3-shot）
    ├── 06_train_sft.py           # LoRA SFT学習
    └── 07_eval_sft.py            # SFT後モデルの評価とベースライン比較表生成
```

## 環境構築

uvでPython 3.12以上の環境を作成し、依存をインストールします。

```bash
cd chapter06
uv sync
source .venv/bin/activate
```

GPU環境はCUDA 12.4 / bf16対応のNVIDIA GPUを想定しています（H100 80GB が標準。RTX 4090 24GB程度でも動作可能）。

LLMアノテーション（`03_annotate_with_llm.py`）にはOpenAI互換のAPIを利用します。`create_dataset/.env` を用意し、利用するAPIのエンドポイントとAPIキーを設定してください。

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=...
```

## 各コードの概要と利用方法

実行は以下の順番が標準です。`data/` 配下に中間生成物・最終データセットが書き出されます。

### 1. データセット構築（`create_dataset/`）

#### `01_jsquad_base.py` — JSQuAD取得とevidence抽出

Hugging Face `sbintuitions/JSQuAD` を取得し、`context` の `[SEP]` を整形した上で、`answer_start` をもとに回答を含む文を抽出して `evidence` フィールドとして付与します。

```bash
uv run python create_dataset/01_jsquad_base.py
# → data/jsquad_base.jsonl
```

#### `02_jemhopqa_context.py` — JEMHopQAコンテキスト構築

Hugging Face `sbintuitions/JEMHopQA` を取得し、`page_ids` を元に日本語Wikipedia APIから対応するページ本文を取得・結合して `context` を構成します。WikipediaのレスポンスはローカルキャッシュされるためAPIへの再アクセスは不要です。

```bash
uv run python create_dataset/02_jemhopqa_context.py
# → data/jemhopqa_base.jsonl
```

#### `03_annotate_with_llm.py` — LLMアノテーション

`related_entities` と `answer_type` を外部LLM API経由で自動付与します。共通プロンプトは `prompts/annotate.txt` に定義されており、複数件を1リクエストにまとめて並列に投げる実装になっています。

```bash
uv run python create_dataset/03_annotate_with_llm.py --target jsquad
uv run python create_dataset/03_annotate_with_llm.py --target jemhopqa
# → data/jsquad_annotated.jsonl, data/jemhopqa_annotated.jsonl
```

#### `04_merge_and_split.py` — 統合・分割・ChatML変換

アノテーション失敗レコードを除外し、`train` / `dev` / `test_jsquad` / `test_jemhopqa` の4分割に整理した上で、SFTで利用するChatML形式（`messages` 配列）に変換します。

```bash
uv run python create_dataset/04_merge_and_split.py
# → data/train.jsonl / dev.jsonl / test_jsquad.jsonl / test_jemhopqa.jsonl
```

### 2. 共通モジュール（`src/`）

スクリプトから`import`されるユーティリティで、単体実行はしません。

- **`data_utils.py`**: JSONLの読み込み、Few-shot例の構築、`answer_type` で層化したサブサンプリング
- **`model_utils.py`**: Unsloth経由でのモデル/トークナイザ読み込み、ChatMLテンプレートの強制適用、バッチ推論
- **`metrics.py`**: 6種の評価指標を計算（JSONパース・スキーマ準拠・Answer EM/F1・Evidence EM/ROUGE-L・Entity F1・Answer Type Acc）

### 3. 学習・評価（`scripts/`）

#### `05_baseline_eval.py` — ベースライン評価

Qwen3.5-4B-Baseに対してプロンプトのみ（追加学習なし）で評価し、shot数を変えてFew-shotプロンプティングの効果を測定します。`test_jsquad` と `test_jemhopqa` の両方を一括で評価します。

```bash
uv run python scripts/05_baseline_eval.py \
    --model Qwen/Qwen3.5-4B-Base \
    --shots 0 1 3 \
    --test-sets test_jsquad test_jemhopqa \
    --output-dir results/baseline
# → results/baseline/{test_set}_{n}shot_predictions.jsonl, summary.json, summary.md
```

#### `06_train_sft.py` — LoRA SFT学習

`configs/sft_base.yaml` をロードし、TRL `SFTTrainer` + Unslothで学習を実行します。`--train-subset` で学習データの件数を絞ることもできます（学習曲線実験で利用）。

```bash
uv run python scripts/06_train_sft.py \
    --config configs/sft_base.yaml \
    --output-dir results/sft
# → results/sft/final_adapter/（LoRAアダプタ）
```

主なハイパーパラメータは `configs/sft_base.yaml` で管理しています（rank=64, alpha=128, lr=2e-4, epochs=3, bf16, target_modules=Q/K/V/O + FFN）。

#### `07_eval_sft.py` — SFT後モデルの評価

学習済みLoRAアダプタを読み込み、ベースモデルに適用した上で評価します。`05_baseline_eval.py` の結果が `--baseline-dir` に存在すれば、比較表（Markdown）を自動生成します。

```bash
uv run python scripts/07_eval_sft.py \
    --model Qwen/Qwen3.5-4B-Base \
    --adapter-path results/sft/final_adapter \
    --test-sets test_jsquad test_jemhopqa \
    --output-dir results/sft_eval \
    --baseline-dir results/baseline
# → results/sft_eval/sft_results.json, comparison.md
```
