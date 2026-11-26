# 2.4 ローカルLLMで業務データを扱うハンズオン

## 実験環境

| 項目 | 内容 |
|---|---|
| マシン | MacBook Pro 14インチ（2024年11月） |
| チップ | Apple M4 Pro |
| メモリ | 48GB |
| OS | macOS Tahoe |
| Ollama | 0.18.0 |
| Python | 3.10 以降 |

## セットアップ

以下のコマンドは、すべてこの `sec04` ディレクトリで実行してください。

```bash
pip install -r requirements.txt
```

[uv](https://docs.astral.sh/uv/) をお使いの方は、代わりに以下を実行してください。以降のコマンドは、`python` を `uv run python` に読み替えてください。

```bash
uv venv
uv pip install -r requirements.txt
```

Ollamaが起動していることを確認してください。macOS / Windows はアプリを起動していればOKです。Linuxなどでサービスが動いていない場合は、次のコマンドで起動します。

```bash
ollama serve
```

本書のハンズオンで使うモデルをあらかじめダウンロードしておきます。

```bash
ollama pull qwen3:4b-instruct
ollama pull gemma3:12b
ollama pull gpt-oss:20b  # メモリ32GB以上の環境を推奨
```

実行結果の出力先として、`runs` ディレクトリを作成しておきます。

```bash
mkdir -p runs
```

## ファイル構成

```
sec04/
├── 02_04_01/
│   ├── classify_and_eval.py    # 2.4.1 メール分類・評価
│   └── mask_and_eval.py        # 2.4.1 PIIマスキング・評価
├── 02_04_02/
│   └── generate_reply.py       # 2.4.2 返信生成（baseline）
├── 02_04_03/
│   └── generate_reply.py       # 2.4.3 返信生成（structured_prompt / two_step）
├── 02_04_04/
│   ├── generate_mail_full.py   # 2.4.4 ダミーデータ生成（LLMに全部任せる版）
│   └── generate_mail_hybrid.py # 2.4.4 ダミーデータ生成（hybrid版）
├── judge/
│   └── judge_replies.py        # 2.4.2〜2.4.3 LLM-as-a-Judge
├── config/
│   └── labels.yaml             # 分類・マスキングのラベル設定
└── data/
    └── 20b_mails.jsonl         # サンプルデータ（ダミーメール10件）
```

---

## 2.4.1 分類・マスキング

### メール分類

```bash
python 02_04_01/classify_and_eval.py \
  --input data/20b_mails.jsonl \
  --out runs/classify_qwen3 \
  --config config/labels.yaml \
  --model qwen3:4b-instruct
```

結果は `runs/classify_qwen3/` に出力されます。

| ファイル | 内容 |
|---|---|
| `predictions.jsonl` | メールごとの予測結果 |
| `errors.jsonl` | パース失敗したレコード |
| `evaluation.json` | 正解率・F1などの集計 |

### PIIマスキング

メール本文から個人情報（PII）を検出・マスキングする実験です。`--task-mode` でLLMに何を依頼するかを切り替えられます。

```bash
# 本文の実験（検出+置換、必須指定）
python 02_04_01/mask_and_eval.py \
  --input data/20b_mails.jsonl \
  --out runs/mask_qwen3 \
  --config config/labels.yaml \
  --model qwen3:4b-instruct \
  --task-mode detect_and_mask \
  --mask-strength required
```

#### task-mode の選び方

| モード | LLMに依頼する内容 | 出力 |
|---|---|---|
| `detect_and_mask` | PII検出と置換の両方（デフォルト） | `entities` と `masked_text` |
| `detect_only` | PII検出のみ。置換はコード側で実施 | `entities`（コード置換結果も評価） |
| `mask_only` | 置換のみ。検出結果は出力しない | `masked_text` |

`detect_only` モードで実行すると、評価時に**コード側で `apply_masking` を実行**して置換後のleak_rateも自動で計算します。「LLM検出 + コード置換」というハイブリッド構成の評価がワンコマンドで行えます。

```bash
# 検出のみ（コード置換のハイブリッド評価も自動実行）
python 02_04_01/mask_and_eval.py \
  --input data/20b_mails.jsonl \
  --out runs/mask_hybrid_qwen3 \
  --config config/labels.yaml \
  --model qwen3:4b-instruct \
  --task-mode detect_only
```

#### mask-strength（detect_and_mask 時のみ有効）

`--mask-strength` で `masked_text` の指示の強度を切り替えられます。

| 強度 | プロンプトの記述 |
|---|---|
| `required`（デフォルト） | 「masked_text を必ず含める」 |
| `optional` | 「masked_text を含めてもよい」 |

```bash
# 任意指定で実行
python 02_04_01/mask_and_eval.py \
  --input data/20b_mails.jsonl \
  --out runs/mask_optional_qwen3 \
  --config config/labels.yaml \
  --model qwen3:4b-instruct \
  --task-mode detect_and_mask \
  --mask-strength optional
```

#### 出力ファイル

結果は `--out` で指定したディレクトリに出力されます。

| ファイル | 内容 |
|---|---|
| `predictions.jsonl` | メールごとの予測結果 |
| `errors.jsonl` | パース失敗したレコード |
| `evaluation.json` | 検出F1・recall・leak_rate などの集計（detect_only 時はハイブリッド集計も含む） |
| `misses.md` | 検出漏れ・誤検出の詳細 |

---

## 2.4.2 返信生成（baseline）

```bash
python 02_04_02/generate_reply.py \
  --input data/20b_mails.jsonl \
  --input-format jsonl \
  --model qwen3:4b-instruct \
  --output-format jsonl \
  > runs/reply_baseline.jsonl
```

### LLM-as-a-Judgeで評価

```bash
python judge/judge_replies.py \
  --mails data/20b_mails.jsonl \
  --mails-format jsonl \
  --replies runs/reply_baseline.jsonl \
  --replies-format jsonl \
  --model gpt-oss:20b \
  > runs/judge_baseline.json
```

---

## 2.4.3 返信生成（改善版）

### structured_prompt

```bash
python 02_04_03/generate_reply.py \
  --input data/20b_mails.jsonl \
  --input-format jsonl \
  --mode structured_prompt \
  --model qwen3:4b-instruct \
  --output-format jsonl \
  > runs/reply_structured.jsonl
```

### two_step

```bash
python 02_04_03/generate_reply.py \
  --input data/20b_mails.jsonl \
  --input-format jsonl \
  --mode two_step \
  --model qwen3:4b-instruct \
  --output-format jsonl \
  > runs/reply_two_step.jsonl
```

STEP1（下書き）とSTEP2（レビュー）で別モデルを使う場合は `--model-draft` と `--model-review` で指定できます。

```bash
python 02_04_03/generate_reply.py \
  --input data/20b_mails.jsonl \
  --input-format jsonl \
  --mode two_step \
  --model-draft qwen3:4b-instruct \
  --model-review gpt-oss:20b \
  --output-format jsonl \
  > runs/reply_two_step_mixed.jsonl
```

---

## 2.4.4 ダミーデータ生成

### LLMに全部任せる版（Stage 1〜3の試行錯誤）

`--stage` で生成方式を切り替えられます。

| stage | 内容 |
|---|---|
| `1` | 一発生成（条件を整理してtype候補を明示） |
| `2` | 一発生成（Stage 1 + 出力例の例示を追加） |
| `3` | 5つのSTEPに分割して生成 |

```bash
python 02_04_04/generate_mail_full.py \
  --count 3 \
  --model qwen3:4b-instruct \
  --stage 1 \
  --output runs/mails_full.jsonl
```

#### 強調記号の効果を試す（--emphasis）

Stage 1のプロンプトの各条件行に【重要】【必須】【絶対遵守】を付与した版を試せます。本書のコラム「強調記号は効くのか」で扱っている実験です。`--stage 1` 指定時のみ有効で、Stage 2/3で指定した場合は警告が出て無視されます。

```bash
python 02_04_04/generate_mail_full.py \
  --count 3 \
  --model gemma3:12b \
  --stage 1 \
  --emphasis \
  --output runs/mails_full_emphasis.jsonl
```

### hybrid版

```bash
python 02_04_04/generate_mail_hybrid.py \
  --count 10 \
  --model gpt-oss:20b \
  --output runs/mails_hybrid.jsonl
```

`--model` は文章生成に使うモデルです。PIIの生成と挿入はコードで処理されます。

---

## 主なオプション

各スクリプトの詳細は `--help` で確認できます。

```bash
python 02_04_01/classify_and_eval.py --help
python 02_04_03/generate_reply.py --help
```