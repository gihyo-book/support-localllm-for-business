# 本リポジトリのサンプルのご利用にあたって

書籍『業務に組み込むためのローカルLLM入門』の読者向けサンプルリポジトリです。

## 書籍情報

| 項目 | 内容 |
|---|---|
| 書名 | 業務に組み込むためのローカルLLM入門 ── 環境構築・RAG・ファインチューニング |
| ISBN | 978-4-297-16061-6 |
| 出版社 | 株式会社技術評論社 |
| 書籍ページ | <https://gihyo.jp/book/2026/978-4-297-16061-6> |

## 提供するファイルやサポート情報の概要

書籍の内容を手元で試すためのサンプルコード、設定ファイル、サンプルデータ、実行手順を掲載しています。

| 章 | 配布内容 | 配布状況 |
|---|---|---|
| 第1章 なぜ今ローカルLLMなのか？ | — | 配布ファイルなし |
| [第2章 手元のPCでローカルLLMを動かす](chapter02/) | OpenAI互換APIの呼び出し、ダミーメールの分類・マスキング・返信生成など | 公開済み |
| 第3章 GPUクラウドでLLMを動かす | GPUクラウド上でのLLM実行、チャットアプリケーションの構築、Function Callingのサンプル | 準備中 |
| [第4章 ローカルLLMを使ったアプリケーションの考え方](chapter04/README.md) | 問い合わせの分類、構造化出力、画像入力のサンプル | 公開済み |
| [第5章 ローカルLLMを使ったRAGアプリケーションの開発](chapter05/README.md) | Ollamaによる埋め込みと、文書の登録・検索・回答生成をつなぐRAGのサンプル | 公開済み |
| [第6章 モデルをチューニングする](chapter06/README.md) | Qwen3.5-4B-BaseのLoRAファインチューニングに使うデータセット構築・学習・評価スクリプトと設定ファイル | 公開済み |

## 必要な環境や利用方法

このリポジトリの **Code → Download ZIP** からファイルを一括で取得するか、Gitでクローンしてください。

```bash
git clone https://github.com/gihyo-book/support-localllm-for-business.git
```

ダウンロード後は、試したい章・節のREADMEを開き、依存パッケージやモデルを用意してください。掲載コードの一部は、書籍の前後のリストや入力ファイルと組み合わせて使います。

| 章 | 主な実行環境・前提 | 手順 |
|---|---|---|
| 第2章 | Python 3.10以降、Ollama。例に応じて `qwen3:4b-instruct`、`gemma3:12b`、`gpt-oss:20b` を使用します。`gpt-oss:20b` の実行にはメモリ32GB以上を推奨します。 | [2.3.4 API呼び出し](chapter02/sec03/02_03_04/README.md)・[2.4 業務データのハンズオン](chapter02/sec04/README.md) |
| 第4章 | Python、OpenAI Python SDK、OllamaなどのOpenAI互換API。画像入力の補助サンプルでは `gemma3:4b` と、手元のPNGまたはJPEG画像を使います。 | [掲載コード一覧](chapter04/README.md)・[画像入力の補助サンプル](chapter04/sec03/04_03_05/quickstart/README.md) |
| 第5章 | Python、Ollama、OpenAI Python SDK、Chroma。主なモデルは `qwen3:8b` と `nomic-embed-text` です。PDFの取り込みを試す場合は `pdfplumber` と手元のPDFを用意してください。 | [掲載コード一覧](chapter05/README.md)・[RAGの補助サンプル](chapter05/sec05/05_05_04/quickstart/README.md) |
| 第6章 | Python 3.12以降、uv、CUDA 12.4とbf16に対応したNVIDIA GPU。書籍の学習例ではH100 80GBを1枚使用します。データセットのアノテーションにはOpenAI互換APIを使います。 | [環境構築・学習・評価の手順](chapter06/README.md) |

第6章のアノテーションを実行するときは、`chapter06/create_dataset/.env` に、実装が参照する `LLM_BASE_URL`（APIのベースURL）、`LLM_KEY`（APIキー）、`LLM_MODEL`（モデル名）を設定してください。

## サポート・お問い合わせ

本書のサポート情報、正誤表、お問い合わせ窓口は、[技術評論社の書籍ページ](https://gihyo.jp/book/2026/978-4-297-16061-6)の「サポート情報」をご確認ください。
