# 『業務に組み込むためのローカルLLM入門』読者サポート

本書のサンプルコード、設定ファイル、サンプルデータ、実行手順を掲載しています。

## 書籍情報

| 項目 | 内容 |
|---|---|
| 書名 | 業務に組み込むためのローカルLLM入門 ── 環境構築・RAG・ファインチューニング |
| ISBN | 978-4-297-16061-6 |
| 出版社 | 株式会社技術評論社 |
| 書籍ページ | [技術評論社の書籍ページ](https://gihyo.jp/book/2026/978-4-297-16061-6) |

## 提供するファイルやサポート情報の概要

| 章 | 配布内容 |
|---|---|
| 第1章 なぜ今ローカルLLMなのか？ | 配布ファイルなし |
| [第2章 手元のPCでローカルLLMを動かす](chapter02/) | OpenAI互換APIの呼び出し、ダミーメールの分類・マスキング・返信生成などのサンプル |
| [第3章 GPUクラウドでLLMを動かす](chapter03/README.md) | vLLM、Open WebUI、MCPツールサーバーを起動するDocker Compose設定 |
| [第4章 ローカルLLMを使ったアプリケーションの考え方](chapter04/README.md) | 問い合わせ分類、構造化出力、画像入力のサンプル |
| [第5章 ローカルLLMを使ったRAGアプリケーションの開発](chapter05/README.md) | Ollamaでの埋め込みと、文書の取り込み・検索・回答生成をつなぐRAGパイプラインのサンプル |
| [第6章 モデルをチューニングする](chapter06/README.md) | Qwen3.5-4B-BaseのLoRAファインチューニングに使うデータセット構築・学習・評価スクリプトと設定ファイル |

## 必要な環境や利用方法

このリポジトリの **Code → Download ZIP** からファイルを取得するか、Gitでクローンしてください。

```bash
git clone https://github.com/gihyo-book/support-localllm-for-business.git
```

試したい章・節のREADMEを開き、必要なライブラリ、モデル、入力ファイルを用意してください。一部の掲載コードは、書籍の前後のリストと組み合わせて使います。

| 章 | 主な実行環境・前提 | 手順 |
|---|---|---|
| 第2章 | Python 3.10以降、Ollama。使用するモデルはサンプルによって異なります。 | [2.3.4 API呼び出し](chapter02/sec03/02_03_04/README.md)・[2.4 ハンズオン](chapter02/sec04/README.md) |
| 第3章 | NVIDIA GPUを搭載したLinuxサーバー、Docker Compose、NVIDIA Container Toolkit。配布設定では `openai/gpt-oss-120b` を使います。 | [配布ファイル](chapter03/README.md)・書籍の3.5節 |
| 第4章 | Python、OpenAI Python SDK、OllamaなどのOpenAI互換API。画像入力の補助サンプルでは `gemma3:4b` とPNGまたはJPEG画像を使います。 | [掲載コード一覧](chapter04/README.md)・[画像入力の補助サンプル](chapter04/sec03/04_03_05/quickstart/README.md) |
| 第5章 | Python、Ollama、OpenAI Python SDK、Chroma。主なモデルは `qwen3:8b` と `nomic-embed-text` です。PDFの取り込み例では `pdfplumber` と入力用のPDFを使います。 | [掲載コード一覧](chapter05/README.md)・[RAGの補助サンプル](chapter05/sec05/05_05_04/quickstart/README.md) |
| 第6章 | Python 3.12以降、uv、CUDA 12.4およびbf16に対応したNVIDIA GPU。書籍の学習例ではH100 80GBを1枚使用します。アノテーションにはOpenAI互換APIを使います。 | [環境構築・学習・評価の手順](chapter06/README.md) |

## ライセンス

各章のサンプルコードは、[MITライセンス](LICENSE)で提供します。

サンプルで使用する外部データセットやモデルは、それぞれの提供元が定めるライセンス・利用条件に従ってください。

## サポート・お問い合わせ

本書のサポート情報、正誤表、お問い合わせ窓口は、[技術評論社の書籍ページ](https://gihyo.jp/book/2026/978-4-297-16061-6)の「サポート情報」をご確認ください。
