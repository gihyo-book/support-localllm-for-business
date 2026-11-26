# 第5章 ローカルLLMを使ったRAGアプリケーションの開発

Ollamaでの埋め込み（5.3.4）とRAGパイプラインの実装（5.5）のサンプルを収録しています。
掲載コードには、書籍の前後のリストや入力ファイルと組み合わせて使う例が含まれます。

| 項番号 | 掲載コード | ファイル |
|---|---|---|
| 5.3.4 | Ollamaでの埋め込み | [`01_ollama_embeddings.py`](sec03/05_03_04/01_ollama_embeddings.py) |
| 5.5.1 | チャンキングの設計 | [`01_chunk_documents.py`](sec05/05_05_01/01_chunk_documents.py) |
| 5.5.1 | インデックス構築 | [`02_build_index.py`](sec05/05_05_01/02_build_index.py) |
| 5.5.2 | リスト5.1 基本的な類似検索 | [`01_retrieve.py`](sec05/05_05_02/01_retrieve.py) |
| 5.5.2 | LLMを使ったリランキング | [`02_rerank_with_llm.py`](sec05/05_05_02/02_rerank_with_llm.py) |
| 5.5.3 | リスト5.2 プロンプト構築と回答生成 | [`01_build_prompt.py`](sec05/05_05_03/01_build_prompt.py) |
| 5.5.4 | RAGパイプラインの統合 | [`01_rag_pipeline.py`](sec05/05_05_04/01_rag_pipeline.py) |
| 5.5.4 | RAGパイプラインの動作確認 | [`02_run_pipeline.py`](sec05/05_05_04/02_run_pipeline.py) |

## 実行環境

Python、Ollama、OpenAI Python SDK、Chromaを使用します。PDFの取り込み例には `pdfplumber` とPDFを用意してください。

## すぐに試せる補助サンプル

[実行手順](sec05/05_05_04/quickstart/README.md)付きの補助サンプルも用意しています。
