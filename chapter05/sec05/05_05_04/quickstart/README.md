# 第5章 ローカルLLMを使ったRAGアプリケーションの開発

5.5節「RAGパイプラインの設計と実装」に対応する最小サンプルです。架空の社内規程を埋め込みモデルでベクトル化してChromaに登録し、検索結果をもとにLLMで回答を生成します。サンプル文書はスクリプトに含まれます。実行のたびにインデックスを作り直します。

## 必要な環境

- Python 3.11または3.12
- Ollama
- 回答生成モデル `qwen3:8b` と埋め込みモデル `nomic-embed-text`

## 実行方法

Ollamaを起動し、モデルを取得してください。以下は `sec05/05_05_04/quickstart` ディレクトリでの実行例です。

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
python -m venv .venv
python -m pip --python .venv install -r requirements.txt
.venv/bin/python rag_demo.py
```

Windowsでは最後のコマンドの `.venv/bin/python` を `.venv\Scripts\python.exe` に読み替えてください。質問を変更するには、ファイル名の後に `"備品購入に承認が必要なのはいくらからですか？"` のように指定します。

既定の質問では、出張交通費の精算期限が「帰着後5営業日以内」であることを示す文が検索結果に含まれます。回答の表現は実行ごとに変わる場合があります。
