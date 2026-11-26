# 第4章 ローカルLLMを使ったアプリケーションの考え方

4.3.5節「マルチモーダルモデルを呼び出す実装例」に対応するサンプルです。画像をOllamaのOpenAI互換APIに渡し、日本語で説明を生成します。

## 必要な環境

- Python 3.11または3.12
- Ollamaと画像入力に対応したモデル（この例では `gemma3:4b`）
- 手元にあるPNGまたはJPEG画像

## 実行方法

Ollamaを起動し、モデルを取得してください。以下は `sec03/04_03_05/quickstart` ディレクトリでの実行例です。

```bash
ollama pull gemma3:4b
python -m venv .venv
python -m pip --python .venv install -r requirements.txt
.venv/bin/python describe_image.py /path/to/image.png
```

Windowsでは最後のコマンドの `.venv/bin/python` を `.venv\Scripts\python.exe` に読み替えてください。別の画像対応モデルを使う場合は `--model モデル名` を指定してください。
