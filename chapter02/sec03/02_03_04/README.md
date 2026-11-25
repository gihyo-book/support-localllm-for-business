# 2.3.4 OpenAI互換APIでローカルモデルを呼び出す

## セットアップ

Python 3.10 以降で動作を確認しています。このディレクトリで、必要なパッケージをインストールします。

```bash
pip install -r requirements.txt
```

[uv](https://docs.astral.sh/uv/) をお使いの方は、代わりに以下を実行してください。

```bash
uv venv
uv pip install -r requirements.txt
```

Ollamaが起動していることを確認し（macOS / Windows はアプリを起動していればOKです）、モデルをダウンロードしておきます。

```bash
ollama pull qwen3:4b-instruct
```

## ファイル構成

| ファイル | 内容 |
|---|---|
| `exec_chat_completion_01.py` | まず動かしてみる（zero-shot） |
| `exec_chat_completion_02.py` | ビジネスメール作成を依頼する |
| `exec_chat_completion_03.py` | systemで前提条件を定義する |
| `exec_chat_completion_04.py` | Few-shotで出力を例示する |

## 実行方法

このディレクトリで実行してください。

```bash
python exec_chat_completion_01.py

# uv の場合
uv run python exec_chat_completion_01.py
```

出力内容は実行ごとに変わります。