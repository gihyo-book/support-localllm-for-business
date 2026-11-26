# 第4章 ローカルLLMを使ったアプリケーションの考え方

問い合わせ分類・構造化出力（4.3.1）と画像入力（4.3.5）のサンプルを収録しています。
掲載コードには、書籍の前後のリストや入力ファイルと組み合わせて使う例が含まれます。

| 項番号 | 掲載コード | ファイル |
|---|---|---|
| 4.3.1 | 問い合わせの分類 | [`01_classify_inquiry.py`](sec03/04_03_01/01_classify_inquiry.py) |
| 4.3.1 | 請求書の構造化データ抽出 | [`02_extract_invoice_info.py`](sec03/04_03_01/02_extract_invoice_info.py) |
| 4.3.1 | JSONの解析と検証 | [`03_parse_structured_output.py`](sec03/04_03_01/03_parse_structured_output.py) |
| 4.3.5 | リスト4.1 領収書画像からの情報抽出 | [`01_receipt_image.py`](sec03/04_03_05/01_receipt_image.py) |
| 4.3.5 | リスト4.2 画像とテキストを組み合わせた製品サポート | [`02_product_support_image.py`](sec03/04_03_05/02_product_support_image.py) |

## 実行環境

Python、OpenAI Python SDK、OllamaなどのOpenAI互換APIを使用します。画像入力には対応モデルとPNGまたはJPEG画像を用意してください。

## すぐに試せる補助サンプル

[実行手順](sec03/04_03_05/quickstart/README.md)付きの補助サンプルも用意しています。
