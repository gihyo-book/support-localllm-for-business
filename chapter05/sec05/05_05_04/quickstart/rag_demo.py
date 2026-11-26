import argparse

import chromadb
from openai import OpenAI


DOCUMENTS = {
    "travel": "出張の交通費は、帰着後5営業日以内に精算します。",
    "equipment": "備品購入が税込1万円以上の場合、購入前に所属長の承認が必要です。",
    "meeting": "会議室の予約は、社内カレンダーで行います。",
}


def embed(client, texts):
    response = client.embeddings.create(model="nomic-embed-text", input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def main():
    parser = argparse.ArgumentParser(description="架空の社内規程を検索して回答します。")
    parser.add_argument("question", nargs="?", default="出張の交通費はいつまでに精算しますか？")
    args = parser.parse_args()

    client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
    collection = chromadb.EphemeralClient().create_collection("sample_rules")
    documents = list(DOCUMENTS.values())
    collection.add(ids=list(DOCUMENTS), documents=documents, embeddings=embed(client, documents))

    matches = collection.query(query_embeddings=embed(client, [args.question]), n_results=2)
    excerpts = matches["documents"][0]
    context = "\n".join(f"- {excerpt}" for excerpt in excerpts)
    response = client.chat.completions.create(
        model="qwen3:8b",
        messages=[
            {
                "role": "system",
                "content": "参考資料だけに基づいて日本語で回答してください。根拠がなければ、資料からは分からないと答えてください。",
            },
            {"role": "user", "content": f"参考資料\n{context}\n\n質問\n{args.question}"},
        ],
        temperature=0,
    )
    print(f"検索結果\n{context}\n\n回答\n{response.choices[0].message.content}")


if __name__ == "__main__":
    main()
