from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy")

def embed_text(text: str) -> list[float]:
    """Ollamaを使ってテキストをベクトル化する"""
    response = client.embeddings.create(
        model="nomic-embed-text",
        input=text,
    )
    return response.data[0].embedding

# 使用例
vector = embed_text("有給休暇の申請方法を教えてください")
print(f"ベクトルの次元数：{len(vector)}")
print(f"最初の5要素：{vector[:5]}")
