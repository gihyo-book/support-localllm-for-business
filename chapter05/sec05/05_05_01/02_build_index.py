import chromadb
from openai import OpenAI
import pdfplumber
import hashlib

# クライアントの初期化
embedding_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="dummy",
)
chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(
    name="company_docs",
    metadata={"hnsw:space": "cosine"},
)
EMBEDDING_MODEL = "nomic-embed-text"

def embed_texts(texts: list[str]) -> list[list[float]]:
    """複数テキストをまとめてベクトル化する"""
    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]

def generate_chunk_id(source: str, global_index: int) -> str:
    """チャンクの一意なIDを生成する（PDF全体での通し番号を使う）"""
    raw = f"{source}_{global_index}"
    return hashlib.md5(raw.encode()).hexdigest()

def index_pdf(pdf_path: str, metadata: dict = None) -> int:
    """PDFをインデックスに追加する。追加したチャンク数を返す"""
    if metadata is None:
        metadata = {}

    # PDFからテキストを抽出する
    all_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            raw_text = page.extract_text()
            if not raw_text:
                continue
            # ページ単位でチャンキングする
            page_metadata = {
                **metadata,
                "page": page_num + 1,
                "source": pdf_path,
            }
            chunks = chunk_text(
                text=raw_text,
                source=pdf_path,
                chunk_size=400,
                overlap=80,
                metadata=page_metadata,
            )
            all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    # バッチでベクトル化してDBに保存する
    batch_size = 32
    total_added = 0
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [chunk.content for chunk in batch]
        # PDF全体での通し番号（i + バッチ内オフセット）でIDを採番する
        ids = [
            generate_chunk_id(pdf_path, i + offset)
            for offset in range(len(batch))
        ]
        metadatas = [chunk.metadata for chunk in batch]
        embeddings = embed_texts(texts)

        # DBに追加する（既存のIDは上書きする）
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_added += len(batch)
        print(f"  {total_added}/{len(all_chunks)} チャンクを処理しました")

    return total_added

# 使用例
print("就業規則をインデックスに追加します...")
count = index_pdf(
    pdf_path="docs/就業規則.pdf",
    metadata={"title": "就業規則", "department": "人事部", "year": 2024},
)
print(f"完了：{count}チャンクを追加しました")
