# パイプラインの初期化
rag = RAGPipeline(
    llm_model="qwen3:8b",
    embedding_model="nomic-embed-text"
)
# ドキュメントの追加
rag.add_pdf(
    pdf_path="docs/就業規則.pdf",
    metadata={"title": "就業規則", "department": "人事部"}
)
rag.add_pdf(
    pdf_path="docs/経費規程.pdf",
    metadata={"title": "経費規程", "department": "経理部"}
)
# 統計確認
stats = rag.get_stats()
print(f"インデックス済みチャンク数：{stats['total_chunks']}")
# 質問してみる
result = rag.query("有給休暇の申請方法を教えてください")
print("\n【回答】")
print(result["answer"])
print("\n【参照元】")
for src in result["sources"]:
    print(f"  - {src}")
# メタデータでフィルタリングしながら質問する
result = rag.query(
    question="経費精算の締め日はいつですか？",
    filter_metadata={"department": "経理部"}
)
print("\n【回答（経理部ドキュメントのみ参照）】")
print(result["answer"])
