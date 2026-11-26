def retrieve(
    query: str,
    top_k: int = 5,
    filter_metadata: dict = None
) -> list[dict]:
    """
    クエリに関連するチャンクを検索して返す
    query：ユーザーの質問
    top_k：取得するチャンク数
    filter_metadata：メタデータによる絞り込み条件
    """
    # クエリをベクトル化する
    query_embedding = embed_texts([query])[0]
    # ベクトルDBで類似検索する
    search_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k
    }
    if filter_metadata:
        search_kwargs["where"] = filter_metadata
    results = collection.query(**search_kwargs)
    # 結果を整形して返す
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "similarity": 1 - results["distances"][0][i]
        })
    return chunks
