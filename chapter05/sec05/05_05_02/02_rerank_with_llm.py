def rerank_with_llm(
    query: str,
    chunks: list[dict],
    llm_client: OpenAI,
    llm_model: str = "qwen3:8b"
) -> list[dict]:
    """
    LLMを使ってチャンクの関連度を再スコアリングする
    """
    reranked = []
    for chunk in chunks:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": """与えられた質問と文章の関連度を0〜10の整数で評価してください。
評価基準：
10：質問への回答が文章に直接含まれている
7〜9：質問に密接に関連する情報が含まれている
4〜6：質問に部分的に関連する情報が含まれている
1〜3：わずかに関連する情報が含まれている
0：質問と無関係
数字のみを返してください。説明は不要です。"""
                },
                {
                    "role": "user",
                    "content": f"質問：{query}\n\n文章：{chunk['content']}"
                }
            ],
            max_tokens=5
        )
        try:
            score = int(response.choices[0].message.content.strip())
            score = max(0, min(10, score))  # 0〜10にクリップする
        except ValueError:
            score = 0
        reranked.append({**chunk, "rerank_score": score})
    # リランクスコアで降順ソートする
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked
def retrieve_and_rerank(
    query: str,
    llm_client: OpenAI,
    top_k: int = 5,
    rerank_top_n: int = 3
) -> list[dict]:
    """
    ベクトル検索→リランキングの2段階検索を行う
    top_k：ベクトル検索で取得する件数（多めに取る）
    rerank_top_n：リランキング後に使用する件数（絞り込む）
    """
    # ベクトル検索で候補を多めに取得する
    candidates = retrieve(query, top_k=top_k)
    # LLMでリランキングする
    reranked = rerank_with_llm(query, candidates, llm_client)
    # 上位N件を返す
    return reranked[:rerank_top_n]
