def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    取得したチャンクと質問からプロンプトを構築する
    """
    if not chunks:
        return f"""参考情報が見つかりませんでした。
質問：{query}
参考情報がないため、この質問には回答できません。
担当部署に直接お問い合わせください。"""
    # チャンクを参考情報としてフォーマットする
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        source_info = meta.get("title", meta.get("source", "不明"))
        page_info = f" p.{meta['page']}" if "page" in meta else ""
        context_parts.append(
            f"【参考情報{i+1}】出典：{source_info}{page_info}\n"
            f"{chunk['content']}"
        )
    context = "\n\n".join(context_parts)
    return f"""以下の参考情報をもとに質問に答えてください。
ルール：
- 参考情報に含まれる内容のみを使って回答してください
- 参考情報にない内容については「記載が見つかりませんでした」と答えてください
- 回答の末尾に参照した出典を記載してください
{context}
質問：{query}"""
def generate_answer(
    query: str,
    chunks: list[dict],
    llm_client: OpenAI,
    llm_model: str = "qwen3:8b"
) -> dict:
    """
    取得したチャンクをもとにLLMで回答を生成する
    """
    prompt = build_prompt(query, chunks)
    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": "あなたは社内情報に詳しいアシスタントです。提供された参考情報をもとに、正確で簡潔な回答を提供してください。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    answer = response.choices[0].message.content
    sources = list({
        chunk["metadata"].get("title", chunk["metadata"].get("source", "不明"))
        for chunk in chunks
    })
    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks)
    }
