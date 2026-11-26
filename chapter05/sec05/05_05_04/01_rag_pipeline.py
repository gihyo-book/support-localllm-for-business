class RAGPipeline:
    """
    RAGパイプラインの全体を管理するクラス
    """

    def __init__(
        self,
        llm_base_url: str = "http://localhost:11434/v1",
        llm_model: str = "qwen3:8b",
        embedding_model: str = "nomic-embed-text",
        db_path: str = "./rag_db",
        collection_name: str = "company_docs",
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        # LLM・Embeddingクライアントの初期化
        self.llm_client = OpenAI(
            base_url=llm_base_url,
            api_key="dummy",
        )
        # ChromaDBの初期化
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """テキストをベクトル化する"""
        response = self.llm_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def _index_chunks(self, chunks: list) -> int:
        """チャンクのリストをバッチでベクトル化し、self.collectionに投入する"""
        if not chunks:
            return 0

        batch_size = 32
        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.content for c in batch]
            # source内での通し番号（i + オフセット）でIDを採番する
            ids = [
                generate_chunk_id(c.source, i + offset)
                for offset, c in enumerate(batch)
            ]
            metadatas = [c.metadata for c in batch]
            embeddings = self.embed(texts)
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total_added += len(batch)
        return total_added

    def add_document(
        self,
        text: str,
        source: str,
        metadata: dict = None,
    ) -> int:
        """ドキュメントをインデックスに追加する"""
        chunks = chunk_text(
            text=text,
            source=source,
            metadata=metadata or {},
        )
        return self._index_chunks(chunks)

    def add_pdf(self, pdf_path: str, metadata: dict = None) -> int:
        """PDFをインデックスに追加する"""
        if metadata is None:
            metadata = {}

        # PDFからページ単位でチャンキングし、全チャンクを連結する
        all_chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                raw_text = page.extract_text()
                if not raw_text:
                    continue
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

        # ID採番（通し番号）と投入はadd_documentと共通の処理に委ねる
        return self._index_chunks(all_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = False,
        rerank_top_n: int = 3,
        filter_metadata: dict = None,
    ) -> list[dict]:
        """クエリに関連するチャンクを検索する"""
        if use_rerank:
            return retrieve_and_rerank(
                query=query,
                llm_client=self.llm_client,
                top_k=top_k,
                rerank_top_n=rerank_top_n,
            )
        return retrieve(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

    def query(
        self,
        question: str,
        top_k: int = 3,
        use_rerank: bool = False,
        filter_metadata: dict = None,
    ) -> dict:
        """
        RAGパイプラインのメインメソッド
        質問を受け取り、検索して回答を生成する
        """
        # 検索フェーズ
        chunks = self.search(
            query=question,
            top_k=top_k,
            use_rerank=use_rerank,
            filter_metadata=filter_metadata,
        )
        # 生成フェーズ
        result = generate_answer(
            query=question,
            chunks=chunks,
            llm_client=self.llm_client,
            llm_model=self.llm_model,
        )
        return result

    def get_stats(self) -> dict:
        """インデックスの統計情報を返す"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
        }
