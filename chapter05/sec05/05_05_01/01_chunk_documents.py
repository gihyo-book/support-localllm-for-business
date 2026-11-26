from dataclasses import dataclass, field

@dataclass
class Chunk:
    """チャンクのデータクラス"""
    content: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 400,
    overlap: int = 80,
    metadata: dict = None,
) -> list[Chunk]:
    """
    テキストをオーバーラップ付きでチャンキングする
    chunk_size：1チャンクの最大文字数
    overlap：前後のチャンクと重複させる文字数
    """
    if metadata is None:
        metadata = {}

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end]
        is_last = end >= len(text)

        # 文の途中で切れないよう句点・改行で調整する（末尾チャンクでは行わない）
        if not is_last:
            for delimiter in ["。\n", "。", "\n\n", "\n"]:
                last_delim = piece.rfind(delimiter)
                if last_delim > chunk_size * 0.5:
                    piece = piece[:last_delim + len(delimiter)]
                    break

        # strip前の長さ＝実際に消費した文字数（前進量の計算に使う）
        consumed = len(piece)

        content = piece.strip()
        if content:
            chunks.append(Chunk(
                content=content,
                source=source,
                chunk_index=chunk_index,
                metadata=metadata.copy(),
            ))
            chunk_index += 1

        # 末尾に到達したら、余分な重複チャンクを作らずに終了する
        if is_last:
            break

        # 次の開始位置：オーバーラップ分だけ戻す。
        # ただし必ず1以上前進させ、無限ループを防ぐ
        start += max(consumed - overlap, 1)

    return chunks
