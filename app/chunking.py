"""Fixed-size word chunking with overlap. A "token" here means a whitespace-separated word."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc: str
    index: int
    text: str

    @property
    def tokens(self) -> int:
        return len(self.text.split())


def chunk_text(doc: str, text: str, size: int, overlap: int) -> list[Chunk]:
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than chunk size")
    words = text.split()
    step = size - overlap
    chunks: list[Chunk] = []
    for start in range(0, max(len(words), 1), step):
        piece = words[start:start + size]
        if not piece:
            break
        chunks.append(Chunk(doc, len(chunks), " ".join(piece)))
        if start + size >= len(words):
            break
    return chunks
