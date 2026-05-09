from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass
class Chunk:
    chunk_id: str
    page_number: int
    text: str


def chunk_pages(pages, chunk_size: int = 1200, overlap: int = 150) -> list[Chunk]:
    """
    Splits extracted PDF pages into overlapping text chunks.

    Expected page object format:
    page.page_number
    page.text
    """
    chunks = []

    for page in pages:
        text = page.text
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end].strip()

            if len(chunk_text) > 80:
                chunks.append(
                    Chunk(
                        chunk_id=f"p{page.page_number}_c{chunk_idx}",
                        page_number=page.page_number,
                        text=chunk_text,
                    )
                )
                chunk_idx += 1

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return chunks


class SimplePageIndex:
    """
    Lightweight local page-aware retrieval index.

    This is suitable for your 4 GB RAM system because it uses TF-IDF instead
    of heavy embedding models.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30000
        )
        self.matrix = None
        self.chunks = []

    def build(self, chunks: list[Chunk]):
        self.chunks = chunks
        docs = [c.text for c in chunks]

        if docs:
            self.matrix = self.vectorizer.fit_transform(docs)
        else:
            self.matrix = None

        return self

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        if self.matrix is None or not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = linear_kernel(query_vec, self.matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:
            score = float(scores[idx])

            if score <= 0:
                continue

            chunk = self.chunks[idx]

            results.append({
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "score": round(score, 4)
            })

        return results
