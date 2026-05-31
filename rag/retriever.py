"""Semantic search over SOP knowledge base."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from rag.embedder import Embedder
from rag.knowledge_base import SOPChunk, get_all_chunks, get_chunks_for_fault


class SOPRetriever:
    """Retrieve top-k SOP chunks relevant to fault context."""

    def __init__(self, embedder: Embedder | None = None, top_k: int = 3) -> None:
        self.embedder = embedder or Embedder()
        self.top_k = top_k
        self._chunks = get_all_chunks()
        self._corpus = [c.text for c in self._chunks]
        self._matrix: np.ndarray | None = None

    def _ensure_index(self) -> None:
        if self._matrix is None:
            self._matrix = self.embedder.embed(self._corpus)

    def retrieve(
        self,
        query: str,
        fault_type: str | None = None,
    ) -> List[Tuple[SOPChunk, float]]:
        """Return top-k (chunk, score) pairs for query."""
        pool = get_chunks_for_fault(fault_type) if fault_type else self._chunks
        if not pool:
            return []

        texts = [c.text for c in pool]
        query_vec = self.embedder.embed_query(query, corpus=texts)
        doc_matrix = self.embedder.embed(texts)

        scores = [
            self.embedder.cosine_similarity(query_vec, doc_matrix[i])
            for i in range(len(pool))
        ]
        ranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
        return ranked[: self.top_k]

    def retrieve_texts(self, query: str, fault_type: str | None = None) -> List[str]:
        return [chunk.text for chunk, _ in self.retrieve(query, fault_type)]
