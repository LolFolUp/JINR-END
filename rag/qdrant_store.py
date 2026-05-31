"""Qdrant-backed vector knowledge base with local fallback retrieval."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from rag.embedder import Embedder
from rag.knowledge_base import SOPChunk, get_all_chunks
from rag.retriever import SOPRetriever


GENERAL_HPC_CHUNKS: List[SOPChunk] = [
    SOPChunk(
        chunk_id="general-001",
        fault_type="general",
        title="HPC incident communication procedure",
        text=(
            "General HPC maintenance: create an incident ticket, notify the NOC, "
            "record impacted jobs, preserve logs, and avoid destructive remediation "
            "while user workloads are active."
        ),
        tags=["incident", "noc", "maintenance"],
    ),
    SOPChunk(
        chunk_id="general-002",
        fault_type="general",
        title="Scheduler drain procedure",
        text=(
            "Before isolating a node, drain it from the scheduler, block new job "
            "placements, verify checkpoint status, and notify affected job owners."
        ),
        tags=["scheduler", "drain", "jobs"],
    ),
]


class QdrantStore:
    """Semantic SOP retrieval using in-memory Qdrant when available."""

    COLLECTION = "hpc_remediation_sops"

    def __init__(self, embedder: Embedder | None = None, top_k: int = 3) -> None:
        self.embedder = embedder or Embedder()
        self.top_k = top_k
        self.chunks = get_all_chunks() + GENERAL_HPC_CHUNKS
        self.retrieval_method = "bow_fallback"
        self._client = None
        self._vectors: np.ndarray | None = None
        self._init_qdrant()

    def _init_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, PointStruct, VectorParams

            texts = [self._document_text(chunk) for chunk in self.chunks]
            vectors = self.embedder.embed(texts)
            if vectors.size == 0:
                return

            client = QdrantClient(":memory:")
            client.recreate_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=int(vectors.shape[1]), distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "fault_type": chunk.fault_type,
                        "title": chunk.title,
                        "text": chunk.text,
                        "tags": chunk.tags,
                    },
                )
                for i, chunk in enumerate(self.chunks)
            ]
            client.upsert(collection_name=self.COLLECTION, points=points)
            self._client = client
            self._vectors = vectors
            self.retrieval_method = "qdrant"
        except Exception:
            self._client = None
            self.retrieval_method = "bow_fallback"

    def retrieve(
        self,
        *,
        fault_type: str,
        rc_node_type: str,
        query: str,
        top_k: int | None = None,
    ) -> Tuple[List[Dict[str, object]], str]:
        """Return top SOP chunks and retrieval method label."""
        limit = top_k or self.top_k
        full_query = f"{fault_type} {rc_node_type} {query}"
        if self._client is not None:
            results = self._retrieve_qdrant(full_query, fault_type, limit)
            if results:
                return results, self.retrieval_method
        return self._retrieve_bow(full_query, fault_type, limit), "bow_fallback"

    def _retrieve_qdrant(self, query: str, fault_type: str, limit: int) -> List[Dict[str, object]]:
        query_vec = self.embedder.embed([query])[0]
        search_limit = min(len(self.chunks), max(limit * 3, limit))
        try:
            hits = self._client.search(
                collection_name=self.COLLECTION,
                query_vector=query_vec.tolist(),
                limit=search_limit,
            )
        except Exception:
            return []

        ranked: List[Dict[str, object]] = []
        for hit in hits:
            payload = hit.payload or {}
            if payload.get("fault_type") not in {fault_type, "general"}:
                continue
            ranked.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "fault_type": payload.get("fault_type"),
                    "title": payload.get("title"),
                    "text": payload.get("text"),
                    "score": float(hit.score),
                }
            )
            if len(ranked) == limit:
                break
        return ranked

    def _retrieve_bow(self, query: str, fault_type: str, limit: int) -> List[Dict[str, object]]:
        retriever = SOPRetriever(embedder=self.embedder, top_k=limit)
        fault_matches = retriever.retrieve(query, fault_type=fault_type)
        general_matches = self._rank_general_chunks(query)
        combined = list(fault_matches) + general_matches
        combined.sort(key=lambda item: item[1], reverse=True)
        return [
            {
                "chunk_id": chunk.chunk_id,
                "fault_type": chunk.fault_type,
                "title": chunk.title,
                "text": chunk.text,
                "score": float(score),
            }
            for chunk, score in combined[:limit]
        ]

    def _rank_general_chunks(self, query: str) -> List[Tuple[SOPChunk, float]]:
        texts = [self._document_text(chunk) for chunk in GENERAL_HPC_CHUNKS]
        query_vec = self.embedder.embed_query(query, corpus=texts)
        doc_matrix = self.embedder.embed(texts)
        return [
            (chunk, self.embedder.cosine_similarity(query_vec, doc_matrix[i]))
            for i, chunk in enumerate(GENERAL_HPC_CHUNKS)
        ]

    @staticmethod
    def _document_text(chunk: SOPChunk) -> str:
        return f"{chunk.fault_type} {chunk.title} {' '.join(chunk.tags)}\n{chunk.text}"
