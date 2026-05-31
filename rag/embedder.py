"""Local text embeddings without paid API calls."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional

import numpy as np

_ST_MODEL: Optional[object] = None


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _bow_vector(tokens: List[str], vocab: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    counts = Counter(tokens)
    for token, count in counts.items():
        if token in vocab:
            vec[vocab[token]] = float(count)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class Embedder:
    """Embed text locally; prefers sentence-transformers, falls back to bag-of-words."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._use_st = False
        self._model = None
        self._vocab: dict[str, int] = {}
        self._try_load_sentence_transformers()

    def _try_load_sentence_transformers(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._use_st = True
        except Exception:
            self._use_st = False

    def _fit_vocab(self, corpus: List[str]) -> None:
        tokens: List[str] = []
        for doc in corpus:
            tokens.extend(_tokenize(doc))
        unique = sorted(set(tokens))
        self._vocab = {t: i for i, t in enumerate(unique)}

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if self._use_st and self._model is not None:
            vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return np.asarray(vectors, dtype=np.float32)

        if not self._vocab:
            self._fit_vocab(texts)
        dim = len(self._vocab) or 1
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = _bow_vector(_tokenize(text), self._vocab)
        return out

    def embed_query(self, query: str, corpus: Optional[List[str]] = None) -> np.ndarray:
        if corpus and not self._use_st:
            self._fit_vocab(corpus + [query])
        return self.embed([query])[0]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
