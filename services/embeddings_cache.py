from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from services.target_identification.runtime import LLM_CACHE, hash_text


class CachedEmbeddings:
    """Shared embedding client with per-text persistent cache."""

    def __init__(
        self,
        *,
        model: str = "models/gemini-embedding-2-preview",
        api_key: Optional[str] = None,
        namespace: str = "default",
        preprocess: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._namespace = namespace
        self._preprocess = preprocess
        self._client: Optional[GoogleGenerativeAIEmbeddings] = None
        self.stats: Dict[str, int] = {
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _get_client(self) -> GoogleGenerativeAIEmbeddings:
        if self._client is None:
            self._client = GoogleGenerativeAIEmbeddings(
                model=self._model,
                api_key=self._api_key,
            )
        return self._client

    def _prepare(self, text: str) -> str:
        cleaned = text or ""
        if self._preprocess is not None:
            cleaned = self._preprocess(cleaned)
        return cleaned

    def _cache_key(self, text: str, *, kind: str) -> str:
        return (
            f"embeddings::{self._namespace}::{self._model}::{kind}::"
            f"{hash_text(text)}"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        prepared = [self._prepare(text) for text in texts]
        results: List[Optional[List[float]]] = [None] * len(prepared)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for idx, text in enumerate(prepared):
            key = self._cache_key(text, kind="doc")
            cached = LLM_CACHE.get(key)
            if cached is not None:
                self.stats["cache_hits"] += 1
                results[idx] = cached
            else:
                self.stats["cache_misses"] += 1
                missing_indices.append(idx)
                missing_texts.append(text)

        if missing_texts:
            embedded = self._get_client().embed_documents(missing_texts)
            for idx, vector in zip(missing_indices, embedded):
                text = prepared[idx]
                key = self._cache_key(text, kind="doc")
                LLM_CACHE[key] = vector
                results[idx] = vector

        return [vector if vector is not None else [] for vector in results]

    def embed_query(self, text: str) -> List[float]:
        prepared = self._prepare(text)
        key = self._cache_key(prepared, kind="query")
        cached = LLM_CACHE.get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached

        self.stats["cache_misses"] += 1
        vector = self._get_client().embed_query(prepared)
        LLM_CACHE[key] = vector
        return vector
