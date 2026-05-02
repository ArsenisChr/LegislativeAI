from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.metrics.pairwise import cosine_similarity

from models.models import Article, CommentTarget
from services.embeddings_cache import CachedEmbeddings
from services.normalizer import normalize_for_comparison
from services.target_identification.prompts import (
    build_free_messages,
    build_narrow_batch_messages,
    build_narrow_single_messages,
)
from services.target_identification.runtime import (
    DEFAULT_CHAT_MODEL_CHAIN,
    LLM_CACHE,
    hash_text,
    invoke_with_retry,
    is_model_unavailable_error,
    is_transient_error,
)
from services.target_identification.schemas import (
    BatchedNLIResult,
    FreeNLIResult,
    NarrowedNLIResult,
)

log_ident = logging.getLogger("legal_analyzer.ident")
log_batch = logging.getLogger("legal_analyzer.batch")
log_cache = logging.getLogger("legal_analyzer.cache")


def _ensure_articles(articles: List) -> List[Article]:
    normalized: List[Article] = []
    for item in articles:
        if isinstance(item, Article):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(
                Article(
                    article_number=str(item.get("article_number", "")),
                    header=str(item.get("header", "")),
                    title=str(item.get("title", "")),
                    body=str(item.get("body", "")),
                )
            )
        else:
            raise TypeError(
                f"Unsupported article type: {type(item)!r}; expected Article or dict."
            )
    return normalized


def _format_candidates_block(candidates: List[Article]) -> str:
    lines: List[str] = []
    for article in candidates:
        body_preview = (article.body or "").strip().replace("\n", " ")
        if len(body_preview) > 1200:
            body_preview = body_preview[:1200] + " [...]"
        lines.append(
            f"--- Άρθρο {article.article_number} ---\n"
            f"Τίτλος: {article.title}\n"
            f"Σώμα: {body_preview}"
        )
    return "\n\n".join(lines)


class TargetIdentifier:
    def __init__(
        self,
        articles: List,
        chat_models: Optional[List[str]] = None,
        embeddings_model: str = "models/gemini-embedding-2-preview",
    ) -> None:
        self.articles: List[Article] = _ensure_articles(articles)
        self._articles_by_number: Dict[str, Article] = {
            a.article_number: a for a in self.articles
        }
        self._chat_models: List[str] = list(chat_models or DEFAULT_CHAT_MODEL_CHAIN)
        self._embeddings_model: str = embeddings_model
        self._api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        self._embeddings = CachedEmbeddings(
            model=self._embeddings_model,
            api_key=self._api_key,
            namespace="target_identification",
            preprocess=normalize_for_comparison,
        )
        self._llm_chain: Optional[List[ChatGoogleGenerativeAI]] = None
        self._narrow_chain: Optional[List[Any]] = None
        self._batch_chain: Optional[List[Any]] = None
        self._free_chain: Optional[List[Any]] = None
        self.stats: Dict[str, int] = {
            "cache_hits": 0,
            "embeddings_cache_hit": 0,
            "llm_calls_ok": 0,
            "llm_retries": 0,
            "model_fallbacks": 0,
            "dropped_invalid_numbers": 0,
            "empty_llm_results": 0,
        }

        log_ident.info(
            "Init TargetIdentifier: %d articles, model chain = %s",
            len(self.articles),
            self._chat_models,
        )
        signature_material = "||".join(
            f"{a.article_number}|{a.title}|{a.body}" for a in self.articles
        )
        self.articles_signature: str = hash_text(signature_material)
        self._corpus_embeddings = self._load_or_build_corpus_embeddings()

    def _get_llm_chain(self) -> List[ChatGoogleGenerativeAI]:
        if self._llm_chain is None:
            self._llm_chain = [
                ChatGoogleGenerativeAI(model=name, api_key=self._api_key)
                for name in self._chat_models
            ]
        return self._llm_chain

    def _get_narrow_chain(self) -> List[Any]:
        if self._narrow_chain is None:
            self._narrow_chain = [
                llm.with_structured_output(NarrowedNLIResult)
                for llm in self._get_llm_chain()
            ]
        return self._narrow_chain

    def _get_batch_chain(self) -> List[Any]:
        if self._batch_chain is None:
            self._batch_chain = [
                llm.with_structured_output(BatchedNLIResult)
                for llm in self._get_llm_chain()
            ]
        return self._batch_chain

    def _get_free_chain(self) -> List[Any]:
        if self._free_chain is None:
            self._free_chain = [
                llm.with_structured_output(FreeNLIResult)
                for llm in self._get_llm_chain()
            ]
        return self._free_chain

    def _corpus_embeddings_cache_key(self) -> str:
        return f"corpus_embeddings::{self._embeddings_model}::{self.articles_signature}"

    def _load_or_build_corpus_embeddings(self) -> np.ndarray:
        if not self.articles:
            return np.zeros((0, 0))

        cache_key = self._corpus_embeddings_cache_key()
        cached = LLM_CACHE.get(cache_key)
        if cached is not None and isinstance(cached, np.ndarray):
            self.stats["embeddings_cache_hit"] = 1
            log_cache.info(
                "Corpus embeddings cache HIT (sig %s, shape %s)",
                self.articles_signature[:8],
                cached.shape,
            )
            return cached

        log_cache.info(
            "Corpus embeddings cache MISS — embedding %d articles...",
            len(self.articles),
        )
        t_start = time.perf_counter()
        corpus_texts = [f"{a.title} {a.body}" for a in self.articles]
        matrix = np.array(self._embeddings.embed_documents(corpus_texts))
        LLM_CACHE[cache_key] = matrix
        log_ident.info(
            "Corpus embedded & cached (%d articles, sig %s, %.1fs)",
            len(self.articles),
            self.articles_signature[:8],
            time.perf_counter() - t_start,
        )
        return matrix

    def _invoke_with_fallback(
        self,
        chain: List[Any],
        messages: Any,
        *,
        label: str = "llm",
    ) -> Any:
        last_exc: Optional[BaseException] = None
        for idx, llm in enumerate(chain):
            model_name = self._chat_models[idx]
            try:
                return invoke_with_retry(
                    llm,
                    messages,
                    context=label,
                    model_name=model_name,
                    stats=self.stats,
                )
            except Exception as exc:
                last_exc = exc
                has_next = idx < len(chain) - 1
                should_fallback = (
                    is_transient_error(exc) or is_model_unavailable_error(exc)
                )
                if should_fallback and has_next:
                    next_model = self._chat_models[idx + 1]
                    reason = "transient" if is_transient_error(exc) else "unavailable"
                    self.stats["model_fallbacks"] += 1
                    log_batch.warning(
                        "%s: %s error on %s (%s) → falling back to %s",
                        label,
                        reason,
                        model_name,
                        type(exc).__name__,
                        next_model,
                    )
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def get_top_k_candidates(
        self, comment_text: str, k: int = 5
    ) -> List[Tuple[Article, float]]:
        if not self.articles or self._corpus_embeddings.size == 0:
            return []

        normalized = normalize_for_comparison(comment_text or "")
        if not normalized:
            return []

        query_embedding = np.array(self._embeddings.embed_query(normalized)).reshape(
            1, -1
        )
        similarities = cosine_similarity(query_embedding, self._corpus_embeddings)[0]

        k = min(k, len(self.articles))
        top_indices = np.argsort(similarities)[::-1][:k]
        return [
            (self.articles[int(idx)], float(similarities[int(idx)]))
            for idx in top_indices
        ]

    def narrow_within_range(
        self,
        comment_text: str,
        chapter_article_numbers: List[str],
    ) -> List[CommentTarget]:
        chapter_articles = [
            self._articles_by_number[num]
            for num in chapter_article_numbers
            if num in self._articles_by_number
        ]
        if not chapter_articles or not (comment_text or "").strip():
            return []

        valid_numbers = [a.article_number for a in chapter_articles]
        cache_key = self._cache_key(comment_text, valid_numbers)

        cached = LLM_CACHE.get(cache_key)
        if cached is None:
            candidates_block = _format_candidates_block(chapter_articles)
            messages = build_narrow_single_messages(comment_text, candidates_block)
            result: NarrowedNLIResult = self._invoke_with_fallback(
                self._get_narrow_chain(),
                messages,
                label="narrow_within_range",
            )
            cached = result.model_dump()
            LLM_CACHE[cache_key] = cached

        return self._narrow_result_to_targets(cached, valid_numbers)

    def narrow_batch(
        self,
        items: List[Tuple[str, str]],
        chapter_article_numbers: List[str],
    ) -> Dict[str, List[CommentTarget]]:
        chapter_articles = [
            self._articles_by_number[num]
            for num in chapter_article_numbers
            if num in self._articles_by_number
        ]
        if not chapter_articles:
            return {cid: [] for cid, _ in items}

        valid_numbers = [a.article_number for a in chapter_articles]
        chapter_tag = f"chapter {valid_numbers[0]}-{valid_numbers[-1]}"
        results: Dict[str, List[CommentTarget]] = {}

        pending: List[Tuple[str, str]] = []
        for cid, text in items:
            if not (text or "").strip():
                results[cid] = []
                continue
            cached = LLM_CACHE.get(self._cache_key(text, valid_numbers))
            if cached is not None:
                self.stats["cache_hits"] += 1
                results[cid] = self._narrow_result_to_targets(cached, valid_numbers)
            else:
                pending.append((cid, text))

        log_batch.info(
            "%s (%d comments): cache %d hit / %d miss",
            chapter_tag,
            len(items),
            len(items) - len(pending),
            len(pending),
        )

        if not pending:
            log_batch.info("%s: fully cached → no LLM call", chapter_tag)
            return results

        candidates_block = _format_candidates_block(chapter_articles)
        comment_lines = [
            f"[{idx}] {text.strip()}" for idx, (_, text) in enumerate(pending, start=1)
        ]
        comments_block = "\n\n".join(comment_lines)
        messages = build_narrow_batch_messages(
            candidates_block=candidates_block,
            comments_block=comments_block,
            expected_count=len(pending),
        )

        log_batch.info(
            "%s: calling %s for %d comments...",
            chapter_tag,
            self._chat_models[0],
            len(pending),
        )
        t_start = time.perf_counter()
        try:
            batched: BatchedNLIResult = self._invoke_with_fallback(
                self._get_batch_chain(),
                messages,
                label=chapter_tag,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            log_batch.error(
                "%s: batched call FAILED after %.1fs — %s: %s",
                chapter_tag,
                elapsed,
                type(exc).__name__,
                exc,
            )
            if is_transient_error(exc) or is_model_unavailable_error(exc):
                log_batch.error(
                    "%s: skipping per-comment fallback (whole model chain "
                    "is unavailable). Try again later.",
                    chapter_tag,
                )
                for cid, _ in pending:
                    results[cid] = []
                return results

            log_batch.warning(
                "%s: non-transient error, falling back to per-comment calls",
                chapter_tag,
            )
            for cid, text in pending:
                try:
                    results[cid] = self.narrow_within_range(text, chapter_article_numbers)
                except Exception as inner_exc:
                    log_batch.error(
                        "%s: per-comment fallback failed for %s — %s: %s",
                        chapter_tag,
                        cid,
                        type(inner_exc).__name__,
                        inner_exc,
                    )
                    results[cid] = []
            return results

        elapsed = time.perf_counter() - t_start
        self.stats["llm_calls_ok"] += 1
        log_batch.info(
            "%s: OK → %d/%d results (%.1fs)",
            chapter_tag,
            len(batched.results),
            len(pending),
            elapsed,
        )

        by_index = {r.comment_index: r for r in batched.results}
        missing_indices: List[int] = []
        specific = chapter_wide = empty = 0

        for idx, (cid, text) in enumerate(pending, start=1):
            per = by_index.get(idx)
            if per is None:
                missing_indices.append(idx)
                results[cid] = []
                empty += 1
                continue
            payload: Dict[str, Any] = {
                "scope": per.scope,
                "article_numbers": list(per.article_numbers),
                "reasoning": per.reasoning,
                "confidence_score": float(per.confidence_score),
            }
            LLM_CACHE[self._cache_key(text, valid_numbers)] = payload
            mapped = self._narrow_result_to_targets(payload, valid_numbers)
            results[cid] = mapped
            if not mapped:
                empty += 1
            elif mapped[0].scope == "chapter_wide":
                chapter_wide += 1
            else:
                specific += 1

        log_batch.info(
            "%s: mapped → %d specific, %d chapter_wide, %d empty",
            chapter_tag,
            specific,
            chapter_wide,
            empty,
        )

        if missing_indices:
            log_batch.warning(
                "%s: LLM skipped entries for indices %s",
                chapter_tag,
                missing_indices,
            )

        return results

    def identify_target_free(
        self, comment_text: str, k: int = 5
    ) -> Optional[CommentTarget]:
        if not (comment_text or "").strip():
            return None

        candidates = self.get_top_k_candidates(comment_text, k=k)
        if not candidates:
            return None

        valid_numbers = [a.article_number for a, _ in candidates]
        cache_key = self._cache_key(comment_text, valid_numbers)

        cached = LLM_CACHE.get(cache_key)
        if cached is None:
            candidates_block = _format_candidates_block([a for a, _ in candidates])
            messages = build_free_messages(comment_text, candidates_block)
            result: FreeNLIResult = self._invoke_with_fallback(
                self._get_free_chain(),
                messages,
                label="identify_target_free",
            )
            cached = result.model_dump()
            LLM_CACHE[cache_key] = cached

        chosen = (cached.get("article_number") or "").strip()
        if not chosen or chosen not in valid_numbers:
            return None

        return CommentTarget(
            article_number=chosen,
            method="ai_nli",
            scope="article",
            chapter_range=[],
            reasoning=(cached.get("reasoning") or "").strip(),
            confidence_score=float(cached.get("confidence_score") or 0.0),
        )

    def _cache_key(
        self, comment_text: str, candidate_numbers: List[str]
    ) -> Tuple[str, Tuple[str, ...], str]:
        return (
            hash_text(comment_text),
            tuple(candidate_numbers),
            self.articles_signature,
        )

    @staticmethod
    def _narrow_result_to_targets(
        result: Dict[str, Any],
        valid_numbers: List[str],
    ) -> List[CommentTarget]:
        scope = result.get("scope", "article")
        reasoning = (result.get("reasoning") or "").strip()
        confidence = float(result.get("confidence_score") or 0.0)

        if scope == "chapter_wide":
            return [
                CommentTarget(
                    article_number="",
                    method="ai_nli_narrowed",
                    scope="chapter_wide",
                    chapter_range=list(valid_numbers),
                    reasoning=reasoning,
                    confidence_score=confidence,
                )
            ]

        raw_numbers = result.get("article_numbers") or []
        valid_set = set(valid_numbers)
        chosen = [n for n in raw_numbers if n in valid_set][:3]

        if not chosen:
            if raw_numbers:
                log_batch.warning(
                    "dropped LLM numbers %s (none in valid set %s, scope=%s)",
                    raw_numbers,
                    valid_numbers,
                    scope,
                )
            else:
                log_batch.debug(
                    "LLM returned no article_numbers (scope=%s, valid=%s)",
                    scope,
                    valid_numbers,
                )
            return []

        return [
            CommentTarget(
                article_number=num,
                method="ai_nli_narrowed",
                scope="article",
                chapter_range=list(valid_numbers),
                reasoning=reasoning,
                confidence_score=confidence,
            )
            for num in chosen
        ]
