"""
Legal NLI pipeline for mapping public consultation comments to the
specific article(s) of a law they actually target.

On opengov.gr comments are posted per chapter, so the "ΑΡΘΡΟ" cell of the
Excel export usually carries a range like "(άρθρα 3 - 10)" which is just
platform metadata, not the citizen's intent. The real targeting signal
lives in the free-text comment body. This module therefore exposes three
AI paths:

    1) `narrow_batch(items, chapter_article_numbers)`
        Batched version of range narrowing: many comments that share the
        same chapter are classified in a single LLM call, sharing the
        chapter context. This is the main path used in production.

    2) `narrow_within_range(comment, chapter_article_numbers)`
        Single-comment range narrowing. Convenience for tests and ad-hoc
        callers; shares the same per-comment cache with `narrow_batch`.

    3) `identify_target_free(comment)`
        Used when the regex detected nothing. Performs retrieve-and-rerank
        across the full article corpus.

LLM answers are memoised on disk via `diskcache.Cache`, keyed on
(comment_text_hash, chapter_range_tuple, articles_signature), so that
re-parsing the same Excel against the same PDF skips any work that has
already been done - even across Streamlit restarts.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from diskcache import Cache
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from google.genai import errors as _genai_errors
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from models.models import Article, CommentTarget
from services.normalizer import normalize_for_comparison


_log_ident = logging.getLogger("legal_analyzer.ident")
_log_batch = logging.getLogger("legal_analyzer.batch")
_log_cache = logging.getLogger("legal_analyzer.cache")


# ---------------------------------------------------------------------------
# Structured-output schemas for Gemini
# ---------------------------------------------------------------------------


class _NarrowedNLIResult(BaseModel):
    """Output of the single-comment range-narrowing path: pick 1-3 articles
    within the chapter or declare the comment chapter-wide."""

    scope: Literal["article", "chapter_wide"] = Field(
        description=(
            "'article' when the comment targets one or more specific "
            "articles in the chapter; 'chapter_wide' when the comment "
            "addresses the chapter as a whole."
        )
    )
    article_numbers: List[str] = Field(
        default_factory=list,
        description=(
            "Between 1 and 3 article numbers from the provided chapter "
            "range when scope='article'. Must be empty when "
            "scope='chapter_wide'."
        ),
    )
    reasoning: str = Field(
        description=(
            "Short legal justification (1-3 sentences, in Greek) citing "
            "concrete parts of the comment and the chosen article(s)."
        )
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score in [0.0, 1.0] for the decision.",
    )


class _BatchedPerCommentResult(BaseModel):
    """One result entry inside a batched response from Gemini."""

    comment_index: int = Field(
        description=(
            "The 1-based index of the comment as presented in the prompt. "
            "Must exactly match the [N] marker of the corresponding comment."
        )
    )
    scope: Literal["article", "chapter_wide"]
    article_numbers: List[str] = Field(default_factory=list)
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class _BatchedNLIResult(BaseModel):
    """Output of the batched range-narrowing path: one entry per input comment."""

    results: List[_BatchedPerCommentResult] = Field(
        description=(
            "Array of classifications, one per input comment. The array "
            "length MUST equal the number of comments provided."
        )
    )


class _FreeNLIResult(BaseModel):
    """Output of the free-retrieval path: pick a single article from top-K."""

    article_number: str = Field(
        description=(
            "The article number the comment targets. Must match exactly "
            "one of the candidate article numbers. Use '' when none fits."
        )
    )
    reasoning: str = Field(
        description="Short legal justification (1-3 sentences, in Greek)."
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score in [0.0, 1.0]."
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


_NARROW_SYSTEM_PROMPT = (
    "Είσαι έμπειρος/η νομικός αναλυτής/τρια εξειδικευμένος/η στη δημόσια "
    "διαβούλευση ελληνικών νομοσχεδίων (opengov.gr). Κάθε σχόλιο "
    "αναρτάται κάτω από ένα ΚΕΦΑΛΑΙΟ του νομοσχεδίου, που περιέχει "
    "πολλά άρθρα· αυτό σημαίνει ότι η λίστα άρθρων που σου δίνεται "
    "είναι το εύρος του κεφαλαίου, ΟΧΙ δήλωση του πολίτη για τα "
    "συγκεκριμένα άρθρα που σχολιάζει.\n\n"
    "Δουλειά σου είναι να εντοπίσεις με ΝΟΜΙΚΟ ΣΥΛΛΟΓΙΣΜΟ ΠΑΝΩ ΣΤΟ "
    "ΠΕΡΙΕΧΟΜΕΝΟ ποιο ή ποια συγκεκριμένα άρθρα του κεφαλαίου στοχεύει "
    "πραγματικά κάθε σχόλιο.\n\n"
    "Κανόνες:\n"
    "- Επίλεξε ΑΥΣΤΗΡΑ από τους αριθμούς των υποψηφίων άρθρων.\n"
    "- Προτίμησε ΕΝΑ άρθρο όταν το σχόλιο εστιάζει σε ένα ζήτημα.\n"
    "- Επίστρεψε 2-3 άρθρα ΜΟΝΟ αν το σχόλιο ρητά θίγει πολλαπλά "
    "  ζητήματα που αντιστοιχούν σε διαφορετικά άρθρα.\n"
    "- Χαρακτήρισε το σχόλιο ως 'chapter_wide' ΜΟΝΟ αν αφορά τη γενική "
    "  κατεύθυνση / φιλοσοφία / σκοπιμότητα όλου του κεφαλαίου και δεν "
    "  μπορεί να εστιαστεί σε συγκεκριμένα άρθρα. Αυτή είναι η σπάνια "
    "  εξαίρεση, όχι ο κανόνας.\n"
    "- Το confidence_score να αντανακλά πόσο σαφής είναι η αντιστοίχιση."
)


_FREE_SYSTEM_PROMPT = (
    "Είσαι έμπειρος/η νομικός αναλυτής/τρια εξειδικευμένος/η στη δημόσια "
    "διαβούλευση ελληνικών νομοσχεδίων. Διαβάζεις ένα σχόλιο και ένα "
    "μικρό σύνολο υποψήφιων άρθρων (ανακτημένα μέσω semantic retrieval) "
    "και αποφασίζεις ποιο ΕΝΑ άρθρο στοχεύει πιο πιθανά το σχόλιο με "
    "βάση νομικό συλλογισμό. Επίλεξε αυστηρά από τους δοσμένους αριθμούς· "
    "αν κανένα δεν ταιριάζει, επέστρεψε κενό article_number και "
    "confidence 0.0."
)


# ---------------------------------------------------------------------------
# Persistent disk cache
# ---------------------------------------------------------------------------

# Cache value schema (dict, per cache key):
#   {"scope": "...", "article_numbers": [...], "reasoning": "...",
#    "confidence_score": 0.0}  # for narrow paths
#   {"article_number": "...", "reasoning": "...", "confidence_score": 0.0}
#                                                        # for free path
#
# Cache key schema: (comment_hash, candidate_numbers_tuple, articles_signature)
#
# Surviving Streamlit restarts means a user can iterate on the UI / prompt /
# UI code without paying LLM cost repeatedly for the same data.

# ---------------------------------------------------------------------------
# Transient-error handling: retries + model fallback
# ---------------------------------------------------------------------------

# Default model fallback chain. The first model is tried first; if it keeps
# returning transient errors (503/429/500) after retries, we move to the
# next. All models must share roughly the same capability level because the
# same prompts and schemas are reused.
_DEFAULT_CHAT_MODEL_CHAIN: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
]

# Exponential backoff: 1s, 2s, 4s, 8s between attempts (max 4 attempts).
_RETRY_MAX_ATTEMPTS = 4
_RETRY_WAIT = wait_exponential(multiplier=1, min=1, max=16)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True for errors worth retrying ON THE SAME MODEL.

    These are server-side / rate-limit errors that are almost always
    temporary: HTTP 5xx and 429. Client errors (400, 403, 404) are NOT
    transient and should not waste retries on the same model.
    """
    if isinstance(exc, _genai_errors.ServerError):
        return True
    if isinstance(exc, _genai_errors.ClientError):
        # `code` is the HTTP status carried on the exception.
        return getattr(exc, "code", None) == 429
    return False


def _is_model_unavailable_error(exc: BaseException) -> bool:
    """Return True for 404 "model not found / deprecated" errors.

    These are not worth retrying on the same model, but they ARE worth
    failing over to the next model in the chain. We also match on the
    error message because langchain_google_genai wraps the underlying
    `google.genai.errors.ClientError` in `ChatGoogleGenerativeAIError`,
    which loses the numeric status code.
    """
    if isinstance(exc, _genai_errors.ClientError):
        return getattr(exc, "code", None) == 404
    msg = str(exc).lower()
    return "not_found" in msg or "404" in msg or "no longer available" in msg


def _invoke_with_retry(
    llm: Any,
    messages: Any,
    *,
    context: str = "",
    model_name: str = "",
    stats: Optional[Dict[str, int]] = None,
) -> Any:
    """Call `llm.invoke(messages)` with exponential backoff on transient errors.

    Non-transient errors (bad request, malformed output, invalid API key) are
    raised immediately so the caller can react without wasting attempts.

    Args:
        llm: langchain chat-model wrapped in `with_structured_output(...)`.
        messages: prompt messages to pass to `llm.invoke`.
        context: human-readable context shown in retry logs (e.g.
            "chapter 1-10").
        model_name: the model this call is targeting; shown in logs.
        stats: optional dict to increment counters into. The function
            writes to `"llm_retries"` every time a retry sleep fires.
    """

    def _on_retry(retry_state: Any) -> None:
        if stats is not None:
            stats["llm_retries"] = stats.get("llm_retries", 0) + 1
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        sleep_s = retry_state.next_action.sleep if retry_state.next_action else 0.0
        _log_batch.warning(
            "%s: %s failed (%s), retry %d/%d in %.1fs",
            context or "llm",
            model_name or "?",
            type(exc).__name__ if exc else "UnknownError",
            retry_state.attempt_number,
            _RETRY_MAX_ATTEMPTS,
            sleep_s,
        )

    retrying = Retrying(
        reraise=True,
        stop=stop_after_attempt(_RETRY_MAX_ATTEMPTS),
        wait=_RETRY_WAIT,
        retry=retry_if_exception(_is_transient_error),
        before_sleep=_on_retry,
    )
    for attempt in retrying:
        with attempt:
            return llm.invoke(messages)
    # Should be unreachable because `reraise=True` guarantees either a return
    # value or a raised exception, but keeps mypy happy.
    raise RuntimeError("Retrying exited without a result")


_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "legal_analyzer" / "llm_cache"
_CACHE_DIR = Path(os.getenv("LEGAL_ANALYZER_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_LLM_CACHE: Cache = Cache(str(_CACHE_DIR))


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def clear_llm_cache() -> None:
    """Wipe the on-disk LLM cache. Useful from a Streamlit admin control."""
    _LLM_CACHE.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_articles(articles: List) -> List[Article]:
    """Accept either a list of dicts (as produced by `split_text`) or a
    list of `Article` dataclasses, and normalize to `Article` objects."""
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
    """Render a list of articles as a compact, labelled prompt block."""
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


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------


class TargetIdentifier:
    """
    Stateful helper that pre-indexes the article corpus once and serves
    per-comment targeting queries. Keeping this as a class avoids
    re-embedding the same articles for every comment (hundreds of comments
    would otherwise trigger hundreds of redundant embedding calls).
    """

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

        # Build a fallback chain of chat models. If the primary model is
        # throwing 503/429 after retries, we transparently fail over to the
        # next one. The same prompts/schemas work across all of them.
        self._chat_models: List[str] = list(chat_models or _DEFAULT_CHAT_MODEL_CHAIN)
        self._embeddings_model: str = embeddings_model
        self._api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        # The embeddings client is instantiated lazily (see
        # `_get_embeddings_client`) so that a fully-cached corpus does not
        # require an API key at all.
        self._embeddings: Optional[GoogleGenerativeAIEmbeddings] = None

        self._llm_chain: List[ChatGoogleGenerativeAI] = [
            ChatGoogleGenerativeAI(model=name, api_key=self._api_key)
            for name in self._chat_models
        ]
        self._narrow_chain = [
            llm.with_structured_output(_NarrowedNLIResult) for llm in self._llm_chain
        ]
        self._batch_chain = [
            llm.with_structured_output(_BatchedNLIResult) for llm in self._llm_chain
        ]
        self._free_chain = [
            llm.with_structured_output(_FreeNLIResult) for llm in self._llm_chain
        ]

        # Lightweight counters surfaced at the end of every parse via the
        # summary log. Reset implicitly on each new TargetIdentifier because
        # the dict is instance-scoped.
        self.stats: Dict[str, int] = {
            "cache_hits": 0,
            "embeddings_cache_hit": 0,
            "llm_calls_ok": 0,
            "llm_retries": 0,
            "model_fallbacks": 0,
            "dropped_invalid_numbers": 0,
            "empty_llm_results": 0,
        }

        _log_ident.info(
            "Init TargetIdentifier: %d articles, model chain = %s",
            len(self.articles),
            self._chat_models,
        )

        # Stable signature over the article corpus, used for cache keying
        # of both LLM answers and the corpus embeddings matrix. Different
        # PDFs (or article edits) invalidate cached entries automatically.
        signature_material = "||".join(
            f"{a.article_number}|{a.title}|{a.body}" for a in self.articles
        )
        self.articles_signature: str = _hash_text(signature_material)

        self._corpus_embeddings = self._load_or_build_corpus_embeddings()

    # ------------------------------------------------------------------
    # Embeddings (lazy client + on-disk corpus cache)
    # ------------------------------------------------------------------

    def _get_embeddings_client(self) -> GoogleGenerativeAIEmbeddings:
        """Instantiate the embeddings client on first use.

        Deferring the pydantic-validated construction here means that a
        fully-cached corpus (no free-path queries) never needs a valid
        API key at all.
        """
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self._embeddings_model,
                api_key=self._api_key,
            )
        return self._embeddings

    def _corpus_embeddings_cache_key(self) -> str:
        """Cache key for the full corpus embedding matrix."""
        return (
            f"corpus_embeddings::{self._embeddings_model}::"
            f"{self.articles_signature}"
        )

    def _load_or_build_corpus_embeddings(self) -> np.ndarray:
        """Return the corpus embedding matrix, loading from disk if possible."""
        if not self.articles:
            return np.zeros((0, 0))

        cache_key = self._corpus_embeddings_cache_key()
        cached = _LLM_CACHE.get(cache_key)
        if cached is not None and isinstance(cached, np.ndarray):
            self.stats["embeddings_cache_hit"] = 1
            _log_cache.info(
                "Corpus embeddings cache HIT (sig %s, shape %s)",
                self.articles_signature[:8],
                cached.shape,
            )
            return cached

        _log_cache.info(
            "Corpus embeddings cache MISS — embedding %d articles...",
            len(self.articles),
        )
        t_start = time.perf_counter()
        corpus_texts = [
            normalize_for_comparison(f"{a.title} {a.body}") for a in self.articles
        ]
        matrix = np.array(
            self._get_embeddings_client().embed_documents(corpus_texts)
        )
        _LLM_CACHE[cache_key] = matrix
        _log_ident.info(
            "Corpus embedded & cached (%d articles, sig %s, %.1fs)",
            len(self.articles),
            self.articles_signature[:8],
            time.perf_counter() - t_start,
        )
        return matrix

    # ------------------------------------------------------------------
    # LLM invocation with retries + model fallback
    # ------------------------------------------------------------------

    def _invoke_with_fallback(
        self,
        chain: List[Any],
        messages: Any,
        *,
        label: str = "llm",
    ) -> Any:
        """Invoke `messages` against the first healthy model in `chain`.

        Each model is tried with exponential backoff (see `_invoke_with_retry`).
        If a model still returns a transient error after exhausting retries
        and there is another model available, we fail over to it. A
        non-transient error is raised immediately.
        """
        last_exc: Optional[BaseException] = None
        for idx, llm in enumerate(chain):
            model_name = self._chat_models[idx]
            try:
                return _invoke_with_retry(
                    llm,
                    messages,
                    context=label,
                    model_name=model_name,
                    stats=self.stats,
                )
            except Exception as exc:
                last_exc = exc
                has_next = idx < len(chain) - 1
                # Fail over to the next model both for transient (overloaded,
                # rate-limited) errors AND for "model not found / deprecated"
                # 404s — if this specific model id is gone, the next one in
                # the chain might still be alive.
                should_fallback = (
                    _is_transient_error(exc)
                    or _is_model_unavailable_error(exc)
                )
                if should_fallback and has_next:
                    next_model = self._chat_models[idx + 1]
                    reason = (
                        "transient"
                        if _is_transient_error(exc)
                        else "unavailable"
                    )
                    self.stats["model_fallbacks"] += 1
                    _log_batch.warning(
                        "%s: %s error on %s (%s) → falling back to %s",
                        label,
                        reason,
                        model_name,
                        type(exc).__name__,
                        next_model,
                    )
                    continue
                raise
        assert last_exc is not None  # defensive; the loop always raises or returns
        raise last_exc

    # ------------------------------------------------------------------
    # Retrieval (used by the free path)
    # ------------------------------------------------------------------

    def get_top_k_candidates(
        self, comment_text: str, k: int = 5
    ) -> List[Tuple[Article, float]]:
        """Return the top-K most semantically similar articles for a comment."""
        if not self.articles or self._corpus_embeddings.size == 0:
            return []

        normalized = normalize_for_comparison(comment_text or "")
        if not normalized:
            return []

        query_embedding = np.array(
            self._get_embeddings_client().embed_query(normalized)
        ).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self._corpus_embeddings)[0]

        k = min(k, len(self.articles))
        top_indices = np.argsort(similarities)[::-1][:k]
        return [
            (self.articles[int(idx)], float(similarities[int(idx)]))
            for idx in top_indices
        ]

    # ------------------------------------------------------------------
    # Path 1a - Range narrowing (single comment)
    # ------------------------------------------------------------------

    def narrow_within_range(
        self,
        comment_text: str,
        chapter_article_numbers: List[str],
    ) -> List[CommentTarget]:
        """Run range narrowing for a single comment. Used by ad-hoc callers
        and tests; production calls go through `narrow_batch` for efficiency.

        Returns a list of `CommentTarget` objects:
          * 1-3 `ai_nli_narrowed` targets when the LLM identifies specific
            articles.
          * a single `ai_nli_narrowed` target with `scope='chapter_wide'` and
            empty `article_number` when the LLM judges the comment to span
            the whole chapter.
          * an empty list when no valid candidates exist.
        """
        chapter_articles = [
            self._articles_by_number[num]
            for num in chapter_article_numbers
            if num in self._articles_by_number
        ]
        if not chapter_articles or not (comment_text or "").strip():
            return []

        valid_numbers = [a.article_number for a in chapter_articles]
        cache_key = self._cache_key(comment_text, valid_numbers)

        cached = _LLM_CACHE.get(cache_key)
        if cached is None:
            candidates_block = _format_candidates_block(chapter_articles)
            user_prompt = (
                "Σχόλιο πολίτη:\n"
                f"\"\"\"\n{comment_text.strip()}\n\"\"\"\n\n"
                "Άρθρα του κεφαλαίου όπου αναρτήθηκε το σχόλιο:\n"
                f"{candidates_block}\n\n"
                "Ποιο ή ποια από τα παραπάνω άρθρα στοχεύει πραγματικά "
                "το σχόλιο; Αν το σχόλιο είναι γενικό για το κεφάλαιο, "
                "δώσε scope='chapter_wide'."
            )
            result: _NarrowedNLIResult = self._invoke_with_fallback(
                self._narrow_chain,
                [
                    ("system", _NARROW_SYSTEM_PROMPT),
                    ("human", user_prompt),
                ],
                label="narrow_within_range",
            )
            cached = result.model_dump()
            _LLM_CACHE[cache_key] = cached

        return self._narrow_result_to_targets(cached, valid_numbers)

    # ------------------------------------------------------------------
    # Path 1b - Range narrowing (batched by chapter)
    # ------------------------------------------------------------------

    def narrow_batch(
        self,
        items: List[Tuple[str, str]],
        chapter_article_numbers: List[str],
    ) -> Dict[str, List[CommentTarget]]:
        """Classify many comments that share the same chapter in as few LLM
        calls as possible. Cache hits are served individually; the remaining
        misses go in a single batched LLM call.

        Args:
            items: List of (comment_id, comment_text) tuples. The id is an
                   opaque string used only to key the return mapping; it is
                   NOT included in the cache key (text+chapter+corpus is).
            chapter_article_numbers: The article numbers of the chapter the
                   comments were posted under.

        Returns:
            Dict mapping each comment_id to its resolved `CommentTarget` list
            (possibly empty).
        """
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

        # Pass 1: drain the cache per comment.
        pending: List[Tuple[str, str]] = []
        for cid, text in items:
            if not (text or "").strip():
                results[cid] = []
                continue
            cached = _LLM_CACHE.get(self._cache_key(text, valid_numbers))
            if cached is not None:
                self.stats["cache_hits"] += 1
                results[cid] = self._narrow_result_to_targets(cached, valid_numbers)
            else:
                pending.append((cid, text))

        _log_batch.info(
            "%s (%d comments): cache %d hit / %d miss",
            chapter_tag,
            len(items),
            len(items) - len(pending),
            len(pending),
        )

        if not pending:
            _log_batch.info("%s: fully cached → no LLM call", chapter_tag)
            return results

        # Pass 2: batched LLM call for cache misses.
        candidates_block = _format_candidates_block(chapter_articles)
        comment_lines = [
            f"[{idx}] {text.strip()}" for idx, (_, text) in enumerate(pending, start=1)
        ]
        comments_block = "\n\n".join(comment_lines)
        user_prompt = (
            "Άρθρα του κεφαλαίου όπου αναρτήθηκαν τα σχόλια:\n"
            f"{candidates_block}\n\n"
            "Σχόλια πολιτών (αναγνωριστικά [N]):\n"
            f"{comments_block}\n\n"
            "Για ΚΑΘΕ σχόλιο [N] επέστρεψε ένα αντικείμενο με: "
            "comment_index=N, scope, article_numbers (από τα παραπάνω άρθρα, "
            "1-3 στοιχεία), reasoning, confidence_score. Η λίστα results "
            "πρέπει να έχει ακριβώς "
            f"{len(pending)} στοιχεία."
        )

        _log_batch.info(
            "%s: calling %s for %d comments...",
            chapter_tag,
            self._chat_models[0],
            len(pending),
        )
        t_start = time.perf_counter()
        try:
            batched: _BatchedNLIResult = self._invoke_with_fallback(
                self._batch_chain,
                [
                    ("system", _NARROW_SYSTEM_PROMPT),
                    ("human", user_prompt),
                ],
                label=chapter_tag,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            _log_batch.error(
                "%s: batched call FAILED after %.1fs — %s: %s",
                chapter_tag,
                elapsed,
                type(exc).__name__,
                exc,
            )
            # If the error is transient OR every model in the chain is
            # unavailable (404), per-comment calls would fail for the same
            # reason. Bail out cleanly instead of burning more quota/time.
            if _is_transient_error(exc) or _is_model_unavailable_error(exc):
                _log_batch.error(
                    "%s: skipping per-comment fallback (whole model chain "
                    "is unavailable). Try again later.",
                    chapter_tag,
                )
                for cid, _ in pending:
                    results[cid] = []
                return results

            # Non-transient (e.g. malformed structured output): per-comment
            # calls might succeed with a simpler schema / shorter prompt.
            _log_batch.warning(
                "%s: non-transient error, falling back to per-comment calls",
                chapter_tag,
            )
            for cid, text in pending:
                try:
                    results[cid] = self.narrow_within_range(
                        text, chapter_article_numbers
                    )
                except Exception as inner_exc:
                    _log_batch.error(
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
        _log_batch.info(
            "%s: OK → %d/%d results (%.1fs)",
            chapter_tag,
            len(batched.results),
            len(pending),
            elapsed,
        )

        by_index = {r.comment_index: r for r in batched.results}
        missing_indices: List[int] = []
        specific = chapter_wide = empty = 0

        # Map results back and cache each one individually for future re-use.
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
            _LLM_CACHE[self._cache_key(text, valid_numbers)] = payload
            mapped = self._narrow_result_to_targets(payload, valid_numbers)
            results[cid] = mapped
            if not mapped:
                empty += 1
            elif mapped[0].scope == "chapter_wide":
                chapter_wide += 1
            else:
                specific += 1

        _log_batch.info(
            "%s: mapped → %d specific, %d chapter_wide, %d empty",
            chapter_tag,
            specific,
            chapter_wide,
            empty,
        )

        if missing_indices:
            _log_batch.warning(
                "%s: LLM skipped entries for indices %s",
                chapter_tag,
                missing_indices,
            )

        return results

    # ------------------------------------------------------------------
    # Path 2 - Free retrieval over the full corpus
    # ------------------------------------------------------------------

    def identify_target_free(
        self, comment_text: str, k: int = 5
    ) -> Optional[CommentTarget]:
        """Run retrieval + NLI when no regex hint is available."""
        if not (comment_text or "").strip():
            return None

        candidates = self.get_top_k_candidates(comment_text, k=k)
        if not candidates:
            return None

        valid_numbers = [a.article_number for a, _ in candidates]
        cache_key = self._cache_key(comment_text, valid_numbers)

        cached = _LLM_CACHE.get(cache_key)
        if cached is None:
            candidates_block = _format_candidates_block([a for a, _ in candidates])
            user_prompt = (
                "Σχόλιο πολίτη:\n"
                f"\"\"\"\n{comment_text.strip()}\n\"\"\"\n\n"
                "Υποψήφια άρθρα (top-k από semantic retrieval):\n"
                f"{candidates_block}\n\n"
                "Ποιο από τα παραπάνω άρθρα στοχεύει το σχόλιο;"
            )
            result: _FreeNLIResult = self._invoke_with_fallback(
                self._free_chain,
                [
                    ("system", _FREE_SYSTEM_PROMPT),
                    ("human", user_prompt),
                ],
                label="identify_target_free",
            )
            cached = result.model_dump()
            _LLM_CACHE[cache_key] = cached

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

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _cache_key(
        self, comment_text: str, candidate_numbers: List[str]
    ) -> Tuple[str, Tuple[str, ...], str]:
        """Compose the stable cache key for a (comment, candidates) pair."""
        return (
            _hash_text(comment_text),
            tuple(candidate_numbers),
            self.articles_signature,
        )

    @staticmethod
    def _narrow_result_to_targets(
        result: Dict[str, Any],
        valid_numbers: List[str],
    ) -> List[CommentTarget]:
        """Convert a cached narrowing payload to `CommentTarget`s, defensively
        validating article numbers against the chapter range."""
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
        # Keep only numbers that actually belong to the chapter, preserve
        # order from the LLM, cap at 3.
        chosen = [n for n in raw_numbers if n in valid_set][:3]

        if not chosen:
            if raw_numbers:
                _log_batch.warning(
                    "dropped LLM numbers %s (none in valid set %s, scope=%s)",
                    raw_numbers,
                    valid_numbers,
                    scope,
                )
            else:
                _log_batch.debug(
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
