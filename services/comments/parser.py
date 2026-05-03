"""Reads an opengov.gr Excel export and maps every comment to the
article(s) it targets.

Decision logic per row (in order):

    1) Regex returns NOTHING           -> free AI path over the full corpus.
    2) Regex returns exactly ONE       -> trust it (`method="regex"`).
    3) Regex returns a RANGE of >= 2   -> treat it as chapter metadata and
                                          let the Legal NLI narrow it down
                                          to 1-3 specific articles (or flag
                                          the comment as chapter-wide).

Case (3) is the common one, because on opengov.gr consultations are
typically posted per chapter rather than per article. To amortise the LLM
cost we GROUP all case-(3) comments that share the same chapter range and
classify them in a single batched LLM call, re-using the chapter context.
"""

import logging
import re
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from models.models import Comment, CommentTarget
from services.target_identification import TargetIdentifier


_log = logging.getLogger("legal_analyzer.parser")

ProgressCallback = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Regex extraction
# ---------------------------------------------------------------------------


def extract_article_range(section_text: str) -> List[str]:
    """
    Extract article numbers from the "ΑΡΘΡΟ" cell as strings, so they can
    match the `article_number` field of our Article model.

    Examples:
      "(άρθρα 2 - 5)" -> ["2", "3", "4", "5"]
      "(άρθρο 3)"     -> ["3"]
    """
    if not section_text or pd.isna(section_text):
        return []

    text = str(section_text).strip()

    # `άρθρ\w+` tolerates any Greek declension that follows: άρθρο (sg. n/a),
    # άρθρα (pl.), άρθρου (sg. gen), άρθρων (pl. gen), etc. Python's default
    # Unicode-aware `\w` matches Greek letters.
    range_match = re.search(
        r'\(άρθρ\w+\s*(\d+)\s*-\s*(\d+)\)', text, re.IGNORECASE
    )
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        return [str(i) for i in range(start, end + 1)]

    single_match = re.search(r'\(άρθρ\w+\s*(\d+)\)', text, re.IGNORECASE)
    if single_match:
        return [single_match.group(1)]

    return []


# ---------------------------------------------------------------------------
# Row classification
# ---------------------------------------------------------------------------


class _Row:
    """Lightweight record of a classified Excel row, used internally."""

    __slots__ = ("comment_id", "body", "raw_article_text", "regex_matches", "targets")

    def __init__(
        self,
        comment_id: str,
        body: str,
        raw_article_text: str,
        regex_matches: List[str],
    ) -> None:
        self.comment_id = comment_id
        self.body = body
        self.raw_article_text = raw_article_text
        self.regex_matches = regex_matches
        self.targets: List[CommentTarget] = []


def _read_rows(excel_file) -> List[_Row]:
    """First pass: read the Excel and produce classified rows with regex
    information only (no LLM calls yet)."""
    df = pd.read_excel(excel_file)
    rows: List[_Row] = []

    for _, row in df.iterrows():
        comment_id = row.get("ΚΩΔΙΚΟΣ ΣΧΟΛΙΟΥ")
        article_text = row.get("ΑΡΘΡΟ")
        comment_body = row.get("ΣΧΟΛΙΟ")

        if pd.isna(comment_body):
            comment_body = ""
        comment_body = str(comment_body)

        raw_article_text = "" if pd.isna(article_text) else str(article_text)
        regex_matches = extract_article_range(article_text)
        comment_id_str = str(comment_id) if comment_id is not None else ""

        rows.append(
            _Row(
                comment_id=comment_id_str,
                body=comment_body,
                raw_article_text=raw_article_text,
                regex_matches=regex_matches,
            )
        )

    return rows


def _apply_single_regex_targets(row: _Row) -> None:
    """Resolve the trivial single-regex case directly, with no LLM."""
    row.targets = [
        CommentTarget(
            article_number=row.regex_matches[0],
            method="regex",
            scope="article",
            chapter_range=[],
            reasoning="Εντοπίστηκε ρητή αναφορά στη στήλη «ΑΡΘΡΟ».",
            confidence_score=1.0,
        )
    ]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def parse_comments_excel(
    excel_file,
    articles: Optional[List] = None,
    ai_top_k: int = 5,
    progress_cb: Optional[ProgressCallback] = None,
) -> List[Comment]:
    """Parse the Excel file and return enriched `Comment` objects.

    The AI-powered paths group by chapter range so that all comments of a
    given chapter share the same LLM call whenever possible. LLM answers
    are memoised on disk, so re-running with unchanged inputs is free.

    Args:
        excel_file:  File-like object or path to the `.xlsx` file.
        articles:    Optional list of Article objects (or dicts) used to
                     power the AI paths. If omitted, only regex-based
                     extraction is performed.
        ai_top_k:    Number of candidate articles passed to the LLM in the
                     free (no-regex) path.
        progress_cb: Optional callback invoked as
                     ``progress_cb(done, total, label)`` while classifying.
    """
    t_start = time.perf_counter()
    _log.info("Reading Excel...")
    rows = _read_rows(excel_file)
    total_rows = len(rows)
    done = 0

    def _tick(label: str, increment: int = 1) -> None:
        nonlocal done
        done += increment
        if progress_cb is not None:
            progress_cb(done, total_rows, label)

    # Resolve single-regex and no-regex/no-identifier cases synchronously.
    # Group range rows by their chapter tuple to enable batching.
    range_groups: "OrderedDict[Tuple[str, ...], List[_Row]]" = OrderedDict()
    free_rows: List[_Row] = []
    single_regex_count = 0

    for row in rows:
        if len(row.regex_matches) == 1:
            _apply_single_regex_targets(row)
            single_regex_count += 1
            _tick(f"Regex match για {row.comment_id}")
        elif len(row.regex_matches) >= 2:
            key = tuple(row.regex_matches)
            range_groups.setdefault(key, []).append(row)
        else:
            free_rows.append(row)

    range_comments = sum(len(v) for v in range_groups.values())
    _log.info(
        "Read %d rows → %d single-regex, %d range (%d chapter groups), %d free",
        total_rows,
        single_regex_count,
        range_comments,
        len(range_groups),
        len(free_rows),
    )

    identifier: Optional[TargetIdentifier] = None
    if articles and (range_groups or free_rows):
        try:
            identifier = TargetIdentifier(articles)
        except Exception as exc:
            # Degrade gracefully to regex-only rather than crashing the whole
            # batch when the API key is missing / invalid / quota-exhausted.
            identifier = None
            _log.error(
                "TargetIdentifier init FAILED — %s: %s. "
                "Falling back to regex-only mode.",
                type(exc).__name__,
                exc,
            )

    # Process the range groups: one batched LLM call per chapter.
    for chapter_range, group_rows in range_groups.items():
        if identifier is None:
            for row in group_rows:
                row.targets = []
                _tick(f"Χωρίς AI για {row.comment_id}")
            continue

        items = [(row.comment_id, row.body) for row in group_rows]
        try:
            batch_results = identifier.narrow_batch(items, list(chapter_range))
        except Exception as exc:
            _log.error(
                "chapter %s-%s: narrow_batch raised %s: %s",
                chapter_range[0],
                chapter_range[-1],
                type(exc).__name__,
                exc,
            )
            batch_results = {cid: [] for cid, _ in items}

        for row in group_rows:
            row.targets = batch_results.get(row.comment_id, [])
            _tick(
                f"Κεφάλαιο {chapter_range[0]}-{chapter_range[-1]}: "
                f"σχόλιο {row.comment_id}"
            )

    # Process the free (no-regex) rows one by one.
    for row in free_rows:
        if identifier is None or not row.body.strip():
            row.targets = []
            _tick(f"Χωρίς AI για {row.comment_id}")
            continue
        try:
            free = identifier.identify_target_free(row.body, k=ai_top_k)
        except Exception as exc:
            _log.error(
                "free retrieval failed for comment %s — %s: %s",
                row.comment_id,
                type(exc).__name__,
                exc,
            )
            free = None
        row.targets = [free] if free is not None else []
        _tick(f"Ελεύθερο retrieval για {row.comment_id}")

    parsed = [
        Comment(
            comment_id=row.comment_id,
            comment=row.body,
            targets=row.targets,
            raw_article_text=row.raw_article_text,
        )
        for row in rows
    ]

    _log_summary(parsed, identifier, time.perf_counter() - t_start)
    return parsed


# ---------------------------------------------------------------------------
# Summary logging
# ---------------------------------------------------------------------------


def _log_summary(
    comments: List[Comment],
    identifier: Optional[TargetIdentifier],
    elapsed: float,
) -> None:
    """Emit the final PARSE SUMMARY block that aggregates per-comment stats
    and LLM activity counters into a single, easy-to-scan report."""
    total = len(comments)
    with_targets = sum(1 for c in comments if c.targets)
    by_method: Dict[str, int] = {}
    by_scope: Dict[str, int] = {}
    for c in comments:
        for t in c.targets:
            by_method[t.method] = by_method.get(t.method, 0) + 1
            by_scope[t.scope] = by_scope.get(t.scope, 0) + 1

    stats = identifier.stats if identifier is not None else {}
    bar = "═" * 54

    _log.info(bar)
    _log.info("PARSE SUMMARY")
    _log.info(bar)
    _log.info("Total comments             : %d", total)
    pct = (100.0 * with_targets / total) if total else 0.0
    _log.info("Comments with any target   : %d (%.1f%%)", with_targets, pct)
    _log.info("Comments with empty targets: %d", total - with_targets)
    _log.info("")
    _log.info("Targets by method:")
    for method in ("regex", "ai_nli_narrowed", "ai_nli"):
        _log.info("  %-22s : %d", method, by_method.get(method, 0))
    _log.info("")
    _log.info("Targets by scope:")
    for scope in ("article", "chapter_wide"):
        _log.info("  %-22s : %d", scope, by_scope.get(scope, 0))
    _log.info("")
    _log.info("LLM activity:")
    _log.info(
        "  %-22s : %s",
        "corpus embeddings",
        "from cache" if stats.get("embeddings_cache_hit") else "recomputed",
    )
    _log.info("  %-22s : %d", "cache hits", stats.get("cache_hits", 0))
    _log.info("  %-22s : %d", "successful LLM calls", stats.get("llm_calls_ok", 0))
    _log.info("  %-22s : %d", "retries", stats.get("llm_retries", 0))
    _log.info("  %-22s : %d", "model fallbacks", stats.get("model_fallbacks", 0))
    _log.info("  %-22s : %.1fs", "total elapsed", elapsed)
    _log.info(bar)
