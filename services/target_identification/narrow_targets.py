"""Map structured LLM outputs to `CommentTarget` rows."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from models.models import CommentTarget

_log = logging.getLogger("legal_analyzer.batch")


def narrow_dict_to_targets(
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
            _log.warning(
                "dropped LLM numbers %s (none in valid set %s, scope=%s)",
                raw_numbers,
                valid_numbers,
                scope,
            )
        else:
            _log.debug(
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
