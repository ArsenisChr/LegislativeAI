"""Normalize upstream article representations and render LLM context blocks."""

from __future__ import annotations

from typing import List

from models.models import Article


def coerce_articles(articles: List) -> List[Article]:
    """Accept `Article` instances or dicts shaped like split-text output."""
    out: List[Article] = []
    for item in articles:
        if isinstance(item, Article):
            out.append(item)
        elif isinstance(item, dict):
            out.append(
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
    return out


def format_candidates_block(candidates: List[Article]) -> str:
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
