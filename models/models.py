from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class ChangeType(Enum):
    UNCHANGED = "unchanged"
    MODIFIED = "modified"
    RENUMBERED = "renumbered"
    RENUMBERED_MODIFIED = "renumbered_modified"
    ADDED = "added"
    REMOVED = "removed"

@dataclass
class Article:
    article_number: str
    header: str
    title: str
    body: str

@dataclass
class DiffSegment:
    operation: str  # e.g., "insert", "delete", "equal", "replace"
    text: str

@dataclass
class ArticleDiff:
    old_article: Optional[Article]
    new_article: Optional[Article]
    change_type: ChangeType
    similarity_score: float = 0.0
    segments: List[DiffSegment] = field(default_factory=list)

@dataclass
class CommentTarget:
    """
    Targeting result linking a comment to a specific article (or to a whole
    chapter, when the comment is not specific to any single article).

    Fields:
    - article_number: The targeted article number; empty string when
      `scope == "chapter_wide"`.
    - method:
        * "regex"            -> a single explicit reference in the "ΑΡΘΡΟ" cell
                                (e.g. "(άρθρο 7)").
        * "ai_nli_narrowed"  -> target chosen by the LLM within the chapter
                                range declared in the "ΑΡΘΡΟ" cell.
        * "ai_nli"           -> target chosen by the LLM across the full
                                article corpus (regex produced nothing).
    - scope:
        * "article"       -> the target is a specific article.
        * "chapter_wide"  -> the comment addresses the chapter as a whole and
                             must be surfaced at the chapter level, not per
                             article.
    - chapter_range: Article numbers of the chapter the comment was posted
      under (if any). Used by the UI to render chapter-wide banners and to
      annotate narrowed targets.
    """
    article_number: str
    method: str
    scope: str = "article"
    chapter_range: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence_score: float = 1.0

@dataclass
class Comment:
    """
    Representation of a public consultation comment from opengov.gr,
    enriched with the list of articles it targets. A single comment may
    target 0, 1 or multiple articles (e.g. when referring to a range).
    """
    comment_id: str
    comment: str
    targets: List[CommentTarget] = field(default_factory=list)
    raw_article_text: str = ""
