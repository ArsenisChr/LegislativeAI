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
