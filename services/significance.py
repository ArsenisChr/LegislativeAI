from models.models import ChangeType, Article, DiffSegment
from typing import List

def classify_change(old_article: Article, new_article: Article, segments: List[DiffSegment]) -> ChangeType:
    # A change is significant if there is an insert or delete that is not just whitespace
    has_changes = False
    for seg in segments:
        if seg.operation in ["insert", "delete"]:
            # Only consider it a change if it contains non-whitespace characters
            if seg.text.strip():
                has_changes = True
                break
                
    is_renumbered = old_article.article_number != new_article.article_number
    
    if not has_changes and not is_renumbered:
        return ChangeType.UNCHANGED
    elif not has_changes and is_renumbered:
        return ChangeType.RENUMBERED
    elif has_changes and is_renumbered:
        return ChangeType.RENUMBERED_MODIFIED
    else:
        return ChangeType.MODIFIED
