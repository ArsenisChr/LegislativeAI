import re
from difflib import SequenceMatcher
from typing import List

from models.models import DiffSegment


def compute_diff(old_text: str, new_text: str) -> List[DiffSegment]:
    """Word-level diff between two texts as `DiffSegment` rows."""
    old_tokens = re.split(r"(\s+)", old_text)
    new_tokens = re.split(r"(\s+)", new_text)

    old_tokens = [t for t in old_tokens if t]
    new_tokens = [t for t in new_tokens if t]

    matcher = SequenceMatcher(None, old_tokens, new_tokens)
    segments = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            segments.append(DiffSegment(operation="equal", text="".join(old_tokens[i1:i2])))
        elif tag == "replace":
            segments.append(DiffSegment(operation="delete", text="".join(old_tokens[i1:i2])))
            segments.append(DiffSegment(operation="insert", text="".join(new_tokens[j1:j2])))
        elif tag == "delete":
            segments.append(DiffSegment(operation="delete", text="".join(old_tokens[i1:i2])))
        elif tag == "insert":
            segments.append(DiffSegment(operation="insert", text="".join(new_tokens[j1:j2])))

    return segments
