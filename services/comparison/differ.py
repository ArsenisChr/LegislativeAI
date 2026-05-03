import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Tuple

from models.models import DiffSegment


_DIFF_PUNCT_RE = re.compile(r"[.,;:]")
_DIFF_WS_RE = re.compile(r"\s+")
_QUOTE_VARIANTS_RE = re.compile(r"[΄’'᾽`´]")


def normalize_for_diff(text: str) -> str:
    """Light cleanup mode used by optional normalized comparison."""
    cleaned = text or ""
    # Unify apostrophe/tonos-like characters frequently found in Greek legal text.
    cleaned = _QUOTE_VARIANTS_RE.sub("'", cleaned)
    # Remove Greek accents/diacritics so e.g. α΄ and α' compare equally.
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = "".join(c for c in cleaned if not unicodedata.combining(c))
    cleaned = _DIFF_PUNCT_RE.sub("", cleaned)
    cleaned = cleaned.lower()
    cleaned = _DIFF_WS_RE.sub(" ", cleaned).strip()
    return cleaned


def compute_diff_tokens(
    old_text: str,
    new_text: str,
    *,
    normalize_text: bool = False,
) -> Tuple[List[DiffSegment], float]:
    """Token-level diff and local edit magnitude on the same token lists.

    Tokens are produced by splitting on whitespace while keeping whitespace runs,
    matching `difflib.SequenceMatcher` granularity used for the coloured diff UI.

    Returns:
        segments: Insert/delete/equal fragments for rendering.
        token_change_fraction: ``1 - matcher.ratio()`` in ``[0, 1]`` (0 = identical sequences).
    """
    if normalize_text:
        old_text = normalize_for_diff(old_text)
        new_text = normalize_for_diff(new_text)

    old_tokens = re.split(r"(\s+)", old_text)
    new_tokens = re.split(r"(\s+)", new_text)

    old_tokens = [t for t in old_tokens if t]
    new_tokens = [t for t in new_tokens if t]

    matcher = SequenceMatcher(None, old_tokens, new_tokens)
    ratio = matcher.ratio() if (old_tokens or new_tokens) else 1.0
    token_change_fraction = max(0.0, min(1.0, 1.0 - ratio))

    segments: List[DiffSegment] = []

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

    return segments, token_change_fraction


def compute_diff(old_text: str, new_text: str) -> List[DiffSegment]:
    """Word-level diff between two texts (segments only); see `compute_diff_tokens` for metrics."""
    segments, _fraction = compute_diff_tokens(old_text, new_text)
    return segments
