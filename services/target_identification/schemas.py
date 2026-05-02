from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class NarrowedNLIResult(BaseModel):
    """Output of the single-comment range-narrowing path."""

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


class BatchedPerCommentResult(BaseModel):
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


class BatchedNLIResult(BaseModel):
    """Output of the batched range-narrowing path."""

    results: List[BatchedPerCommentResult] = Field(
        description=(
            "Array of classifications, one per input comment. The array "
            "length MUST equal the number of comments provided."
        )
    )


class FreeNLIResult(BaseModel):
    """Output of the free-retrieval path."""

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
