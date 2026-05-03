"""Application services grouped by domain.

- ``services.comparison`` — matching old vs new articles, diff, significance.
- ``services.documents`` — PDF load + split into article dicts.
- ``services.llm`` — disk cache + Gemini embeddings wrapper.
- ``services.comments`` — Excel consultation export parsing.
- ``services.target_identification`` — LLM routing from comments to articles.
- ``services.infra`` — process-wide logging setup.
- ``services.utils`` — tiny shared helpers (e.g. article number sort keys).
"""
