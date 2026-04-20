# Legislative AI Analyzer

![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![Langchain](https://img.shields.io/badge/Langchain-Framework-1C3C3C?logo=langchain)
![Gemini AI](https://img.shields.io/badge/Gemini-Embeddings-8E75B2?logo=google)

An advanced, AI-powered tool designed to analyze, parse, and compare complex Greek legislative documents (PDFs). This tool acts as an automated legal assistant, extracting articles from complex legal PDFs, intelligently aligning previous and new versions of laws, and generating actionable, colored diffs and legal impact assessments.

## 🚀 Core Features & Technical Highlights

This project demonstrates strong capabilities in **Natural Language Processing (NLP)**, **Document Parsing**, and **System Architecture**.

### 1. Robust PDF Extraction & Parsing (`pdfplumber` + Custom Heuristics)
Legal PDFs in Greece (ΦΕΚ) are notoriously difficult to parse due to multi-column layouts, tables, and non-standard spacing. 
- Utilizes `pdfplumber` (wrapped via Langchain Loaders) for high-fidelity text extraction that respects document geometry.
- Uses advanced **Regex and custom Python heuristics** to automatically split massive text blocks into structured `Article` objects (identifying headers, titles, and body text), while aggressively filtering out noise (page numbers, trailing signatures, and chapter headers).

### 2. Intelligent "Compare & Align" Algorithm (Hybrid Scoring)
The core engineering challenge in legislative comparison is that **articles get renumbered** (e.g., Old Article 3 becomes New Article 4) or split. A simple 1-to-1 sequential comparison fails.

This project implements a sophisticated **Hybrid Scoring Matcher**:
- **Lexical Matching:** Uses `scikit-learn`'s **TF-IDF Vectorizer** and Cosine Similarity to find exact word-level matches.
- **Semantic Matching (AI):** Leverages `langchain-google-genai` and **Google Gemini Embeddings** (`gemini-embedding-2-preview`) to understand the *meaning* of the text, allowing the system to match articles even if the legislator completely paraphrased the content.
- **Alignment Pipeline:** Computes a weighted hybrid score (70% Semantic / 30% Lexical). It extracts the Top-K candidates and performs a greedy one-to-one bipartite matching to accurately link old articles to their new counterparts.

### 3. Granular Diffing & Classification
Once articles are intelligently matched, the system computes the exact changes:
- Uses `difflib.SequenceMatcher` for word-level granularity.
- **Change Classification:** Automatically categorizes articles into `UNCHANGED`, `MODIFIED`, `ADDED`, `REMOVED`, `RENUMBERED`, or `RENUMBERED_MODIFIED`.
- Renders a GitHub-style, color-coded HTML diff directly inside the Streamlit UI.

### 4. Public Consultation (OpenGov) Comments Analysis — Legal NLI Pipeline
The tool seamlessly integrates public feedback from the opengov.gr platform and goes beyond naive regex mapping. Because comments on opengov are posted **per chapter** (spanning multiple articles), the `"ΑΡΘΡΟ"` column in the `.xlsx` export is only a coarse range (e.g. `άρθρα 3-10`) — not the specific article the citizen is arguing about.

The `target_identification.py` service implements a three-way dispatch that decides, per comment, how to resolve the actual target article(s):

- **Path A — Regex (explicit reference):** When the cell contains a single explicit `(άρθρο N)`, the target is trusted verbatim with `confidence = 1.0`.
- **Path B — AI-narrowed (range → 1–3 articles):** When the cell contains a range, the chapter is passed to an LLM together with the comment, and the model picks the 1–3 most relevant articles *within that range* — or flags the comment as `chapter_wide` when it truly addresses the whole chapter. Per-comment `reasoning` and `confidence_score` are returned.
- **Path C — AI-free (no range at all):** A RAG step embeds all articles with `gemini-embedding-2-preview`, retrieves the top-K candidates via cosine similarity, and the same structured-output LLM picks the final targets.

Key engineering details:
- **Batched LLM calls:** all comments in the same chapter are resolved in a **single** LLM request (Pydantic `with_structured_output`), dramatically reducing round-trips and cost.
- **Persistent disk cache (`diskcache`):** LLM answers **and** the corpus embeddings matrix are cached under `~/.cache/legal_analyzer/llm_cache/`, keyed by a stable hash of the article corpus. Re-runs on the same PDF perform zero network calls.
- **Model fallback chain + retries (`tenacity`):** transient 5xx/429 errors trigger exponential-backoff retries, and permanent `404 model not found` errors transparently fail over to the next model in the chain (`gemini-2.5-flash` → `gemini-2.5-flash-lite`).
- **Structured logging (`services/logging_setup.py`):** every parse emits a human-readable `PARSE SUMMARY` with counters for cache hits, successful LLM calls, retries, model fallbacks, and a breakdown by target method/scope.

---

## ⚠️ Technical Challenges & Ongoing Work

To demonstrate the real-world complexity of this project to engineering teams and recruiters, here are some of the core algorithmic and NLP problems we are actively solving:

### 1. The "Compare & Alignment" Problem (Split & Diff)
When a new law is drafted, articles are not simply rewritten; they are often **split into multiple new articles**, merged, or drastically reordered.
- **Challenge:** A standard 1-to-1 diffing algorithm fails when "Old Article 4" becomes "New Article 5" and "New Article 6".
- **Solution/WIP:** Enhancing our Hybrid Scoring algorithm to support **1-to-N and N-to-M bipartite matching**, ensuring that even when legislative text is fractured across multiple new sections, the semantic alignment holds and the diff remains accurate.

### 2. Legal NLI & Argument Targeting — *(implemented)*
When citizens leave comments on public consultation platforms (OpenGov), they rarely use structured references. A comment might say "This contradicts the previous clause" or reference an article by its context rather than its number.
- **Challenge:** Accurately mapping a free-text comment to the exact article it refers to when Regex is too coarse or absent.
- **Solution:** The Legal NLI pipeline described above (Section 4) uses LLMs with structured outputs to narrow a chapter-level range down to the 1–3 most relevant articles, or to perform full-corpus retrieval when no range is present. Each prediction carries a `confidence_score` and a human-readable `reasoning`, surfaced in the UI so the legislator can audit uncertain cases.

---

## 🛠️ Tech Stack & Libraries

- **Frontend / UI:** [Streamlit](https://streamlit.io/) (Interactive data tables, master-detail layout for comments analysis)
- **Data Structures & Processing:** Python `dataclasses` (Strict typing and low memory overhead), `pandas`, and `openpyxl` (Excel parsing)
- **AI & NLP:**
  - `langchain` & `langchain-google-genai` (GenAI integrations, structured outputs, embeddings)
  - Google Gemini: `gemini-embedding-2-preview` for retrieval; `gemini-2.5-flash` / `gemini-2.5-flash-lite` for NLI (with automatic fallback)
  - `pydantic` (schema-validated LLM outputs)
  - `scikit-learn` (TF-IDF vectorization for the diff pipeline)
- **Reliability & Performance:**
  - `diskcache` (persistent on-disk cache for LLM responses and corpus embeddings)
  - `tenacity` (exponential-backoff retries on transient API errors)
- **Document Processing:** `pdfplumber` (For precise PDF text extraction)
- **Package Management:** `uv` (Ultra-fast Python package installer)

---

## 📁 Project Architecture (Separation of Concerns)

The backend is built with clean architecture principles in mind:

```text
legal_analyzer/
├── main.py                       # Streamlit entry point, master-detail UI for comments
├── models/
│   └── models.py                 # Dataclasses: Article, DiffSegment, ArticleDiff,
│                                 #             CommentTarget, Comment
└── services/
    ├── extract_text.py           # PDF parsing via Langchain
    ├── split_text.py             # Regex heuristics for Article structuring
    ├── comments_parser.py        # Excel parsing, 3-way dispatch (regex / narrowed / free),
    │                             # batching per chapter, PARSE SUMMARY reporting
    ├── target_identification.py  # Legal NLI pipeline: RAG retrieval, batched LLM calls,
    │                             # disk cache, retry + model fallback chain
    ├── logging_setup.py          # Centralized, idempotent logger configuration
    ├── normalizer.py             # Text normalization (accents, punctuation removal)
    ├── scorer.py                 # TF-IDF & Gemini Embedding matrix computations
    ├── matcher.py                # Hybrid scoring & one-to-one alignment logic
    ├── differ.py                 # Sequence matching & word-level diffing
    ├── significance.py           # Change classification logic
    └── pipeline.py               # The orchestrator tying the services together
```

### Caching layout

```text
~/.cache/legal_analyzer/llm_cache/    # diskcache store
├── corpus_embeddings::<model>::<sha1(articles)>   # np.ndarray, ~1 per PDF
└── <sha1(comment + valid_numbers + corpus_sig)>   # per-comment LLM response
```

The cache location can be overridden via the `LEGAL_ANALYZER_CACHE_DIR`
environment variable (useful for CI or sandboxed test runs). The cache
can also be cleared from the UI (**Comments Analysis** tab → 🗑️ Clear
LLM cache).

---

## 💻 Getting Started

### Prerequisites
- Python 3.13+
- `uv` package manager installed
- A Google Gemini API Key (only needed for the *first* run on a given PDF — subsequent runs can be fully served from cache)

### Installation

1. Clone the repository
2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```
3. Create a `.env` file in the root directory and add your Google API Key:
   ```env
   GOOGLE_API_KEY="your_api_key_here"
   ```
   > **Note:** `.env` is listed in `.gitignore`. Never commit API keys.

### Running the App

```bash
uv run streamlit run main.py
```

**Scenario A: Law Comparison (Diffing)**
1. Go to the **Uploads** tab and upload your Old Law (PDF) and New Law (PDF).
2. Click **Extract Text** to parse the documents.
3. Move to the **Tables & Analysis** tab to view the structured data, and click **Compare and align articles**.
4. View the final, color-coded results in the **Article Differences** tab!

**Scenario B: Public Comments Analysis**
1. Go to the **Uploads** tab and upload your Initial Law (PDF) and the OpenGov Comments (`.xlsx`).
2. Click **Parse Comments & Initial Law**. A progress bar tracks the Legal NLI pipeline in real time.
3. Move to the **Comments Analysis** tab. The left pane lists every article with its comment count; clicking an entry reveals the matched comments on the right, each with a confidence badge (🟢/🟡/🔴), the AI's reasoning, and an optional "Hide uncertain" filter.
4. The collapsible **🔍 Parse stats** panel at the top exposes totals, breakdowns by method (`regex` / `ai_nli_narrowed` / `ai_nli`) and scope (`article` / `chapter_wide`), and a cache-clear button.
