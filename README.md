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

---

## 🛠️ Tech Stack & Libraries

- **Frontend / UI:** [Streamlit](https://streamlit.io/) (Interactive data tables and UI components)
- **Data Structures:** Python `dataclasses` (Strict typing and low memory overhead) & `pandas`
- **AI & NLP:** 
  - `langchain` & `langchain-google-genai` (For GenAI Integrations and Embeddings)
  - `scikit-learn` (For TF-IDF Vectorization)
- **Document Processing:** `pdfplumber` (For precise PDF text extraction)
- **Package Management:** `uv` (Ultra-fast Python package installer)

---

## 📁 Project Architecture (Separation of Concerns)

The backend is built with clean architecture principles in mind:

```text
legal_analyzer/
├── main.py                  # Streamlit entry point and UI logic
├── models/
│   └── models.py            # Dataclasses (Article, DiffSegment, ArticleDiff) and Enums
└── services/
    ├── extract_text.py      # PDF parsing via Langchain
    ├── split_text.py        # Regex heuristics for Article structuring
    ├── normalizer.py        # Text normalization (accents, punctuation removal)
    ├── scorer.py            # TF-IDF & Gemini Embedding matrix computations
    ├── matcher.py           # Hybrid scoring & one-to-one alignment logic
    ├── differ.py            # Sequence matching & word-level diffing
    ├── significance.py      # Change classification logic
    └── pipeline.py          # The orchestrator tying the services together
```

---

## 💻 Getting Started

### Prerequisites
- Python 3.13+
- `uv` package manager installed
- A Google Gemini API Key

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

### Running the App

```bash
uv run streamlit run main.py
```

1. Go to the **Uploads** tab and upload your Old Law (PDF) and New Law (PDF).
2. Click **Extract Text** to parse the documents.
3. Move to the **Tables & Analysis** tab to view the structured data, and click **Compare and align articles**.
4. View the final, color-coded results in the **Article Differences** tab!
