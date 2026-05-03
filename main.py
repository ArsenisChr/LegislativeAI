import streamlit as st
from models.models import ChangeType
from services.extract_text import load_legal_document
from services.split_text import extract_and_split_documents
from services.pipeline import run_comparison_pipeline
from services.comments_parser import parse_comments_excel
from services.target_identification import clear_llm_cache
from services.logging_setup import setup_logging
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
setup_logging()
st.set_page_config(page_title="Legislative AI", layout="wide")
st.markdown(
    """
    <style>
        .block-container {
            max-width: 100%;
            padding-left: 5rem;
            padding-right: 5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Confidence threshold below which an AI-inferred target is considered
# uncertain and moved into a separate "needs-review" section of the UI.
LOW_CONFIDENCE_THRESHOLD = 0.4


def _filter_legal_report_diffs(results, filter_mode: str):
    """Filter article diffs for the legal report tab."""
    if filter_mode == "modified":
        return [
            d
            for d in results
            if d.change_type
            in (ChangeType.MODIFIED, ChangeType.RENUMBERED_MODIFIED)
        ]
    if filter_mode == "unchanged":
        return [d for d in results if d.change_type == ChangeType.UNCHANGED]
    return list(results)


def _confidence_icon(score: float) -> str:
    """Traffic-light icon for a confidence score."""
    if score >= 0.7:
        return "🟢"
    if score >= 0.4:
        return "🟡"
    return "🔴"


def _short_badge(target) -> str:
    """Short, single-line badge for a `CommentTarget`."""
    if target.method == "regex":
        return "🟢 **Ρητή αναφορά**"
    icon = _confidence_icon(target.confidence_score)
    label = "AI (narrowed)" if target.method == "ai_nli_narrowed" else "AI (free)"
    return f"{icon} **{label}** · conf {target.confidence_score:.2f}"


def _render_comment_flat(comment, target) -> None:
    """Render a single comment as a flat bordered card (no nested expanders)."""
    with st.container(border=True):
        st.markdown(f"{_short_badge(target)}  ·  `#{comment.comment_id}`")
        st.write(comment.comment)
        if target.method != "regex" and target.reasoning:
            st.caption(f"💭 {target.reasoning}")


def _render_debug_panel(articles, all_comments) -> None:
    """Collapsed-by-default panel with the same numbers as the PARSE SUMMARY
    log line, plus a cache-clear button. Keeps the UI tidy while still giving
    a quick sanity-check view next to the Comments tab."""
    total = len(all_comments)
    with_targets = sum(1 for c in all_comments if c.targets)
    by_method: dict = {}
    by_scope: dict = {}
    for c in all_comments:
        for t in c.targets:
            by_method[t.method] = by_method.get(t.method, 0) + 1
            by_scope[t.scope] = by_scope.get(t.scope, 0) + 1

    with st.expander("🔍 Parse stats", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total comments", total)
        col2.metric(
            "With target",
            with_targets,
            f"{(100 * with_targets / total):.0f}%" if total else "—",
        )
        col3.metric("Empty", total - with_targets)

        st.caption("Targets by method / scope")
        st.write({"method": by_method, "scope": by_scope})

        st.caption(f"PDF articles: {len(articles)}")

        if st.button("🗑️ Clear LLM cache"):
            clear_llm_cache()
            st.success(
                "Cache cleared. Re-run 'Parse Comments & Initial Law' "
                "to force fresh LLM calls."
            )


def _collect_hits_per_article(articles, all_comments):
    """Bucket comments per article into (chapter_wide, confident, low_conf)."""
    buckets: dict = {}
    for article in articles:
        art_num = article['article_number']
        chapter_wide, confident, low_conf = [], [], []
        for comment in all_comments:
            if not hasattr(comment, 'targets'):
                continue
            for target in comment.targets:
                if (
                    target.scope == 'chapter_wide'
                    and art_num in target.chapter_range
                ):
                    chapter_wide.append((comment, target))
                elif target.article_number == art_num:
                    if (
                        target.method == "regex"
                        or target.confidence_score >= LOW_CONFIDENCE_THRESHOLD
                    ):
                        confident.append((comment, target))
                    else:
                        low_conf.append((comment, target))
        buckets[art_num] = (chapter_wide, confident, low_conf)
    return buckets


def _render_article_detail(article, hits, hide_uncertain: bool) -> None:
    """Render the right-hand detail pane for a single selected article."""
    chapter_wide, confident, low_conf = hits

    st.markdown(f"### {article['header']}")
    if article.get('title'):
        st.markdown(f"*{article['title']}*")

    with st.expander("📄 Κείμενο άρθρου", expanded=False):
        st.write(article['body'])

    total = len(chapter_wide) + len(confident) + len(low_conf)
    if total == 0:
        st.info("Δεν υπάρχουν σχόλια για αυτό το άρθρο.")
        return

    if chapter_wide:
        st.markdown(f"#### 📌 Γενικά σχόλια κεφαλαίου ({len(chapter_wide)})")
        st.caption(
            "Αφορούν τη γενική κατεύθυνση του κεφαλαίου — εμφανίζονται "
            "σε όλα τα άρθρα του."
        )
        for comment, target in chapter_wide:
            _render_comment_flat(comment, target)

    if confident:
        st.markdown(f"#### Σχόλια για το άρθρο ({len(confident)})")
        for comment, target in confident:
            _render_comment_flat(comment, target)

    if low_conf and not hide_uncertain:
        st.markdown(
            f"#### ⚠️ Αβέβαια στόχευση — προς έλεγχο ({len(low_conf)})"
        )
        st.caption(
            "Το AI επέστρεψε χαμηλή βεβαιότητα. Ελέγξτε χειροκίνητα αν "
            "όντως αφορούν αυτό το άρθρο."
        )
        for comment, target in low_conf:
            _render_comment_flat(comment, target)


def _render_comments_analysis(articles, all_comments) -> None:
    """Master-detail layout: article list on the left, selected article on the right."""
    if not articles:
        st.info("Δεν έχουν εξαχθεί άρθρα από τον αρχικό νόμο.")
        return

    buckets = _collect_hits_per_article(articles, all_comments)

    def _count(art_num: str) -> int:
        cw, cf, lc = buckets[art_num]
        return len(cw) + len(cf) + len(lc)

    col_list, col_detail = st.columns([1, 3], gap="large")

    with col_list:
        st.markdown("**Άρθρα**")
        hide_uncertain = st.toggle(
            "Απόκρυψη αβέβαιων",
            value=False,
            help=(
                "Απόκρυψη σχολίων με confidence κάτω από "
                f"{LOW_CONFIDENCE_THRESHOLD:.1f}."
            ),
        )

        options = [a['article_number'] for a in articles]
        headers = {a['article_number']: a['header'] for a in articles}

        def _fmt(num: str) -> str:
            n = _count(num)
            marker = "•" if n else "·"
            return f"{marker} {headers[num]}  ({n})"

        with st.container(height=520, border=False):
            selected = st.radio(
                "Επιλογή άρθρου",
                options=options,
                format_func=_fmt,
                label_visibility="collapsed",
                key="comments_analysis_selected_article",
            )

    with col_detail:
        article = next(
            (a for a in articles if a['article_number'] == selected), None
        )
        if not article:
            st.info("Επιλέξτε ένα άρθρο από τη λίστα αριστερά.")
            return
        _render_article_detail(article, buckets[selected], hide_uncertain)


def main():
    st.title("Legislative AI")
    st.write("Analyze and understand Greek legislative documents with ease.")

    # Δημιουργία των 4 views με τη μορφή tabs
    tab_uploads, tab_analysis, tab_comments, tab_report = st.tabs([
        "📁 Uploads", 
        "📊 Tables & Analysis", 
        "💬 Comments Analysis",
        "📝 Article Differences"
    ])

    # --- VIEW 1: UPLOADS ---
    with tab_uploads:
        st.subheader("Upload Documents")
        initial_law = st.file_uploader("Upload your initial legislative document here:", type=["pdf"])
        final_law = st.file_uploader("Upload your final legislative document here:", type=["pdf"])
        comments = st.file_uploader("Upload opengov comments here:", type=["xlsx"])

        if initial_law and comments and not final_law:
            st.success("Initial law and comments uploaded successfully!")
            
            if st.button("Parse Comments & Initial Law"):
                # Extract and split the initial legal document FIRST, so we
                # have the article corpus available for the Legal NLI paths.
                with st.spinner("Extracting & splitting the initial law..."):
                    raw_docs = load_legal_document(initial_law)
                    articles = extract_and_split_documents(raw_docs)
                    st.session_state['articles_initial_law'] = articles

                # Comment parsing may invoke the LLM per row, which takes a
                # noticeable amount of time. Render a live progress bar.
                progress_placeholder = st.empty()
                progress_bar = progress_placeholder.progress(
                    0.0, text="Ανάλυση σχολίων..."
                )

                def _on_progress(done: int, total: int, label: str) -> None:
                    fraction = done / total if total else 1.0
                    progress_bar.progress(fraction, text=f"{label} — {done}/{total}")

                st.session_state['parsed_comments'] = parse_comments_excel(
                    comments,
                    articles=articles,
                    progress_cb=_on_progress,
                )
                progress_placeholder.empty()
                st.success("Data parsed! Go to 'Comments Analysis' tab.")

        if initial_law and final_law and comments:
            st.success("Both Documents uploaded successfully!")
            if st.button("Extract text"):
                with st.spinner("Extracting text..."):
                    st.session_state['documents_initial_law'] = load_legal_document(initial_law)
                    st.session_state['documents_final_law'] = load_legal_document(final_law)
                    st.success("Extraction completed! See the articles in the tab 'Tables & Analysis'.")

            if 'documents_initial_law' in st.session_state and 'documents_final_law' in st.session_state:
                st.write("Initial law documents:")
                st.write("Number of pages: " + str(len(st.session_state['documents_initial_law'])))
                st.write("Final law documents:")
                st.write("Number of pages: " + str(len(st.session_state['documents_final_law'])))

    # --- VIEW 2: TABLES & ANALYSIS ---
    with tab_analysis:
        st.subheader("Analysis Results")
        if initial_law and final_law and comments:
            if 'documents_initial_law' in st.session_state and 'documents_final_law' in st.session_state:
                
                if 'articles_initial_law' not in st.session_state:
                    st.write("Splitting documents into articles...")
                    st.session_state['articles_initial_law'] = extract_and_split_documents(st.session_state['documents_initial_law'])
                    st.session_state['articles_final_law'] = extract_and_split_documents(st.session_state['documents_final_law'])
                
                articles_initial_law = st.session_state['articles_initial_law']
                articles_final_law = st.session_state['articles_final_law']
                
                st.write("Initial law articles:")
                st.write("Number of articles: " + str(len(articles_initial_law)))
                st.dataframe(articles_initial_law, width='stretch')
                
                st.write("Final law articles:")
                st.write("Number of articles: " + str(len(articles_final_law)))
                st.dataframe(articles_final_law, width='stretch')
                
                if st.button("Compare and align articles"):
                    with st.spinner("Comparing and aligning articles (this may take a few seconds)..."):
                        comparison_results = run_comparison_pipeline(articles_initial_law, articles_final_law)
                        st.session_state['comparison_results'] = comparison_results
                        st.success("Comparison completed! Go to the 'Article Differences' tab to see the report.")
            else:
                st.info("Please click 'Extract text' in the 'Uploads' tab to extract the data.")
        else:
            st.info("Please upload all 3 documents in the 'Uploads' tab to see the analysis.")

    # --- VIEW 3: COMMENTS ANALYSIS ---
    with tab_comments:
        st.subheader("Comments for each article")

        if 'parsed_comments' in st.session_state and 'articles_initial_law' in st.session_state:
            articles = st.session_state['articles_initial_law']
            all_comments = st.session_state['parsed_comments']

            _render_debug_panel(articles, all_comments)
            _render_comments_analysis(articles, all_comments)
        else:
            st.info("Please upload the initial law and comments (excel) in the 'Uploads' tab.")

    # --- VIEW 4: LEGAL REPORT ---
    with tab_report:
        st.subheader("Final Legal Report")
        if initial_law and final_law and comments:
            if 'comparison_results' in st.session_state:
                st.write("### Comparison Results")
                results = st.session_state['comparison_results']
                if "legal_report_filter" not in st.session_state:
                    st.session_state["legal_report_filter"] = "all"

                f1, f2, f3 = st.columns(3)
                if f1.button("Όλα", key="legal_report_filter_all"):
                    st.session_state["legal_report_filter"] = "all"
                if f2.button("Modified", key="legal_report_filter_modified"):
                    st.session_state["legal_report_filter"] = "modified"
                if f3.button("Unchanged", key="legal_report_filter_unchanged"):
                    st.session_state["legal_report_filter"] = "unchanged"

                mode = st.session_state["legal_report_filter"]
                filtered = _filter_legal_report_diffs(results, mode)
                st.write(f"Εμφανίζονται {len(filtered)} από {len(results)} άρθρα.")

                for diff in filtered:
                    if diff.change_type.value == "added":
                        title = f"Τελικό - {diff.new_article.header}"
                    elif diff.change_type.value == "removed":
                        title = f"Αρχικό - {diff.old_article.header}"
                    elif diff.change_type.value in ["renumbered", "renumbered_modified"]:
                        title = f"Αρχικό {diff.old_article.header} ➜ Τελικό {diff.new_article.header}"
                    else:
                        title = f"Ιδιο {diff.old_article.header}"
                        
                    with st.expander(f"{title}  |  Status: {diff.change_type.value.upper()}"):
                        st.write(f"**Similarity Score:** {diff.similarity_score:.2f}")
                        
                        if diff.change_type.value in ["added", "removed", "unchanged"]:
                            art = diff.new_article if diff.new_article else diff.old_article
                            st.info(art.title)
                            st.write(art.body)
                        else:
                            # Show Diff Segments
                            st.write("#### Changes:")
                            diff_text = ""
                            for seg in diff.segments:
                                if seg.operation == "insert":
                                    diff_text += f"<span style='background-color: #d4edda; color: #155724;'>{seg.text}</span>"
                                elif seg.operation == "delete":
                                    diff_text += f"<span style='background-color: #f8d7da; color: #721c24; text-decoration: line-through;'>{seg.text}</span>"
                                else:
                                    diff_text += seg.text
                            st.markdown(diff_text, unsafe_allow_html=True)
            else:
                st.warning("Please run 'Compare and align articles' in the 'Tables & Analysis' tab first.")
        else:
            st.warning("A complete upload is required to generate the legal report.")

if __name__ == "__main__":
    main()