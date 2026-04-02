import streamlit as st
from services.extract_text import load_legal_document
from services.split_text import extract_and_split_documents
from services.pipeline import run_comparison_pipeline
from services.comments_parser import parse_comments_excel
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

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
                with st.spinner("Parsing data..."):
                    # Parse the Excel
                    st.session_state['parsed_comments'] = parse_comments_excel(comments)
                    
                    # Extract and split the initial legal document
                    raw_docs = load_legal_document(initial_law)
                    st.session_state['articles_initial_law'] = extract_and_split_documents(raw_docs)
                    
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
            
            for article in articles:
                # To article is a dictionary with keys: 'article_number', 'header', 'title', 'body'
                art_num = article['article_number']
                
                # Find which comments relate to this article
                relevant_comments = [c for c in all_comments if art_num in c['articles']]
                
                # Create an expander for each article with the title and number of comments
                with st.expander(f"{article['header']} ({len(relevant_comments)} comments)"):
                    # 1. Display the text of the article
                    st.markdown(f"**{article['title']}**")
                    st.write(article['body'])
                    st.markdown("---")
                    
                    # 2. Display the comments
                    if relevant_comments:
                        st.markdown("#### Comments:")
                        for comment in relevant_comments:
                            st.info(f"**ID: {comment['comment_id']}**\n\n{comment['comment']}")
                    else:
                        st.write("No comments found for this article.")
        else:
            st.info("Please upload the initial law and comments (excel) in the 'Uploads' tab.")

    # --- VIEW 4: LEGAL REPORT ---
    with tab_report:
        st.subheader("Final Legal Report")
        if initial_law and final_law and comments:
            if 'comparison_results' in st.session_state:
                st.write("### Comparison Results")
                results = st.session_state['comparison_results']
                st.write(f"Total articles processed: {len(results)}")
                
                for diff in results:
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