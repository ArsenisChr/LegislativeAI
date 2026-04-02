import re
import pandas as pd
from typing import List, Dict

def extract_article_range(section_text: str) -> List[str]:
    """
    Extracts article numbers as strings so they can match our Article models.
    e.g., "(άρθρα 2 - 5)" -> ["2", "3", "4", "5"]
    e.g., "(άρθρο 3)" -> ["3"]
    """
    if not section_text or pd.isna(section_text):
        return []
        
    text = str(section_text).strip()
    
    # Range match: e.g. "(άρθρα 2 - 5)"
    range_match = re.search(r'\(άρθρα?\s*(\d+)\s*-\s*(\d+)\)', text, re.IGNORECASE)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        return [str(i) for i in range(start, end + 1)]
        
    # Single match: e.g. "(άρθρο 3)"
    single_match = re.search(r'\(άρθρα?\s*(\d+)\)', text, re.IGNORECASE)
    if single_match:
        return [single_match.group(1)]
        
    return []

def parse_comments_excel(excel_file) -> List[Dict]:
    """
    Reads the Excel and returns a list of dictionaries mapping comments to articles.
    """
    df = pd.read_excel(excel_file)
    
    comments_list = []
    
    for index, row in df.iterrows():
        comment_id = row.get("ΚΩΔΙΚΟΣ ΣΧΟΛΙΟΥ")
        article_text = row.get("ΑΡΘΡΟ")
        comment_body = row.get("ΣΧΟΛΙΟ")
        
        # Handle NaN comments
        if pd.isna(comment_body):
            comment_body = ""
            
        articles = extract_article_range(article_text)
        
        comments_list.append({
            "comment_id": comment_id,
            "articles": articles,
            "comment": comment_body
        })
        
    return comments_list
