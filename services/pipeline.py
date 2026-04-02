from typing import List
from models.models import Article, ArticleDiff, ChangeType
from services.normalizer import normalize_for_comparison
from services.scorer import Scorer
from services.matcher import match_articles
from services.differ import compute_diff
from services.significance import classify_change

def run_comparison_pipeline(old_dicts: List[dict], new_dicts: List[dict]) -> List[ArticleDiff]:
    old_articles = [Article(**d) for d in old_dicts]
    new_articles = [Article(**d) for d in new_dicts]
    
    if not old_articles and not new_articles:
        return []
        
    scorer = Scorer()
    
    # Normalize texts (combining title and body for scoring)
    old_norm = [normalize_for_comparison(f"{a.title} {a.body}") for a in old_articles]
    new_norm = [normalize_for_comparison(f"{a.title} {a.body}") for a in new_articles]
    
    # Compute scores
    tfidf_scores = scorer.compute_tfidf_similarities(old_norm, new_norm)
    embed_scores = scorer.compute_embedding_similarities(old_norm, new_norm)
    
    # Match articles
    matched_pairs, unmatched_old, unmatched_new = match_articles(
        old_articles, new_articles, tfidf_scores, embed_scores, threshold=0.6
    )
    
    results = []
    
    # 1. Process matched pairs
    for old_art, new_art, score in matched_pairs:
        # Compare just the body, or title + body? 
        # Usually it's best to show diff for both, but let's combine them for the diff segment
        old_text = f"{old_art.title}\n\n{old_art.body}".strip()
        new_text = f"{new_art.title}\n\n{new_art.body}".strip()
        
        segments = compute_diff(old_text, new_text)
        change_type = classify_change(old_art, new_art, segments)
        
        results.append(ArticleDiff(
            old_article=old_art,
            new_article=new_art,
            change_type=change_type,
            similarity_score=score,
            segments=segments
        ))
        
    # 2. Process unmatched old (Removed)
    for old_art in unmatched_old:
        results.append(ArticleDiff(
            old_article=old_art,
            new_article=None,
            change_type=ChangeType.REMOVED,
            similarity_score=0.0
        ))
        
    # 3. Process unmatched new (Added)
    for new_art in unmatched_new:
        results.append(ArticleDiff(
            old_article=None,
            new_article=new_art,
            change_type=ChangeType.ADDED,
            similarity_score=0.0
        ))
        
    # Sort results
    # We can sort by the new article number if present, else old article number
    def get_sort_key(diff: ArticleDiff):
        art = diff.new_article if diff.new_article else diff.old_article
        num_str = art.article_number if art else ""
        import re
        m = re.match(r"^(\d+)([Α-ΩA-Z]?)$", num_str)
        if not m:
            return (999999, "")
        return (int(m.group(1)), m.group(2) or "")
        
    results.sort(key=get_sort_key)
    
    return results
