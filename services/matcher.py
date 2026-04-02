import numpy as np
from typing import List, Tuple
from models.models import Article

def match_articles(
    old_articles: List[Article], 
    new_articles: List[Article], 
    tfidf_scores: np.ndarray, 
    embed_scores: np.ndarray, 
    threshold: float = 0.6
) -> Tuple[List[Tuple[Article, Article, float]], List[Article], List[Article]]:
    
    # Hybrid scoring: e.g., 70% Embeddings, 30% TF-IDF
    combined_scores = 0.7 * embed_scores + 0.3 * tfidf_scores
    
    matched_pairs = []
    matched_old_indices = set()
    matched_new_indices = set()
    
    if combined_scores.size > 0:
        # Flatten and sort indices descending by score
        flat_scores = combined_scores.flatten()
        sorted_indices = np.argsort(flat_scores)[::-1]
        
        num_new = len(new_articles)
        
        for idx in sorted_indices:
            old_idx = int(idx // num_new)
            new_idx = int(idx % num_new)
            
            if old_idx in matched_old_indices or new_idx in matched_new_indices:
                continue
                
            score = combined_scores[old_idx, new_idx]
            
            if score < threshold:
                break  # The rest are lower than threshold
                
            matched_pairs.append((old_articles[old_idx], new_articles[new_idx], float(score)))
            matched_old_indices.add(old_idx)
            matched_new_indices.add(new_idx)
        
    unmatched_old = [old_articles[i] for i in range(len(old_articles)) if i not in matched_old_indices]
    unmatched_new = [new_articles[i] for i in range(len(new_articles)) if i not in matched_new_indices]
    
    return matched_pairs, unmatched_old, unmatched_new
