import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

class Scorer:
    def __init__(self):
        # Requires GOOGLE_API_KEY environment variable to be set.
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.vectorizer = TfidfVectorizer()
    
    def compute_tfidf_similarities(self, old_texts: list[str], new_texts: list[str]) -> np.ndarray:
        if not old_texts or not new_texts:
            return np.zeros((len(old_texts), len(new_texts)))
            
        # Fit on both to ensure same vocabulary
        self.vectorizer.fit(old_texts + new_texts)
        old_vecs = self.vectorizer.transform(old_texts)
        new_vecs = self.vectorizer.transform(new_texts)
        
        return cosine_similarity(old_vecs, new_vecs)
        
    def compute_embedding_similarities(self, old_texts: list[str], new_texts: list[str]) -> np.ndarray:
        if not old_texts or not new_texts:
            return np.zeros((len(old_texts), len(new_texts)))
            
        old_embeds = self.embeddings_model.embed_documents(old_texts)
        new_embeds = self.embeddings_model.embed_documents(new_texts)
        
        return cosine_similarity(old_embeds, new_embeds)
