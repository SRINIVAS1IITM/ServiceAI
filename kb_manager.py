import json
import os
from typing import List, Optional, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .models import KBArticle

KNOWLEDGE_BASE_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base.json')
# Using a smaller, faster model for demo purposes.
# For better accuracy, consider "all-mpnet-base-v2" or other larger models.
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 

class KBManager:
    def __init__(self):
        self.kb: List[Dict] = []
        self.documents: List[str] = [] # Text to be embedded
        self.index = None
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._load_kb()
        self._build_index()

    def _load_kb(self):
        if not os.path.exists(KNOWLEDGE_BASE_FILE):
            print(f"Warning: Knowledge base file not found at {KNOWLEDGE_BASE_FILE}")
            return
        with open(KNOWLEDGE_BASE_FILE, 'r') as f:
            self.kb = json.load(f)
        
        # Prepare documents for embedding. We'll embed the 'answer' for retrieval.
        # Could also embed "topic: " + "answer" or "question: " + "answer"
        self.documents = [item['answer'] for item in self.kb]
        print(f"Loaded {len(self.kb)} articles from knowledge base.")

    def _build_index(self):
        if not self.documents:
            print("No documents to build index from.")
            return
        
        print("Building FAISS index...")
        embeddings = self.model.encode(self.documents, convert_to_tensor=False)
        embeddings = np.array(embeddings).astype('float32') # FAISS requires float32
        
        if embeddings.shape[0] > 0:
            self.index = faiss.IndexFlatL2(embeddings.shape[1]) # L2 distance
            self.index.add(embeddings)
            print(f"FAISS index built with {self.index.ntotal} vectors.")
        else:
            print("No embeddings generated, FAISS index not built.")


    def search_kb(self, query: str, top_k: int = 1, threshold: float = 0.75) -> Optional[KBArticle]:
        """
        Searches the KB using semantic search.
        Threshold is a similarity score (cosine similarity based, higher is better).
        FAISS L2 distance is lower is better. We need to convert.
        A common way to convert L2 distance to a similarity score (0 to 1) is:
        similarity = 1 / (1 + L2_distance)
        Or, if embeddings are normalized (which Sentence Transformers usually does):
        cosine_similarity = 1 - (L2_distance^2 / 2)
        Let's use the latter as it's more standard for normalized embeddings.
        """
        if self.index is None or self.index.ntotal == 0:
            print("KB index not available.")
            return None

        query_embedding = self.model.encode([query], convert_to_tensor=False).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        if indices.size > 0 and indices[0][0] != -1 : # -1 means no result
            best_idx = indices[0][0]
            best_dist = distances[0][0]
            
            # Convert L2 distance to cosine similarity (assuming normalized embeddings)
            # Cosine similarity = 1 - (L2_distance^2 / 2)
            # Max L2 distance for normalized vectors is 2 (when vectors are opposite).
            # Min L2 distance is 0 (when vectors are identical).
            similarity_score = 1 - (best_dist**2 / 2)
            
            print(f"Found article with L2 dist: {best_dist}, Sim Score: {similarity_score}")

            if similarity_score >= threshold:
                article_data = self.kb[best_idx]
                return KBArticle(
                    id=article_data['id'],
                    topic=article_data['topic'],
                    answer=article_data['answer'],
                    score=similarity_score
                )
        return None

# Initialize once
kb_manager = KBManager()