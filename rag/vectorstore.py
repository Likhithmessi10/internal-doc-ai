import os
import json
import numpy as np
from typing import List, Dict, Tuple

class DiskVectorStore:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)  # ensure folder exists
        self.meta_path = os.path.join(index_dir, "meta.json")
        self.vectors_path = os.path.join(index_dir, "vectors.npy")

        # initialize in-memory data
        self.embeddings = None
        self.meta = []

    def save(self):
        if self.embeddings is None:
            raise ValueError("No embeddings to save.")
        np.save(self.vectors_path, self.embeddings)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.vectors_path) and os.path.exists(self.meta_path):
            self.embeddings = np.load(self.vectors_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            raise FileNotFoundError("No index found. Build it first.")

    def build(self, docs: List[Dict], embed_fn):
        """Builds the vector index from docs using embed_fn"""
        texts = [d['text'] for d in docs]
        self.embeddings = embed_fn(texts)  # embeddings must be np.ndarray
        self.meta = docs
        self.save()

    def clear(self):
        """Deletes stored index files and resets memory"""
        if os.path.exists(self.vectors_path):
            os.remove(self.vectors_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
        self.embeddings, self.meta = None, []

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        """Returns top_k most similar docs for the query_vec"""
        if self.embeddings is None:
            self.load()

        A = self.embeddings.astype(np.float32)
        q = query_vec.astype(np.float32)

        # normalize for cosine similarity
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        sims = A_norm @ q_norm
        idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.meta[i]) for i in idx]
