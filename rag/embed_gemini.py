import os
import numpy as np
from typing import List
import google.generativeai as genai

_EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

def _ensure_client():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set. Set it in your environment or .env file.")
    genai.configure(api_key=key)

def _extract_vector(resp):
    if isinstance(resp, dict):
        emb = resp.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            return emb["values"]
        return emb
    emb = getattr(resp, "embedding", None)
    if isinstance(emb, dict) and "values" in emb:
        return emb["values"]
    return emb

def embed_texts(texts: List[str]) -> np.ndarray:
    _ensure_client()
    vecs = []
    for t in texts:
        resp = genai.embed_content(model=_EMBED_MODEL, content=t)
        v = _extract_vector(resp)
        vecs.append(v)
    return np.array(vecs, dtype=np.float32)

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
