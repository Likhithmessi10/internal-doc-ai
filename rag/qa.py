import os
from typing import List, Dict
import google.generativeai as genai
from rag.embed_gemini import embed_query
from rag.vectorstore import DiskVectorStore

_GEN_MODEL = os.getenv("GEN_MODEL", "models/gemini-1.5-flash")

def _ensure_client():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set. Set it in your environment or .env file.")
    genai.configure(api_key=key)

def format_context(hits: List[Dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        source = hit["metadata"].get("source", "unknown")
        txt = hit["text"].strip()
        if len(txt) > 1200:
            txt = txt[:1200] + "..."
        blocks.append(f"[{i}] Source: {source}\n{txt}")
    return "\n\n".join(blocks)

def answer_question(question: str, index_dir: str = ".index", top_k: int = 4) -> Dict:
    _ensure_client()
    store = DiskVectorStore(index_dir)
    store.load()
    q_vec = embed_query(question)
    results = store.search(q_vec, top_k=top_k)
    hits = [m for _, m in results]
    ctx = format_context(hits)

    sys_prompt = (
        "You are an internal documentation assistant. Answer using ONLY the provided CONTEXT. "
        "If the answer is not in the context, say you don't see it in the docs and suggest where it might be. "
        "Be concise and include citations like [1], [2] that refer to the sources in CONTEXT."
    )

    full_prompt = f"{sys_prompt}\n\nQUESTION: {question}\n\nCONTEXT:\n{ctx}"

    model = genai.GenerativeModel(_GEN_MODEL)
    resp = model.generate_content(full_prompt)
    answer = resp.text if hasattr(resp, "text") else str(resp)

    return {"answer": answer, "hits": hits}
