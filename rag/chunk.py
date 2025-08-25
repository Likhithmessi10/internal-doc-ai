from typing import List, Dict

def simple_chunk(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunks.append(text[i:end])
        if end == n:
            break
        i = end - overlap if end - overlap > i else end
    return chunks

def make_docs_chunks(text: str, source_path: str, max_chars: int = 1000, overlap: int = 200) -> List[Dict]:
    parts = simple_chunk(text, max_chars, overlap)
    return [{"text": p, "metadata": {"source": source_path}} for p in parts]
