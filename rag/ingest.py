import os
from typing import List, Dict
from pypdf import PdfReader
from docx import Document

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_md(path: str) -> str:
    return read_txt(path)

def read_pdf(path: str) -> str:
    text_parts = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join([t for t in text_parts if t])

def read_docx(path: str) -> str:
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

EXT_READERS = {
    ".txt": read_txt,
    ".md": read_md,
    ".pdf": read_pdf,
    ".docx": read_docx,
}

def load_folder(folder: str) -> List[Dict]:
    out = []
    for root, _, files in os.walk(folder):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in EXT_READERS:
                p = os.path.join(root, name)
                try:
                    text = EXT_READERS[ext](p)
                    if text and text.strip():
                        out.append({"text": text, "path": p})
                except Exception:
                    continue
    return out
