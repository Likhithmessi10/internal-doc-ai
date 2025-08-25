import os
import streamlit as st
from dotenv import load_dotenv
from rag.ingest import load_folder
from rag.chunk import make_docs_chunks
from rag.embed_gemini import embed_texts
from rag.vectorstore import DiskVectorStore
from rag.qa import answer_question

load_dotenv()

st.set_page_config(page_title="Internal Docs AI Assistant", page_icon="🧠", layout="wide")

st.markdown(
    '<h1 style="margin-bottom:0">🧠 Internal Docs AI Assistant</h1>'
    '<div style="opacity:.7;margin-top:0">Final Day Hackathon Build</div>', 
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Settings")
    workspace = st.text_input("Workspace (namespace)", value="default")
    index_root = os.getenv("INDEX_ROOT", ".index")
    index_dir = os.path.join(index_root, workspace)
    st.caption(f"Index path: `{index_dir}`")

    st.divider()
    st.subheader("Indexing")
    max_chars = st.slider("Chunk size (chars)", 400, 2000, 1000, step=100)
    overlap = st.slider("Overlap (chars)", 50, 500, 200, step=50)

    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("Top-K", 1, 10, 4)

    st.divider()
    st.subheader("Models")
    st.text_input("GEN_MODEL", value=os.getenv("GEN_MODEL", "models/gemini-1.5-flash"), key="gen_model")
    st.text_input("EMBED_MODEL", value=os.getenv("EMBED_MODEL", "models/text-embedding-004"), key="emb_model")
    st.caption("Set GOOGLE_API_KEY in your environment or .env file.")

tab_ingest, tab_chat = st.tabs(["📥 Ingest", "💬 Chat"])

with tab_ingest:
    st.subheader("Upload or Select Folder")
    mode = st.radio("Choose input mode", ["Upload files", "Use a folder path"], horizontal=True)

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    if mode == "Upload files":
        ups = st.file_uploader("Drop multiple files (.txt, .md, .pdf, .docx)", type=["txt","md","pdf","docx"], accept_multiple_files=True)
    else:
        folder = st.text_input("Docs folder path", value="sample_docs")

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        if st.button("🧱 Build Index", type="primary", use_container_width=True):
            # gather paths
            if mode == "Upload files":
                if not ups:
                    st.error("Please upload at least one file.")
                    st.stop()
                # Save
                os.makedirs(upload_dir, exist_ok=True)
                saved = []
                for uf in ups:
                    dest = os.path.join(upload_dir, uf.name)
                    with open(dest, "wb") as out:
                        out.write(uf.read())
                    saved.append(dest)
                src_folder = upload_dir
            else:
                src_folder = folder

            with st.status("Loading and chunking documents...", expanded=True) as stat:
                files = load_folder(src_folder)
                st.write(f"Found {len(files)} file(s).")
                all_chunks = []
                for f in files:
                    chunks = make_docs_chunks(f["text"], f["path"], max_chars=max_chars, overlap=overlap)
                    all_chunks.extend(chunks)
                st.write(f"Total chunks: {len(all_chunks)}")

            if not all_chunks:
                st.error("No content to index. Check your folder path and file types.")
            else:
                with st.status("Embedding & saving index...", expanded=True):
                    store = DiskVectorStore(index_dir=index_dir)
                    store.build(all_chunks, embed_texts)
                st.success(f"Index built at `{index_dir}`.")

    with colB:
        if st.button("🧹 Clear Index", use_container_width=True):
            store = DiskVectorStore(index_dir=index_dir)
            store.clear()
            st.success("Index cleared.")

    with colC:
        if st.button("📚 Show Sources", use_container_width=True):
            try:
                store = DiskVectorStore(index_dir=index_dir)
                store.load()
                st.info(f"Indexed chunks: {len(store.meta)}")
                for i, m in enumerate(store.meta[:25], start=1):
                    st.write(f"{i}. {m['metadata'].get('source','unknown')}")
                if len(store.meta) > 25:
                    st.caption(f"...and {len(store.meta)-25} more")
            except Exception as e:
                st.error(str(e))

with tab_chat:
    st.subheader("Ask your docs")
    if "chat" not in st.session_state:
        st.session_state.chat = []  # list of dicts: {"q": str, "a": str, "hits": [...]}

    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input("Your question", placeholder="e.g., What's our refund policy?")
        submitted = st.form_submit_button("Send")
    if submitted and q.strip():
        try:
            with st.spinner("Thinking..."):
                out = answer_question(q, index_dir=index_dir, top_k=top_k)
            st.session_state.chat.append(out | {"q": q})
        except Exception as e:
            st.error(str(e))

    # Render chat history (newest first at bottom)
    for item in st.session_state.chat:
        st.markdown(f"**You:** {item['q']}")
        st.markdown(f"**AI:** {item['answer']}")
        with st.expander("Sources"):
            for i, h in enumerate(item.get("hits", []), start=1):
                src = h['metadata'].get('source', 'unknown')
                st.markdown(f"- **[{i}]** `{src}`")
                st.caption(h["text"][:300] + ("..." if len(h["text"])>300 else ""))
        st.divider()
