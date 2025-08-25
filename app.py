import os
import streamlit as st
from dotenv import load_dotenv
from rag.ingest import load_folder
from rag.chunk import make_docs_chunks
from rag.embed_gemini import embed_texts
from rag.vectorstore import DiskVectorStore
from rag.qa import answer_question

load_dotenv()

st.set_page_config(
    page_title="Internal Docs AI Assistant",
    page_icon="",
    layout="wide"
)

# --------- HEADER ---------
st.markdown("""
<style>
.hero-title {
    font-size: 4rem;   /* Increased from 2.8rem to 4rem */
    font-weight: 800;
    text-align: center;
    margin-bottom: 10px;
    background: linear-gradient(90deg, #007BFF, #00C9A7, #6A5ACD);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientMove 6s infinite alternate;
    letter-spacing: 1px;
}
.hero-subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #444;
    margin-bottom: 25px;
    opacity: 0.9;
}
.background-lines {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 200px;
    background: repeating-linear-gradient(
        45deg,
        rgba(0, 123, 255, 0.08) 0px,
        rgba(0, 123, 255, 0.08) 2px,
        transparent 2px,
        transparent 6px
    );
    animation: moveLines 10s linear infinite;
    z-index: -1;
}
@keyframes gradientMove {
    from { background-position: 0% 50%; }
    to { background-position: 100% 50%; }
}
@keyframes moveLines {
    from { background-position: 0 0; }
    to { background-position: 200px 200px; }
}
</style>

<div class="background-lines"></div>
<h1 class="hero-title">Internal Docs AI Assistant</h1>
""", unsafe_allow_html=True)



# --------- SIDEBAR ---------
with st.sidebar:
    st.title("Configuration")

    st.subheader("Workspace")
    workspace = st.text_input("Namespace", value="default")
    index_root = os.getenv("INDEX_ROOT", ".index")
    index_dir = os.path.join(index_root, workspace)
    st.caption(f"Index path: `{index_dir}`")

    st.subheader("Indexing")
    max_chars = st.slider("Chunk size (characters)", 400, 2000, 1000, step=100)
    overlap = st.slider("Overlap (characters)", 50, 500, 200, step=50)

    st.subheader("Retrieval")
    top_k = st.slider("Top-K Results", 1, 10, 4)

    st.subheader("Models")
    st.text_input("GEN_MODEL", value=os.getenv("GEN_MODEL", "models/gemini-1.5-flash"), key="gen_model")
    st.text_input("EMBED_MODEL", value=os.getenv("EMBED_MODEL", "models/text-embedding-004"), key="emb_model")
    st.caption("Make sure GOOGLE_API_KEY is set in your environment or .env file.")

# --------- MAIN CONTENT ---------
tab_ingest, tab_chat = st.tabs(["Ingest Documents", "Chat with Docs"])

# ---- Ingest Tab ----
with tab_ingest:
    st.subheader("Upload or Select Folder")
    mode = st.radio("Choose input mode", ["Upload files", "Use a folder path"], horizontal=True)

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    if mode == "Upload files":
        ups = st.file_uploader(
            "Upload multiple documents (.txt, .md, .pdf, .docx)",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True
        )
    else:
        folder = st.text_input("Docs folder path", value="sample_docs")

    colA, colB, colC = st.columns([1, 1, 1])

    with colA:
        if st.button("Build Index", type="primary", use_container_width=True):
            if mode == "Upload files":
                if not ups:
                    st.error("Please upload at least one file.")
                    st.stop()
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
                st.error("No content to index. Check folder path and file types.")
            else:
                with st.status("Embedding & saving index...", expanded=True):
                    store = DiskVectorStore(index_dir=index_dir)
                    store.build(all_chunks, embed_texts)
                st.success(f"Index built at `{index_dir}`.")

    with colB:
        if st.button("Clear Index", use_container_width=True):
            store = DiskVectorStore(index_dir=index_dir)
            store.clear()
            st.success("Index cleared successfully.")

    with colC:
        if st.button("Show Sources", use_container_width=True):
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

# ---- Chat Tab ----
with tab_chat:
    st.subheader("Chat with your Documents")

    # Inject CSS for modern chat UI
    st.markdown("""
    <style>
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 15px;
        border-radius: 12px;
        background-color: #fafafa;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }
    .chat-bubble-user {
        background: #d1e9ff;
        color: #000;
        padding: 12px 16px;
        border-radius: 15px 15px 0 15px;
        margin: 10px 0;
        max-width: 70%;
        margin-left: auto;
        animation: slideInRight 0.4s ease;
    }
    .chat-bubble-ai {
        background: #f1f3f4;
        color: #000;
        padding: 12px 16px;
        border-radius: 15px 15px 15px 0;
        margin: 10px 0;
        max-width: 70%;
        margin-right: auto;
        animation: slideInLeft 0.4s ease;
    }
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Chat history area
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for item in st.session_state.chat:
        st.markdown(f"<div class='chat-bubble-user'><b>You:</b><br>{item['q']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-ai'><b>AI:</b><br>{item['answer']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input at bottom
    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input("Type your question...", placeholder="Ask me anything from your docs")
        submitted = st.form_submit_button("Send")
    if submitted and q.strip():
        try:
            with st.spinner("Thinking..."):
                out = answer_question(q, index_dir=index_dir, top_k=top_k)
            st.session_state.chat.append(out | {"q": q})
            st.rerun()
        except Exception as e:
            st.error(str(e))
