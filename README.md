# 📚 Internal Docs Q&A Agent

An AI-powered internal documentation assistant built for the **AI Agent Hackathon (by Project Space)**.  
This app indexes team documents (PDF, DOCX, TXT, MD) and allows users to ask natural language questions such as:  

👉 *“What’s our refund policy?”*  
👉 *“How to request design assets?”*  

The system retrieves relevant content from your docs and provides concise answers with references.

---

## ✨ Features
- 🔍 **Multi-format ingestion** — supports `.pdf`, `.docx`, `.txt`, `.md`
- 📑 **Smart chunking** of large documents
- 🤖 **Embeddings with Gemini** for semantic search
- ⚡ **Vector search** using cosine similarity
- 💬 **Natural language Q&A** powered by Google Generative AI
- 🖥️ **Streamlit interface** (easy to use, clean UI)

---

## 📂 Project Structure
├── app.py # Streamlit app (UI)

├── rag/

│ ├── ingest.py # Document ingestion

│ ├── chunk.py # Chunking docs into pieces

│ ├── embed_gemini.py # Embedding using Gemini

│ ├── vectorstore.py # Disk-based vector index

│ └── qa.py # Question answering

├── sample_docs/ # Example documents

├── requirements.txt # Dependencies

└── README.md # Project documentation

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repo

git clone https://github.com/likhithmessi10/internal-doc-ai.git

cd internal-doc-ai

### 2️⃣ Create a virtual environment
python -m venv venv

source venv/bin/activate   # Mac/Linux

venv\Scripts\activate      # Windows


### 3️⃣ Install dependencies
pip install -r requirements.txt

### 4️⃣ Add API key
create a .env file in the project root:

GOOGLE_API_KEY=your_api_key_here

INDEX_DIR=.index


### 5️⃣ Run the app
streamlit run app.py

---

## 🚀 Usage

Upload or point to your documents (PDF, DOCX, TXT, MD).

Click Build Index to preprocess & embed them.

Ask natural language questions in the Ask tab.

Get instant answers with citations.

---

## 🛠️ Tech Stack

Streamlit
 – UI

Google Generative AI
 – LLM & embeddings

NumPy
 – Vector math

Python-docx
 – Word docs

PyPDF
 – PDF parsing

 ---

 ## 👨‍💻 Team

Built by Likhith
 & Surya Vanapalli @SuryaVanapalli14

for the AI Agent Hackathon by Project Space (Virtual) 🎉
