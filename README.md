# 📊 Annual Report Q&A — RAG Application

A Retrieval-Augmented Generation (RAG) application that lets you upload any company's annual report and ask natural language questions about it. Built with LangChain, FAISS, Groq (Llama 3), and Streamlit.

🔴 **[Live Demo](https://annual-report-rag-chatbot-by-geetanshu.streamlit.app/)**

---

## What it does

Upload any annual report PDF — TCS, Infosys, Apple, Tesla, or any company — and ask questions like:

- *"What was the total revenue for FY 2026?"*
- *"Who is the CEO?"*
- *"What are the major risk factors?"*
- *"What is the dividend declared this year?"*

The app reads the document, finds the most relevant sections, and generates a precise, grounded answer — no hallucination, no guessing.

---

## How it works

```
PDF Upload  →  Text Extraction  →  Chunking (1000 chars, 100 overlap)
                                          ↓
                              Embedding (all-MiniLM-L6-v2)
                                          ↓
                               FAISS Vector Store (in-memory)
                                          ↓
          User Question  →  Similarity Search (top 6 chunks)
                                          ↓
                         Prompt Template + Groq LLM (Llama 3.1)
                                          ↓
                                  Grounded Answer
```

This is a standard RAG (Retrieval-Augmented Generation) pipeline:
1. **Indexing** — The PDF is split into chunks and converted to vectors using a HuggingFace embedding model
2. **Retrieval** — The user's question is converted to a vector and FAISS finds the most semantically similar chunks
3. **Generation** — The retrieved chunks and question are sent to Llama 3.1 via Groq API, which generates a grounded answer

---

## Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain (LCEL) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (free, local) |
| Vector Store | FAISS (in-memory) |
| LLM | Llama 3.1 8B via Groq API (free) |
| PDF Loader | PyPDFLoader |
| Frontend | Streamlit |
| Secret Management | python-dotenv |

---

## Project Structure

```
annual-report-qa/
├── app.py              ← complete RAG application
├── requirements.txt    ← all dependencies with pinned versions
├── .env                ← API keys (not committed to GitHub)
├── .gitignore          ← keeps secrets out of version control
└── README.md
```

---

## Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-here.git
cd annual-report-qa
```

### 2. Create and activate virtual environment
```bash
conda create -n rag-project python=3.10
conda activate rag-project
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

### 5. Run the app
```bash
streamlit run app.py
```

---

## Key Learnings & Design Decisions

**Chunk size of 1000 characters with 100 overlap** — Annual reports have dense financial tables and long paragraphs. Smaller chunks lose context; larger chunks overwhelm the LLM. 100-character overlap prevents answers from being split across chunk boundaries.

**k=6 retrieval** — Tested with k=3 initially. For broad questions like risk factors, k=3 returned shallow answers. k=6 retrieves enough context for comprehensive answers without exceeding the LLM's context window.

**Groq over HuggingFace Inference API** — HuggingFace's free inference API has become restrictive with model support. Groq provides faster, more reliable free access to Llama 3.1.

**`@st.cache_resource` for vector store** — Rebuilding the FAISS index on every question would take 60+ seconds. Caching it in memory after the first PDF upload makes subsequent questions near-instant.


## Geetanshu Singh Rajawat
**Geetanshu Singh Rajawat**  
ML Engineer | MSc AI & ML (ongoing)  
[LinkedIn](https://www.linkedin.com/in/your-linkedin) | [GitHub](https://github.com/your-username)
