# 🔬 ArXiv ML Research Assistant

> A production-grade RAG system that answers questions about ML research by searching 
> and synthesizing information from 700+ ArXiv papers in real time.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)
![LLaMA](https://img.shields.io/badge/LLaMA_3.3_70B-Groq-purple)

---

## 🎯 What It Does

Ask any question about ML research in plain English. The system searches 10,000+ 
chunks across 700+ ArXiv papers and answers strictly from the retrieved content — 
no hallucination, every claim cited.

**Example queries:**
- *"What methods reduce hallucination in LLMs?"*
- *"How does chain of thought reasoning improve performance?"*
- *"What evaluation metrics are used for RAG systems?"*
- *"How does knowledge distillation compress large models?"*

---

## 📊 Evaluation Results

Evaluated using a custom LLM-based evaluation framework across 10 test questions:

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | **0.79** | Answers grounded in retrieved sources |
| **Answer Relevancy** | **0.93** | Answers directly address the question |
| **Context Quality** | **0.64** | Retrieved chunks are relevant |
| **Overall** | **0.79** | Average across all metrics |

> Scores improved significantly after targeted corpus expansion:
> faithfulness improved from 0.51 → 0.79 (+55%) by identifying weak topics 
> through evaluation and adding focused papers.

---

## 🏗️ Architecture
User Query
│
▼
┌─────────────────┐
│  Bi-Encoder     │  all-MiniLM-L6-v2
│  Retrieval      │  Fetch top-20 candidates
└────────┬────────┘
│
▼
┌─────────────────┐
│  Cross-Encoder  │  ms-marco-MiniLM-L-6-v2
│  Reranker       │  Score and rerank to top-5
└────────┬────────┘
│
▼
┌─────────────────┐
│  LLM Generation │  LLaMA 3.3 70B via Groq
│  with Context   │  Grounded answer + citations
└────────┬────────┘
│
▼
Cited Answer
**Key design decisions:**
- **Two-stage retrieval** — bi-encoder for speed, cross-encoder for precision
- **Section-aware chunking** — splits by Abstract/Methods/Results, not arbitrary word count
- **Strict grounding** — system prompt enforces citation of sources only
- **Custom evaluation** — LLM-based faithfulness and relevancy scoring

---

## 📚 Knowledge Base

| Source | Papers | Method |
|--------|--------|--------|
| ArXiv API (cs.LG, cs.CL, cs.AI) | 301 papers | Full PDF extraction |
| HuggingFace ML-ArXiv-Papers | 485 papers | Abstract + title |
| **Total** | **786 papers** | **10,297+ chunks** |

Topics covered: LLMs, RAG, Chain-of-Thought, RLHF, Diffusion Models, 
Fine-tuning, Knowledge Distillation, Hallucination Detection, Instruction Tuning,
Dense Retrieval, and more.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Vector Database | ChromaDB (persistent, local) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | LLaMA 3.3 70B via Groq API |
| Backend | FastAPI + Server-Sent Events streaming |
| Frontend | Vanilla JS + HTML/CSS |
| PDF Parsing | PyMuPDF |

---

## 🚀 Running Locally

### Prerequisites
- Python 3.10+
- Groq API key (free at console.groq.com)

### Setup

```bash
# Clone the repo
git clone https://github.com/Architaa-1010/arxiv-rag.git
cd arxiv-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo "GROQ_API_KEY=your_key_here" > .env
```

### Build the knowledge base

```bash
# Fetch and ingest ArXiv papers (takes ~30 mins, downloads PDFs)
python build_corpus.py

# Re-chunk with section-aware splitting
python smart_ingest.py

# Add targeted papers from HuggingFace
python hf_ingest.py
```

### Run the app

```bash
# Terminal 1: start backend
uvicorn api:app --reload --port 8000

# Terminal 2: open frontend
start frontend/index.html  # Windows
open frontend/index.html   # Mac
```

---

## 📁 Project Structure
arxiv-rag/
├── api.py              # FastAPI backend with streaming
├── app.py              # Streamlit prototype (Week 2)
├── build_corpus.py     # ArXiv paper fetcher
├── smart_ingest.py     # Section-aware chunking pipeline
├── hf_ingest.py        # HuggingFace dataset ingestion
├── reranker.py         # Two-stage retrieval module
├── evaluate.py         # Custom RAG evaluation framework
├── main.py             # CLI interface
├── frontend/
│   └── index.html      # Production web UI
└── data/               # Downloaded PDFs (not in repo)
---

## 🔍 How It Works

### 1. Ingestion Pipeline
Papers are fetched from ArXiv API filtered to ML categories (cs.LG, cs.CL, cs.AI). 
PDFs are parsed with PyMuPDF, then split using **section-aware chunking** — detecting 
headers like "Abstract", "Methods", "Results" and splitting each section independently. 
This preserves semantic coherence compared to naive fixed-size chunking.

### 2. Two-Stage Retrieval
**Stage 1 — Bi-encoder retrieval:** The query is embedded using `all-MiniLM-L6-v2` 
and the top-20 most similar chunks are fetched from ChromaDB using cosine similarity.

**Stage 2 — Cross-encoder reranking:** Each of the 20 candidates is scored by 
`ms-marco-MiniLM-L-6-v2`, which jointly encodes the query and chunk to produce 
a precise relevance score. The top-5 are passed to the LLM.

### 3. Grounded Generation
The top-5 chunks are passed as context to LLaMA 3.3 70B with a strict system prompt 
that enforces source citation and prohibits answering from training knowledge.

### 4. Evaluation
A custom evaluation framework measures faithfulness (are claims supported by sources?), 
answer relevancy (does the answer address the question?), and context quality 
(are retrieved chunks relevant?). Scores improved from 0.647 → 0.787 overall after 
targeted corpus expansion.

---

## 📈 What I Learned

- Naive fixed-size chunking significantly hurts retrieval quality — section-aware 
  chunking improved answer grounding
- Two-stage retrieval (bi-encoder + cross-encoder) meaningfully improves precision 
  over single-stage semantic search
- Evaluation-driven development: measuring faithfulness scores exposed corpus gaps, 
  which guided targeted data collection
- Small local LLMs (llama3.2:1b) don't follow grounding instructions reliably — 
  larger models (70B) are necessary for faithful RAG

---

## 🗺️ Future Work

- [ ] Add citation graph — surface foundational papers behind any answer
- [ ] Weekly digest mode — auto-summarize new cs.LG papers
- [ ] Deploy to cloud (Render/Railway)
- [ ] Add BM25 hybrid retrieval alongside dense retrieval
- [ ] Expand corpus to 5,000+ papers

---

*Built as a portfolio project to demonstrate end-to-end ML engineering:
retrieval systems, vector databases, LLM integration, and evaluation-driven development.*