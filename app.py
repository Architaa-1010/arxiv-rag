import streamlit as st
import chromadb
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

st.set_page_config(
    page_title="ArXiv Research Assistant",
    page_icon="🔬",
    layout="wide"
)
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #0f1117;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a2f3e;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #8b92a5 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #6366f1 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* Sidebar buttons (example questions) */
[data-testid="stSidebar"] .stButton button {
    background-color: #1e2433;
    color: #c9d1e0;
    border: 1px solid #2a2f3e;
    border-radius: 8px;
    text-align: left;
    font-size: 13px;
    padding: 8px 12px;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #6366f1;
    border-color: #6366f1;
    color: white;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background-color: #1e2433 !important;
    border: 1px solid #2a2f3e !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 15px !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #1a1f2e;
    border-radius: 12px;
    border: 1px solid #2a2f3e;
    padding: 12px;
    margin-bottom: 8px;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #161b27;
    border-radius: 12px;
    border: 1px solid #6366f120;
    padding: 12px;
    margin-bottom: 8px;
}

/* Expander (sources) */
[data-testid="stExpander"] {
    background-color: #1a1f2e;
    border: 1px solid #2a2f3e !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #6366f1 !important;
    font-weight: 600;
}

/* Title */
h1 {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
}

/* Caption under title */
.stApp [data-testid="stCaptionContainer"] p {
    color: #8b92a5 !important;
    font-size: 13px;
}

/* Links inside sources */
a {
    color: #6366f1 !important;
    text-decoration: none !important;
}
a:hover {
    text-decoration: underline !important;
}

/* Spinner */
[data-testid="stSpinner"] {
    color: #6366f1 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2a2f3e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }
</style>
""", unsafe_allow_html=True)
# -------------------------------------------------------
# load models once and cache them — streamlit reruns the
# whole script on every interaction so caching is important
# -------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(
        name="arxiv_smart",
        metadata={"hnsw:space": "cosine"}
    )

@st.cache_resource
def load_groq():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

embedder = load_embedder()
collection = load_collection()
groq_client = load_groq()

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = load_reranker()

def retrieve_chunks(query, n_results=5):
    # stage 1: fetch 20 candidates
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=20
    )
    candidates = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        candidates.append({
            "text": doc,
            "title": meta["title"],
            "authors": meta["authors"],
            "published": meta["published"],
            "paper_id": meta["paper_id"],
            "section": meta.get("section", "unknown")
        })

    # stage 2: rerank and return top 5
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:n_results]


def build_context(chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"""
Source [{i+1}] — {chunk['section'].upper()} section
Paper  : {chunk['title']}
Authors: {chunk['authors']}
Content: {chunk['text']}
---
"""
    return context


def ask(question, chat_history):
    chunks = retrieve_chunks(question)
    context = build_context(chunks)

    system_prompt = """You are a research assistant that answers questions strictly based on ML research papers.

STRICT RULES:
- Answer ONLY using information from the provided sources
- NEVER use your own training knowledge
- Always cite sources like [1], [2] after every claim
- If sources lack information, say so honestly
- Be specific and technical

Retrieved sources:
""" + context

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024,
        temperature=0.1
    )

    return response.choices[0].message.content, chunks


# -------------------------------------------------------
# UI layout
# -------------------------------------------------------

# sidebar
with st.sidebar:
    st.markdown("### 🔬 ArXiv Assistant")
    st.markdown("<p style='color:#8b92a5;font-size:12px;margin-top:-10px'>ML Research · RAG System</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.metric("Papers in KB", "301")
    st.metric("Total Chunks", f"{collection.count():,}")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("Ask any question about ML research. The assistant searches 301 ArXiv papers and answers strictly from them.")
    st.markdown("---")
    st.markdown("**Example questions:**")
    example_questions = [
        "What methods reduce hallucination in LLMs?",
        "How does chain of thought reasoning work?",
        "What is retrieval augmented generation?",
        "How does RLHF train language models?",
        "What are the latest diffusion model techniques?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# main area
st.title("ArXiv ML Research Assistant")
st.caption("Answers grounded in 301 ML papers from ArXiv · Powered by Groq + LLaMA 3.3 70B")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 View sources"):
                seen = set()
                for i, chunk in enumerate(msg["sources"]):
                    if chunk["paper_id"] not in seen:
                        st.markdown(f"**[{i+1}] {chunk['title']}**")
                        st.caption(f"{chunk['authors']} · {chunk['published'][:10]} · Section: {chunk['section']}")
                        st.markdown(f"[View on ArXiv](https://arxiv.org/abs/{chunk['paper_id']})")
                        st.markdown("---")
                        seen.add(chunk["paper_id"])

# handle example question button clicks
if "pending_question" in st.session_state:
    user_input = st.session_state.pending_question
    del st.session_state.pending_question
else:
    user_input = st.chat_input("Ask a question about ML research...")

# process input
if user_input:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # get answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving → reranking → generating..."):
            answer, sources = ask(user_input, st.session_state.messages[:-1])
        st.markdown(answer)
        with st.expander("📄 View sources"):
            seen = set()
            for i, chunk in enumerate(sources):
                if chunk["paper_id"] not in seen:
                    st.markdown(f"**[{i+1}] {chunk['title']}**")
                    st.caption(f"{chunk['authors']} · {chunk['published'][:10]} · Section: {chunk['section']}")
                    st.markdown(f"[View on ArXiv](https://arxiv.org/abs/{chunk['paper_id']})")
                    st.markdown("---")
                    seen.add(chunk["paper_id"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })