import streamlit as st
import chromadb
import os
import zipfile
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="ArXiv ML Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #F7F5F2; }
[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E5E7EB; }
[data-testid="stMetricValue"] { color: #C96C4A !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #6B7280 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
.stButton button { background-color: #F7F5F2; border: 1px solid #E5E7EB; border-radius: 8px; color: #374151; font-size: 12px; text-align: left; }
.stButton button:hover { background-color: #C96C4A; border-color: #C96C4A; color: white; }
[data-testid="stChatInput"] textarea { background-color: #F7F5F2 !important; border: 1px solid #E5E7EB !important; }
h1 { background: linear-gradient(135deg, #C96C4A, #D9A441); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800 !important; }
a { color: #C96C4A !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def setup():
    # unzip chroma if needed
    if not os.path.exists("./chroma_deploy") and os.path.exists("chroma_deploy.zip"):
        st.info("Setting up knowledge base for first run...")
        with zipfile.ZipFile("chroma_deploy.zip", "r") as z:
            z.extractall(".")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    client = chromadb.PersistentClient(path="./chroma_deploy")
    collection = client.get_or_create_collection(
        name="arxiv_smart",
        metadata={"hnsw:space": "cosine"}
    )
    
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    return embedder, reranker, collection, groq_client


embedder, reranker, collection, groq_client = setup()


def retrieve_and_rerank(query, n_results=5):
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
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:n_results]


def ask(question, chat_history):
    chunks = retrieve_and_rerank(question)
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"Source [{i+1}] — {chunk['section'].upper()}\nPaper: {chunk['title']}\nContent: {chunk['text']}\n---\n"

    system_prompt = """You are a research assistant that answers questions strictly based on ML research papers.
RULES: Answer ONLY from provided sources. Cite as [1],[2]. If insufficient info, say so honestly.

Sources:
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


# sidebar
with st.sidebar:
    st.markdown("### 🔬 ArXiv Assistant")
    st.markdown("<p style='color:#6B7280;font-size:12px;margin-top:-10px'>ML Research · RAG System</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Papers", "786")
    with col2:
        st.metric("Chunks", f"{collection.count():,}")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Faithfulness", "0.79")
    with col4:
        st.metric("Relevancy", "0.93")

    st.markdown("---")
    st.markdown("**Example questions:**")
    
    examples = [
        "What methods reduce hallucination in LLMs?",
        "How does chain of thought reasoning work?",
        "What is retrieval augmented generation?",
        "How does RLHF train language models?",
        "How do cross-encoders differ from bi-encoders?",
    ]
    for q in examples:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# main
st.title("ArXiv ML Research Assistant")
st.caption("Answers grounded in 786 ML papers · Two-stage retrieval with cross-encoder reranking · Powered by LLaMA 3.3 70B")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Sources"):
                seen = set()
                for i, c in enumerate(msg["sources"]):
                    if c["paper_id"] not in seen:
                        st.markdown(f"**[{i+1}] {c['title']}**")
                        st.caption(f"{c['authors']} · {c['published'][:10]} · {c['section']}")
                        st.markdown(f"[View on ArXiv](https://arxiv.org/abs/{c['paper_id']})")
                        st.divider()
                        seen.add(c["paper_id"])

if "pending" in st.session_state:
    user_input = st.session_state.pending
    del st.session_state.pending
else:
    user_input = st.chat_input("Ask a question about ML research...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving → reranking → generating..."):
            answer, sources = ask(user_input, st.session_state.messages[:-1])
        st.markdown(answer)
        with st.expander("📄 Sources"):
            seen = set()
            for i, c in enumerate(sources):
                if c["paper_id"] not in seen:
                    st.markdown(f"**[{i+1}] {c['title']}**")
                    st.caption(f"{c['authors']} · {c['published'][:10]} · {c['section']}")
                    st.markdown(f"[View on ArXiv](https://arxiv.org/abs/{c['paper_id']})")
                    st.divider()
                    seen.add(c["paper_id"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })