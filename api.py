from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
import os
import json
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="arxiv_smart",
    metadata={"hnsw:space": "cosine"}
)
print(f"Ready. {collection.count()} chunks loaded.\n")


class ChatRequest(BaseModel):
    question: str
    history: list = []


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


@app.get("/health")
def health():
    return {"status": "ok", "chunks": collection.count()}


@app.post("/chat")
async def chat(req: ChatRequest):
    chunks = retrieve_and_rerank(req.question)
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
    for msg in req.history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.question})

    # deduplicate sources for response
    seen = set()
    unique_sources = []
    for chunk in chunks:
        if chunk["paper_id"] not in seen:
            unique_sources.append({
                "title": chunk["title"],
                "authors": chunk["authors"],
                "published": chunk["published"][:10],
                "paper_id": chunk["paper_id"],
                "section": chunk["section"],
                "url": f"https://arxiv.org/abs/{chunk['paper_id']}"
            })
            seen.add(chunk["paper_id"])

    def stream():
        # first send sources as a JSON line
        yield f"data: {json.dumps({'type': 'sources', 'sources': unique_sources})}\n\n"

        # then stream the answer token by token
        stream_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            stream=True
        )
        for chunk in stream_response:
            token = chunk.choices[0].delta.content
            if token:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")