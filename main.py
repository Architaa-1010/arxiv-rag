import chromadb
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.\n")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="arxiv_smart",
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def retrieve_chunks(query, collection, n_results=5):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "title": meta["title"],
            "authors": meta["authors"],
            "published": meta["published"],
            "paper_id": meta["paper_id"]
        })
    return chunks

def build_context(chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"""
Source [{i+1}]
Paper  : {chunk['title']}
Authors: {chunk['authors']}
Published: {chunk['published']}
Content: {chunk['text']}
---
"""
    return context

def ask(question, collection, chat_history):
    print("Searching knowledge base...")
    chunks = retrieve_chunks(question, collection)
    context = build_context(chunks)

    system_prompt = """You are a research assistant that answers questions strictly based on ML research papers.

STRICT RULES:
- Answer ONLY using information from the provided sources below
- NEVER use your own training knowledge to answer
- If the sources don't contain enough information, say: "I don't have enough information in my knowledge base to answer this fully."
- Always cite sources like [1], [2] etc after every claim
- Be specific and technical — this is for ML researchers

Here are the retrieved sources:

""" + context

    chat_history.append({
        "role": "user",
        "content": question
    })

    print("Thinking...\n")
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            *chat_history
        ],
        max_tokens=1024,
        temperature=0.1  # low temperature = more focused, less creative
    )

    answer = response.choices[0].message.content

    chat_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer, chunks

def print_sources(chunks):
    print("\nSources used:")
    seen = set()
    for i, chunk in enumerate(chunks):
        paper_id = chunk["paper_id"]
        if paper_id not in seen:
            print(f"  [{i+1}] {chunk['title'][:70]}")
            print(f"       {chunk['authors']}")
            print(f"       https://arxiv.org/abs/{paper_id}")
            seen.add(paper_id)

def main():
    print("="*60)
    print("  ArXiv Research Assistant")
    print("="*60)

    collection = get_collection()
    total = collection.count()
    print(f"Knowledge base loaded: {total} chunks from research papers")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    chat_history = []

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        answer, chunks = ask(question, collection, chat_history)
        print(f"\nAssistant: {answer}")
        print_sources(chunks)
        print()

if __name__ == "__main__":
    main()