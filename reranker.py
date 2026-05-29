from sentence_transformers import CrossEncoder
import chromadb
from sentence_transformers import SentenceTransformer

# load both models
print("Loading models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Models ready.\n")


def retrieve_and_rerank(query, collection, fetch_k=20, final_k=5):
    """
    Two-stage retrieval:
    Stage 1 - fetch_k=20 candidates using fast bi-encoder
    Stage 2 - rerank them, return final_k=5 best using cross-encoder
    """

    # stage 1: fast retrieval
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_k
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

    # stage 2: rerank using cross-encoder
    # cross-encoder takes [query, chunk] pairs and scores them
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    # attach scores and sort
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    return ranked[:final_k]


def test_comparison(query, collection):
    """Show side by side: before and after reranking"""

    print(f"Query: '{query}'\n")

    # WITHOUT reranking — just top 5 from bi-encoder
    query_embedding = embedder.encode([query]).tolist()
    basic_results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )
    print("WITHOUT reranking (bi-encoder only):")
    for i, (doc, meta) in enumerate(zip(
        basic_results["documents"][0],
        basic_results["metadatas"][0]
    )):
        print(f"  [{i+1}] {meta['title'][:60]}... (section: {meta.get('section','?')})")

    print()

    # WITH reranking
    reranked = retrieve_and_rerank(query, collection)
    print("WITH reranking (cross-encoder):")
    for i, chunk in enumerate(reranked):
        print(f"  [{i+1}] {chunk['title'][:60]}... "
              f"(section: {chunk['section']}, score: {chunk['rerank_score']:.3f})")


if __name__ == "__main__":
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="arxiv_smart",
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Collection loaded: {collection.count()} chunks\n")
    print("="*60)

    test_comparison(
        "how does LoRA reduce the number of trainable parameters",
        collection
    )

    print("\n" + "="*60 + "\n")

    test_comparison(
        "what evaluation metrics are used for RAG systems",
        collection
    )