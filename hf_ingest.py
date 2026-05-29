import chromadb
import json
import hashlib
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.\n")

# -------------------------------------------------------
# Keywords targeting the 3 weak topics from evaluation:
# - cross-encoders / bi-encoders / dense retrieval
# - fine-tuning challenges
# - instruction tuning
# -------------------------------------------------------
TOPIC_KEYWORDS = {
    "retrieval_encoders": [
        "cross-encoder", "bi-encoder", "dense retrieval",
        "dual encoder", "reranking", "passage retrieval",
        "semantic search", "dense passage"
    ],
    "finetuning_challenges": [
        "fine-tuning challenges", "catastrophic forgetting",
        "continual learning", "parameter efficient",
        "LoRA", "adapter tuning", "PEFT",
        "overfitting language model", "full fine-tuning"
    ],
    "instruction_tuning": [
        "instruction tuning", "instruction following",
        "FLAN", "InstructGPT", "supervised fine-tuning",
        "task instruction", "prompt tuning",
        "instruction dataset", "alpaca"
    ],
    "rlhf_alignment": [
        "reinforcement learning human feedback",
        "RLHF", "reward model", "PPO language",
        "preference learning", "constitutional AI",
        "DPO", "direct preference optimization"
    ]
}

ML_CATEGORIES = {"cs.LG", "cs.CL", "cs.AI", "cs.IR", "stat.ML"}


def matches_topic(paper, keywords):
    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    return any(kw.lower() in text for kw in keywords)


def get_stored_ids(collection):
    results = collection.get()
    stored = set()
    for id in results["ids"]:
        paper_id = id.rsplit("_chunk_", 1)[0]
        stored.add(paper_id)
    return stored


def chunk_text(text, chunk_size=400, overlap=40):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 80:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def store_papers(papers, collection):
    total_chunks = 0
    for paper in papers:
        paper_id = paper["id"].replace("/", "_")

        # combine title + abstract as the text
        text = f"Title: {paper.get('title','')}\n\nAbstract: {paper.get('abstract','')}"
        chunks = chunk_text(text)

        if not chunks:
            continue

        chunk_ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
        embeddings = embedder.encode(chunks).tolist()
        metadatas = [
            {
                "title": paper.get("title", "Unknown"),
                "authors": "Unknown",
                "published": "Unknown",
                "paper_id": paper_id,
                "section": "abstract",
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]

        collection.upsert(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        total_chunks += len(chunks)

    return total_chunks


def main():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="arxiv_smart",
        metadata={"hnsw:space": "cosine"}
    )
    stored_ids = get_stored_ids(collection)
    print(f"Current database: {collection.count()} chunks, {len(stored_ids)} papers\n")

    print("Loading HuggingFace ArXiv dataset (streaming — no full download)...")
    print("This streams papers one by one so it starts immediately.\n")

    # stream the dataset — don't download all 2M papers
    dataset = load_dataset(
        "CShorten/ML-ArXiv-Papers",
        split="train",
        streaming=True
    )

    # collect matched papers per topic
    topic_papers = {topic: [] for topic in TOPIC_KEYWORDS}
    target_per_topic = 150  # papers per topic
    total_scanned = 0
    max_scan = 300000  # scan up to 300K papers to find matches

    print("Scanning papers for target topics...")
    for paper in dataset:
        total_scanned += 1

        # this dataset is already filtered to cs.LG — no category check needed
        # generate a unique id from title since 'id' field may not exist
        title = paper.get("title", "")
        if not title:
            continue
        
        import hashlib
        paper_id = hashlib.md5(title.encode()).hexdigest()[:16]
        paper["id"] = paper_id
        
        if paper_id in stored_ids:
            continue
        # check each topic
        for topic, keywords in TOPIC_KEYWORDS.items():
            if len(topic_papers[topic]) < target_per_topic:
                if matches_topic(paper, keywords):
                    topic_papers[topic].append(paper)
                    break  # don't double-count same paper

        # progress update every 10K
        if total_scanned % 10000 == 0:
            counts = {t: len(p) for t, p in topic_papers.items()}
            print(f"  Scanned {total_scanned:,} papers | Found: {counts}")

        # stop if we have enough for all topics
        if all(len(p) >= target_per_topic for p in topic_papers.values()):
            print(f"  All topics filled after scanning {total_scanned:,} papers!")
            break

        if total_scanned >= max_scan:
            print(f"  Reached scan limit of {max_scan:,} papers")
            break

    print("\nPapers found per topic:")
    all_papers = []
    seen_ids = set()
    for topic, papers in topic_papers.items():
        print(f"  {topic:<25} {len(papers)} papers")
        for p in papers:
            pid = p.get("id", "").replace("/", "_")
            if pid not in seen_ids:
                all_papers.append(p)
                seen_ids.add(pid)

    print(f"\nTotal unique papers to add: {len(all_papers)}")
    print("Embedding and storing...")

    # store in batches of 50
    batch_size = 50
    total_added = 0
    for i in range(0, len(all_papers), batch_size):
        batch = all_papers[i:i+batch_size]
        chunks_added = store_papers(batch, collection)
        total_added += len(batch)
        print(f"  Stored {total_added}/{len(all_papers)} papers ({chunks_added} chunks in this batch)")

    print(f"\nDone!")
    print(f"  Papers added    : {len(all_papers)}")
    print(f"  Total in database: {collection.count()} chunks")


if __name__ == "__main__":
    main()