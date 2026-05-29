import arxiv
import os
import fitz
import urllib.request
import chromadb
import time
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.\n")

# -------------------------------------------------------
# These are the ML topics we'll pull papers from.
# Each query targets a specific subfield so the corpus
# is focused and high quality.
# -------------------------------------------------------
SEARCH_QUERIES = [
    ("cat:cs.LG AND large language models", 60),
    ("cat:cs.CL AND transformer attention mechanism", 60),
    ("cat:cs.LG AND reinforcement learning from human feedback", 40),
    ("cat:cs.LG AND diffusion models image generation", 40),
    ("cat:cs.CL AND retrieval augmented generation", 60),
    ("cat:cs.LG AND fine-tuning pretrained models", 40),
    ("cat:cs.AI AND reasoning chain of thought", 40),
    ("cat:cs.LG AND vector embeddings semantic search", 40),
    ("cat:cs.CL AND instruction tuning alignment", 40),
    ("cat:cs.LG AND neural network optimization", 40),
    # new queries below
    ("cat:cs.CL AND prompt engineering few shot", 40),
    ("cat:cs.LG AND LoRA parameter efficient fine-tuning", 40),
    ("cat:cs.AI AND AI agents tool use", 40),
    ("cat:cs.LG AND hallucination factuality language models", 40),
    ("cat:cs.CL AND question answering benchmark evaluation", 40),
    ("cat:cs.LG AND knowledge distillation compression", 30),
    ("cat:cs.LG AND mixture of experts sparse models", 30),
    ("cat:cs.CL AND summarization abstractive extractive", 30),
    ("cat:cs.AI AND multimodal vision language models", 40),
    ("cat:cs.LG AND graph neural networks", 30),

    ("cat:cs.LG AND knowledge distillation model compression", 40),
    ("cat:cs.LG AND instruction tuning RLHF alignment", 40),
    ("cat:cs.IR AND dense retrieval bi-encoder cross-encoder", 40),
    ("cat:cs.LG AND parameter efficient fine-tuning challenges", 40),
]
# total = ~500 papers

ML_CATEGORIES = {"cs.LG", "cs.CL", "cs.AI", "stat.ML"}


def fetch_papers(query, max_results):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            for result in client.results(search):
                if not any(cat in ML_CATEGORIES for cat in result.categories):
                    continue
                papers.append({
                    "id": result.entry_id,
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "abstract": result.summary,
                    "published": str(result.published),
                    "categories": result.categories,
                    "pdf_url": result.pdf_url,
                })
            break  # success, exit retry loop

        except Exception as e:
            wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s...
            print(f"  Rate limited. Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
            time.sleep(wait_time)
    
    return papers


def download_pdf(paper, download_dir="data"):
    os.makedirs(download_dir, exist_ok=True)
    paper_id = paper["id"].split("/")[-1]
    filepath = os.path.join(download_dir, f"{paper_id}.pdf")
    if os.path.exists(filepath):
        return filepath
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(paper["pdf_url"], headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(filepath, "wb") as f:
                f.write(response.read())
        return filepath
    except Exception as e:
        print(f"  Failed to download {paper_id}: {e}")
        return None


def extract_text(filepath):
    try:
        doc = fitz.open(filepath)
        full_text = ""
        for page_num, page in enumerate(doc):
            full_text += f"\n--- Page {page_num + 1} ---\n{page.get_text()}"
        doc.close()
        return full_text
    except Exception as e:
        print(f"  Failed to extract {filepath}: {e}")
        return ""


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 100:  # skip tiny chunks
            chunks.append(chunk)
        start = end - overlap
    return chunks


def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="arxiv_papers",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def get_stored_ids(collection):
    # get all ids already in the database so we don't re-embed
    results = collection.get()
    stored = set()
    for id in results["ids"]:
        paper_id = id.rsplit("_chunk_", 1)[0]
        stored.add(paper_id)
    return stored


def store_paper(paper, collection):
    chunks = paper["chunks"]
    if not chunks:
        return 0

    chunk_ids = [
        f"{paper['id'].split('/')[-1]}_chunk_{i}"
        for i in range(len(chunks))
    ]
    embeddings = embedder.encode(chunks).tolist()
    metadatas = [
        {
            "title": paper["title"],
            "authors": ", ".join(paper["authors"][:3]),
            "published": paper["published"],
            "paper_id": paper["id"].split("/")[-1],
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
    return len(chunks)


def main():
    collection = get_collection()
    stored_ids = get_stored_ids(collection)
    print(f"Already in database: {len(stored_ids)} papers, {collection.count()} chunks\n")

    total_papers = 0
    total_chunks = 0
    failed = 0

    for query, max_results in SEARCH_QUERIES:
        print(f"\nFetching: {query}")
        papers = fetch_papers(query, max_results)
        print(f"  Found {len(papers)} papers")

        for paper in papers:
            paper_id = paper["id"].split("/")[-1]

            # skip if already in database
            if paper_id in stored_ids:
                continue

            # download
            filepath = download_pdf(paper)
            if not filepath:
                failed += 1
                continue

            # extract text
            text = extract_text(filepath)
            if not text.strip():
                failed += 1
                continue

            # chunk
            chunks = chunk_text(text)
            paper["chunks"] = chunks

            # embed and store
            count = store_paper(paper, collection)
            stored_ids.add(paper_id)
            total_papers += 1
            total_chunks += count

            print(f"  [{total_papers}] {paper['title'][:55]}... → {count} chunks")

            # small pause to be polite to arxiv servers
            time.sleep(2)

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  New papers added : {total_papers}")
    print(f"  New chunks added : {total_chunks}")
    print(f"  Failed downloads : {failed}")
    print(f"  Total in database: {collection.count()} chunks")


if __name__ == "__main__":
    main()