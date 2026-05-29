import arxiv
import os
import fitz
import urllib.request
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# load embedding model once (downloads on first run, ~90MB)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.")

def fetch_papers(query, max_results=10):
    print(f"\nSearching ArXiv for: {query}")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in client.results(search):
        categories = result.categories
        ml_categories = {"cs.LG", "cs.CL", "cs.AI", "stat.ML"}
        if not any(cat in ml_categories for cat in categories):
            print(f"  Skipping ({categories[0]}): {result.title[:50]}...")
            continue
        paper = {
            "id": result.entry_id,
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary,
            "published": str(result.published),
            "categories": categories,
            "pdf_url": result.pdf_url,
        }
        papers.append(paper)
        print(f"  Found: {result.title[:70]}...")
    return papers


def download_pdf(paper, download_dir="data"):
    os.makedirs(download_dir, exist_ok=True)
    paper_id = paper["id"].split("/")[-1]
    filepath = os.path.join(download_dir, f"{paper_id}.pdf")
    if os.path.exists(filepath):
        print(f"  Already exists, skipping: {paper_id}")
        return filepath
    print(f"  Downloading: {paper['title'][:60]}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(paper["pdf_url"], headers=headers)
    with urllib.request.urlopen(req) as response:
        with open(filepath, "wb") as f:
            f.write(response.read())
    print(f"  Saved to: {filepath}")
    return filepath


def extract_text(filepath):
    doc = fitz.open(filepath)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    doc.close()
    return full_text


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def store_in_chromadb(papers):
    # connect to (or create) a local ChromaDB database
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # create a collection (like a table) called "arxiv_papers"
    # if it already exists, just connect to it
    collection = client.get_or_create_collection(
        name="arxiv_papers",
        metadata={"hnsw:space": "cosine"}  # cosine similarity for semantic search
    )
    
    total_chunks = 0
    
    for paper in papers:
        print(f"\nEmbedding: {paper['title'][:60]}...")
        
        chunks = paper["chunks"]
        
        # create a unique id for each chunk
        chunk_ids = [
            f"{paper['id'].split('/')[-1]}_chunk_{i}"
            for i in range(len(chunks))
        ]
        
        # embed all chunks at once (converts text -> vectors)
        print(f"  Generating {len(chunks)} embeddings...")
        embeddings = embedder.encode(chunks).tolist()
        
        # metadata stored alongside each chunk for display later
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
        
        # store everything in ChromaDB
        collection.upsert(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        total_chunks += len(chunks)
        print(f"  Stored {len(chunks)} chunks in ChromaDB")
    
    print(f"\nDone. Total chunks in database: {collection.count()}")
    return collection


def search(query, collection, n_results=3):
    print(f"\nSearching for: '{query}'")
    
    # embed the query using the same model
    query_embedding = embedder.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    
    print(f"\nTop {n_results} results:\n")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"Result {i+1}")
        print(f"  Paper : {meta['title'][:70]}")
        print(f"  Authors: {meta['authors']}")
        print(f"  Chunk  : {doc[:200]}...")
        print()
    
    return results


if __name__ == "__main__":
    # step 1: fetch papers
    papers = fetch_papers("cat:cs.LG AND LoRA fine-tuning", max_results=5)
    
    # step 2: download pdfs
    print("\n--- Downloading PDFs ---")
    for paper in papers:
        filepath = download_pdf(paper)
        paper["filepath"] = filepath
    
    # step 3: extract and chunk text
    print("\n--- Extracting text ---")
    for paper in papers:
        text = extract_text(paper["filepath"])
        paper["chunks"] = chunk_text(text)
        print(f"  {paper['title'][:60]} → {len(paper['chunks'])} chunks")
    
    # step 4: embed and store in chromadb
    print("\n--- Storing in ChromaDB ---")
    collection = store_in_chromadb(papers)
    
    # step 5: test a search query
    print("\n--- Testing semantic search ---")
    search("how does LoRA reduce trainable parameters?", collection)