import arxiv
import os
import re
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

ML_CATEGORIES = {"cs.LG", "cs.CL", "cs.AI", "stat.ML"}

# sections we care about — in order of importance
IMPORTANT_SECTIONS = [
    "abstract", "introduction", "related work", "background",
    "method", "methods", "methodology", "approach", "model",
    "experiment", "experiments", "evaluation", "results",
    "discussion", "conclusion", "conclusions"
]

# sections to skip entirely — pure noise for RAG
SKIP_SECTIONS = [
    "references", "bibliography", "acknowledgement",
    "acknowledgements", "appendix", "funding"
]


def detect_sections(text):
    """
    Split paper text into named sections.
    Returns a list of (section_name, section_text) tuples.
    """
    lines = text.split("\n")
    sections = []
    current_section = "abstract"
    current_text = []

    for line in lines:
        stripped = line.strip()

        # a section header is typically:
        # short (under 60 chars), not ending with period,
        # and matches a known section name
        if (
            len(stripped) > 2
            and len(stripped) < 60
            and not stripped.endswith(".")
            and not stripped.endswith(",")
        ):
            lower = stripped.lower()
            # remove numbering like "1.", "2.1", "I." from start
            clean = re.sub(r"^[\d]+[\.\d]*\s*", "", lower).strip()
            clean = re.sub(r"^[ivxlcdm]+\.\s*", "", clean).strip()

            # check if it matches a known section
            matched = None
            for sec in IMPORTANT_SECTIONS + SKIP_SECTIONS:
                if clean == sec or clean.startswith(sec):
                    matched = sec
                    break

            if matched:
                # save previous section
                if current_text:
                    sections.append((current_section, "\n".join(current_text)))
                current_section = matched
                current_text = []
                continue

        current_text.append(line)

    # save last section
    if current_text:
        sections.append((current_section, "\n".join(current_text)))

    return sections


def chunk_section(text, section_name, chunk_size=400, overlap=40):
    """Chunk a single section's text into smaller pieces."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 80:  # skip tiny chunks
            chunks.append({
                "text": chunk,
                "section": section_name
            })
        start = end - overlap
    return chunks


def extract_and_chunk(filepath):
    """Extract text from PDF and split into section-aware chunks."""
    try:
        doc = fitz.open(filepath)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
    except Exception as e:
        print(f"  Failed to extract: {e}")
        return []

    # detect sections
    sections = detect_sections(full_text)

    # chunk each section separately, skip noise sections
    all_chunks = []
    for section_name, section_text in sections:
        if section_name in SKIP_SECTIONS:
            continue
        chunks = chunk_section(section_text, section_name)
        all_chunks.extend(chunks)

    return all_chunks


def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    # new collection so old naive chunks don't mix with smart ones
    collection = client.get_or_create_collection(
        name="arxiv_smart",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def get_stored_ids(collection):
    results = collection.get()
    stored = set()
    for id in results["ids"]:
        paper_id = id.rsplit("_chunk_", 1)[0]
        stored.add(paper_id)
    return stored


def store_paper(paper, chunks, collection):
    if not chunks:
        return 0

    chunk_ids = [
        f"{paper['id'].split('/')[-1]}_chunk_{i}"
        for i in range(len(chunks))
    ]
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts).tolist()
    metadatas = [
        {
            "title": paper["title"],
            "authors": ", ".join(paper["authors"][:3]),
            "published": paper["published"],
            "paper_id": paper["id"].split("/")[-1],
            "section": c["section"],
            "chunk_index": i
        }
        for i, c in enumerate(chunks)
    ]
    collection.upsert(
        ids=chunk_ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    return len(chunks)


def main():
    collection = get_collection()
    stored_ids = get_stored_ids(collection)
    print(f"Smart collection: {len(stored_ids)} papers, {collection.count()} chunks\n")

    # re-process all PDFs already in the data/ folder
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs to process\n")

    total_papers = 0
    total_chunks = 0

    for pdf_file in pdf_files:
        paper_id = pdf_file.replace(".pdf", "")

        if paper_id in stored_ids:
            continue

        filepath = os.path.join("data", pdf_file)
        chunks = extract_and_chunk(filepath)

        if not chunks:
            continue

        # build minimal paper dict from filename
        paper = {
            "id": f"https://arxiv.org/abs/{paper_id}",
            "title": paper_id,  # we'll fix this below
            "authors": ["Unknown"],
            "published": "Unknown",
        }

        # try to get proper metadata from arxiv
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[paper_id.split("v")[0]])
            for result in client.results(search):
                paper["title"] = result.title
                paper["authors"] = [a.name for a in result.authors]
                paper["published"] = str(result.published)
                break
            time.sleep(1)
        except Exception:
            pass

        count = store_paper(paper, chunks, collection)
        stored_ids.add(paper_id)
        total_papers += 1
        total_chunks += count

        # show section breakdown for first 3 papers
        if total_papers <= 3:
            section_counts = {}
            for c in chunks:
                section_counts[c["section"]] = section_counts.get(c["section"], 0) + 1
            print(f"[{total_papers}] {paper['title'][:55]}...")
            for sec, count_s in section_counts.items():
                print(f"     {sec:<20} {count_s} chunks")
        else:
            print(f"[{total_papers}] {paper['title'][:65]}... → {count} chunks")

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Papers processed : {total_papers}")
    print(f"  Total chunks     : {total_chunks}")
    print(f"  Total in database: {collection.count()} chunks")


if __name__ == "__main__":
    main()