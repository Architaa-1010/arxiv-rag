import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

print("Creating a deployment-ready subset of the knowledge base...")

# connect to your full local database
source_client = chromadb.PersistentClient(path="./chroma_db")
source_collection = source_client.get_or_create_collection(
    name="arxiv_smart",
    metadata={"hnsw:space": "cosine"}
)

print(f"Full database: {source_collection.count()} chunks")

# create a smaller deployment database
deploy_path = "./chroma_deploy"
if os.path.exists(deploy_path):
    shutil.rmtree(deploy_path)

deploy_client = chromadb.PersistentClient(path=deploy_path)
deploy_collection = deploy_client.get_or_create_collection(
    name="arxiv_smart",
    metadata={"hnsw:space": "cosine"}
)

# copy all chunks — ChromaDB files are already compact
print("Copying chunks to deployment database...")
batch_size = 500
offset = 0
total_copied = 0

while True:
    results = source_collection.get(
        limit=batch_size,
        offset=offset,
        include=["documents", "metadatas", "embeddings"]
    )
    
    if not results["ids"]:
        break
    
    deploy_collection.upsert(
        ids=results["ids"],
        documents=results["documents"],
        embeddings=results["embeddings"],
        metadatas=results["metadatas"]
    )
    
    total_copied += len(results["ids"])
    offset += batch_size
    print(f"  Copied {total_copied} chunks...")
    
    if len(results["ids"]) < batch_size:
        break

print(f"\nDeployment database ready: {deploy_collection.count()} chunks")

# zip it up
print("Compressing...")
shutil.make_archive("chroma_deploy", "zip", ".", "chroma_deploy")
print(f"Created chroma_deploy.zip ({os.path.getsize('chroma_deploy.zip') / 1024 / 1024:.1f} MB)")
print("\nReady to upload to HuggingFace!")