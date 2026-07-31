import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENV", "us-east1-gcp")          # adjust if needed
index_name = os.getenv("PINECONE_INDEX", "legal-index-v1")

CHUNKS_DIR = os.path.join("data", "chunks")
DIM = 384  # typical for all-MiniLM-L6-v2

# Set your Pinecone region and cloud as appropriate
CLOUD = "aws"          # Or as specified by mentor/dashboard
REGION = "us-east-1"   # Or as specified by mentor/dashboard

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# List indexes and create if missing
existing_indexes = [idx.name for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION)
    )
    print(f"Created new Pinecone index '{index_name}'.")

index = pc.Index(index_name)

# Model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

for fname in os.listdir(CHUNKS_DIR):
    if fname.endswith('.json'):
        with open(os.path.join(CHUNKS_DIR, fname), 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        vectors = []
        for i, chunk in enumerate(chunks):
            emb = model.encode(chunk).tolist()
            meta = {
                "filename": fname.replace('_chunks.json', ''), 
                "chunk_id": i, 
                "text": chunk[:200]  # Preview only, not full text
            }
            vectors.append({
                "id": f"{fname}_{i}",
                "values": emb,
                "metadata": meta
            })
        # Upsert batch of vectors
        index.upsert(vectors)
        print(f"Indexed {len(vectors)} chunks from {fname}.")

print("All text chunks embedded and indexed in Pinecone.")
