import json
import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# -------- CONFIG --------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-cases"
EMBEDDINGS_FILE = Path(r"D:\Intern\embeddings.json")
BATCH_SIZE = 100

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY environment variable.")

# Initialize embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not present
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Connect to index
index = pc.Index(INDEX_NAME)

# --- Load data ---
with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    try:
        data_list = json.load(f)
    except json.JSONDecodeError:
        f.seek(0)
        data_list = [json.loads(line) for line in f if line.strip()]

# --- Upload in batches ---
batch = []
count = 0

for i, item in enumerate(data_list, start=1):
    text = item.get("content")
    if not text:
        continue

    vector = embeddings_model.embed_query(text)
    vector_id = str(item.get("id", f"item_{i}"))
    batch.append((vector_id, vector, {"content": text}))

    if len(batch) == BATCH_SIZE:
        index.upsert(batch)
        count += len(batch)
        print(f"✅ Uploaded {count} vectors...")
        batch = []

if batch:
    index.upsert(batch)
    count += len(batch)
    print(f"✅ Uploaded final batch ({count} total)")

stats = index.describe_index_stats()
print("📊 Index Stats:", stats)
