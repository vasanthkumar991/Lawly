import json
import os
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# -------- CONFIG --------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-cases"
EMBEDDINGS_FILE = Path(r"D:\Intern\embeddings.json")
BATCH_SIZE = 100

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY and/or PINECONE_API_KEY environment variables.")

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(api_key=GEMINI_API_KEY)

# Create index if not present
existing_indexes = [i["name"] for i in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"⚙️ Creating new index {INDEX_NAME} (dim=1536)")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,   # matches text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# --- Load JSONL or JSON data ---
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

    # Generate OpenAI embedding
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

    vector_id = str(item.get("id", f"item_{i}"))
    batch.append((vector_id, emb, {"content": text}))

    if len(batch) == BATCH_SIZE:
        index.upsert(batch)
        count += len(batch)
        print(f"✅ Uploaded {count} vectors...")
        batch = []

if batch:
    index.upsert(batch)
    count += len(batch)
    print(f"✅ Uploaded final batch ({count} total)")

print("📊 Index Stats:", index.describe_index_stats())
