from pinecone import Pinecone, ServerlessSpec
import os
from openai import OpenAI

# --- API KEYS ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY and/or PINECONE_API_KEY environment variables.")

# --- INITIALIZE CLIENTS ---
client = OpenAI(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "legal-cases"

# Check if index exists, else create
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"⚙️ Creating index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,  # text-embedding-3-large output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("⏳ Index created. Please wait a minute before querying.")
index = pc.Index(INDEX_NAME)


def embed_text(text: str):
    """Generate embeddings using new OpenAI SDK."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def retrieve(query: str, top_k: int = 5):
    """Retrieve top matching chunks from Pinecone index."""
    emb = embed_text(query)
    res = index.query(vector=emb, top_k=top_k, include_metadata=True)

    print(f"\n🔍 Query: {query}\n")
    for match in res["matches"]:
        meta = match.get("metadata", {})
        title = meta.get("title", "UnknownDoc")
        heading = meta.get("heading", "")
        print(f"{match['id']} ({match['score']:.3f}) — {title}: {heading}")

    return res


if __name__ == "__main__":
    retrieve("What is the significance of Article 19A?")
