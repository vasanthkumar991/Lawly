import os
from openai import OpenAI
from pinecone import Pinecone
import sys

# -------- CONFIG --------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-cases"
TOP_K = 5

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY and/or PINECONE_API_KEY environment variables.")

# -------- INITIALIZE --------
try:
    print("🔹 Initializing OpenAI and Pinecone connection...")
    client = OpenAI(api_key=GEMINI_API_KEY)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"❌ Index '{INDEX_NAME}' not found in Pinecone account.")
        sys.exit(1)

    index = pc.Index(INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {INDEX_NAME}")

except Exception as e:
    print(f"⚠️ Initialization failed: {e}")
    sys.exit(1)

# -------- FUNCTION --------
def interactive_search():
    print("\n=== ⚖️ Legal Statutory Text Retrieval Test ===")
    query_text = input("Enter your search query: ").strip()
    keywords_input = input("Enter keywords to verify relevance (comma-separated): ").strip()

    if not query_text:
        print("⚠️ Query cannot be empty.")
        return

    expected_keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]
    if not expected_keywords:
        print("⚠️ No keywords entered; skipping relevance check.")

    # Generate embedding using OpenAI (1536D)
    print("\n🧠 Generating embedding for query...")
    query_vector = client.embeddings.create(
        model="text-embedding-3-small",  # 1536D model
        input=query_text
    ).data[0].embedding

    # Query Pinecone
    print(f"🔍 Searching top {TOP_K} most relevant chunks...")
    results = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)

    if not results.matches:
        print("⚠️ No results found.")
        return

    print(f"\n=== Results for Query: '{query_text}' ===")
    found_relevant = False

    for rank, match in enumerate(results.matches, start=1):
        meta = match.get('metadata', {})
        snippet = meta.get("content", "").strip().lower()
        print(f"\n🔸 Rank {rank} | ID: {match.id} | Score: {match.score:.4f}")
        print("Snippet:", snippet[:300].replace("\n", " ") + "...")

        if expected_keywords and any(kw in snippet for kw in expected_keywords):
            found_relevant = True
            print("✅ Relevant statutory text FOUND here!")
    if expected_keywords:
        if found_relevant:
            print("\n🎯 Test PASSED: Relevant statutory text found in top results.")
        else:
            print("\n❌ Test FAILED: Relevant statutory text NOT found in top results.")

# -------- MAIN --------
if __name__ == "__main__":
    try:
        interactive_search()
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
    except Exception as e:
        print(f"⚠️ Error during search: {e}")
