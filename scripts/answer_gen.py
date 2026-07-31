import os
from openai import OpenAI
from pinecone import Pinecone
import textwrap

# ===== CONFIG =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-cases"
TOP_K = 5

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY and/or PINECONE_API_KEY environment variables.")

# ===== INIT =====
print("🔹 Initializing clients...")
client = OpenAI(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Index '{INDEX_NAME}' not found in Pinecone account.")
index = pc.Index(INDEX_NAME)
print(f"✅ Connected to Pinecone index: {INDEX_NAME}")

# ===== RETRIEVE FUNCTION =====
def retrieve_context(query: str, top_k: int = TOP_K):
    emb = client.embeddings.create(
        model="text-embedding-3-small",  # 1536-D embeddings
        input=query
    ).data[0].embedding

    results = index.query(vector=emb, top_k=top_k, include_metadata=True)
    if not results.matches:
        return []

    contexts = []
    for m in results.matches:
        meta = m.get("metadata", {})
        snippet = meta.get("content", "")[:500].strip().replace("\n", " ")
        contexts.append({
            "id": m.id,
            "score": m.score,
            "text": snippet
        })
    return contexts

# ===== GENERATE ANSWER =====
def generate_answer(query: str, contexts: list):
    if not contexts:
        return "No relevant statutory text found. Please try another query."

    context_text = "\n\n".join(
        [f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)]
    )

    system_prompt = (
        "You are a concise Indian legal assistant. "
        "Use only the provided context to answer the question. "
        "Cite relevant section numbers or excerpts. "
        "If the answer is uncertain, list potentially relevant sections "
        "and advise consulting qualified legal counsel."
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer concisely with citations."

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
    )

    answer = response.choices[0].message.content.strip()
    return answer

# ===== MAIN INTERACTIVE LOOP =====
def main():
    print("\n=== ⚖️  Legal QA System ===")
    query = input("Enter your legal question: ").strip()
    if not query:
        print("⚠️ Empty query.")
        return

    print("\n🔍 Retrieving relevant statutory text...")
    contexts = retrieve_context(query)
    if not contexts:
        print("❌ No relevant results found.")
        return

    print("\n📘 Retrieved Contexts (Top Results):")
    for c in contexts:
        print(textwrap.fill(f"• {c['id']} (score {c['score']:.3f}): {c['text']}", width=100))
        print()

    print("💬 Generating concise legal answer...")
    answer = generate_answer(query, contexts)

    print("\n=== 🧾 Answer ===")
    print(textwrap.fill(answer, width=100))

# ===== RUN =====
if __name__ == "__main__":
    main()
