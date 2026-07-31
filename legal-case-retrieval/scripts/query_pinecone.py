import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Load API credentials & configs
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX", "legal-index-v1")

# Set up Pinecone and embeddings
pc = Pinecone(api_key=api_key, environment=env)
index = pc.Index(index_name)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Take question input from user
query = input("Type your legal question: ")
query_emb = model.encode(query).tolist()

# Search Pinecone index for top 5 relevant chunks
result = index.query(
    vector=query_emb,
    top_k=5,
    include_metadata=True
)

# Print results
print("\nTop 5 results for your query:\n")
for match in result['matches']:
    print("- Filename:", match['metadata']['filename'])
    print("  Chunk ID:", match['metadata']['chunk_id'])
    print("  Text preview:", match['metadata']['text'], "\n")
