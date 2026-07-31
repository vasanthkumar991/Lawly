from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
print(pc.list_indexes())
