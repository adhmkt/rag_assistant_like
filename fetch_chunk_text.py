from dotenv import load_dotenv
load_dotenv()
import os
from pinecone import Pinecone
import sys

if len(sys.argv) < 2:
    print("Usage: python fetch_chunk_text.py <vector_id>")
    sys.exit(1)

vector_id = sys.argv[1]

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = os.environ["PINECONE_INDEX"]
index = pc.Index(INDEX_NAME)

# Fetch the vector by ID
result = index.fetch(ids=[vector_id])

vectors = result.get('vectors', {}) if isinstance(result, dict) else getattr(result, 'vectors', {})
if vector_id not in vectors:
    print(f"Vector ID {vector_id} not found.")
    sys.exit(1)

metadata = vectors[vector_id].get('metadata', {}) if isinstance(vectors[vector_id], dict) else getattr(vectors[vector_id], 'metadata', {})
chunk_text = metadata.get('chunk_text', None)

print(f"Vector ID: {vector_id}")
print(f"chunk_text: {chunk_text if chunk_text else '[No chunk_text in metadata]'}")
print(f"Full metadata: {metadata}")
