from dotenv import load_dotenv
load_dotenv()
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = os.environ["PINECONE_INDEX"]
index = pc.Index(INDEX_NAME)

# Print total vector count and sample metadata

import random
import json
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")

# Print only serializable parts of stats
print("\nSerializable parts of index stats:")
if isinstance(stats, dict):
    for k, v in stats.items():
        try:
            json.dumps(v)
            print(f"{k}: {v}")
        except TypeError:
            print(f"{k}: (not JSON serializable, type={type(v)}) -> {repr(v)}")
else:
    print(repr(stats))

# Fetch and print metadata for up to 3 sample vectors (using a zero and a random vector)
if stats['total_vector_count'] > 0:
    D = 3072  # embedding dimension for text-embedding-3-large
    print("\nQuerying with zero vector...")
    sample_zero = index.query(vector=[0.0]*D, top_k=3, include_metadata=True)
    matches_zero = sample_zero.get('matches', []) if isinstance(sample_zero, dict) else sample_zero.matches
    if not matches_zero:
        print("No sample vectors returned by zero vector query.")
    else:
        for match in matches_zero:
            print(f"Sample ID: {match['id']}")
            print(f"Metadata: {match['metadata']}")

    print("\nQuerying with random vector...")
    rand_vec = [random.uniform(-1, 1) for _ in range(D)]
    sample_rand = index.query(vector=rand_vec, top_k=3, include_metadata=True)
    matches_rand = sample_rand.get('matches', []) if isinstance(sample_rand, dict) else sample_rand.matches
    if not matches_rand:
        print("No sample vectors returned by random vector query.")
    else:
        for match in matches_rand:
            print(f"Sample ID: {match['id']}")
            print(f"Metadata: {match['metadata']}")
else:
    print("No vectors found in Pinecone index.")
