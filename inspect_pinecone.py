import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ["PINECONE_INDEX"]

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Query all vectors (limit to 10 for inspection)
print(f"Inspecting up to 10 vectors from index: {INDEX_NAME}")

# Pinecone does not support listing all vectors directly, so we use fetch with known IDs or describe_index_stats
stats = index.describe_index_stats()
print("Index stats:", stats)

# If you know some IDs, you can fetch them directly. Otherwise, try to get some IDs from stats or Supabase.
# For demonstration, let's try to fetch by namespace (default)

# This will only work if you know some IDs. Otherwise, you can use the stats['namespaces'] to get counts.

# If you have a list of IDs, replace this:
ids = [
    "15e73450-dd13-4bb5-81b6-3da86459885f",
    "165b5067-a179-4803-8343-d815d24d2cdd",
    "19aa4c19-75b9-4e63-81d3-434a73986904",
    "1ac22a95-7246-4be3-891f-c76f60707487"
]

if ids:
    res = index.fetch(ids=ids)
    for k, v in res["vectors"].items():
        print(f"ID: {k}")
        print("Metadata:", v.get("metadata"))
        print()
else:
    print("No IDs provided. Use the Pinecone dashboard or Supabase to get some vector IDs.")
    print("Or, use describe_index_stats() output above to check namespaces and counts.")
