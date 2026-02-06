import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX = os.environ["PINECONE_INDEX"]
API_KEY = os.environ["PINECONE_API_KEY"]
CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
REGION = os.environ.get("PINECONE_REGION", "us-east-1")

# Use embedding dimension matching your embedding model.
# text-embedding-3-large = 3072, text-embedding-3-small = 1536
EMBED_DIM = 3072

pc = Pinecone(api_key=API_KEY)

existing = [i["name"] for i in pc.list_indexes()]
if INDEX not in existing:
    pc.create_index(
        name=INDEX,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    print("Created index:", INDEX)
else:
    print("Index already exists:", INDEX)
