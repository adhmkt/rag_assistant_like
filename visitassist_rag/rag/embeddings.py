import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

EMBED_MODEL = "text-embedding-3-large"  # dim=3072
oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = oai.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]
