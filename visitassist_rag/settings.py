import os

class Settings:
    ENV: str = os.getenv("VISITASSIST_ENV", "dev")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENV: str = os.getenv("PINECONE_ENV", "")
    DEFAULT_LANG: str = "pt"
    # Add more config as needed

settings = Settings()
