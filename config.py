import os
from dataclasses import dataclass

@dataclass
class Settings:
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1200))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # Vector store path
    index_dir: str = os.getenv("INDEX_DIR", "index_store")
    
    # LangSmith
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")


settings = Settings()
