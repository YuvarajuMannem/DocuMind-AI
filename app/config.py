import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise Document Intelligence RAG Engine"
    API_V1_STR: str = "/api/v1"
    
    # Vector DB & Embeddings Config
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_TYPE: str = "FAISS"
    VECTOR_DB_DIR: str = "./vector_store"
    
    # Text Chunking Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 4
    
    # LLM Settings (Supports Mock/HuggingFace/OpenAI)
    LLM_PROVIDER: str = "huggingface"  # options: 'huggingface', 'openai', 'local'
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
