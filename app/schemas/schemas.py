from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SourceDocument(BaseModel):
    content_snippet: str = Field(..., description="Extracted relevant text chunk snippet")
    source_file: str = Field(..., description="Name of the source document file")
    page_number: Optional[int] = Field(None, description="Page number in source document")
    similarity_score: Optional[float] = Field(None, description="Vector search similarity score")

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the key responsibilities of the data engineer?", min_length=3)
    top_k: Optional[int] = Field(4, ge=1, le=10, description="Number of context chunks to retrieve")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceDocument]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_tokens_estimated: int

class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_created: int
    total_indexed_documents: int
    status: str
    message: str

class VectorStoreStats(BaseModel):
    total_vector_indexes: int
    embedding_model: str
    vector_db_backend: str
    status: str
    chunk_size: int
    chunk_overlap: int
