import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, status
from fastapi.responses import StreamingResponse

from app.schemas.schemas import (
    QueryRequest, 
    QueryResponse, 
    DocumentUploadResponse, 
    VectorStoreStats
)
from app.core.rag_engine import rag_engine
from app.config import settings

router = APIRouter()

@router.post(
    "/documents/upload", 
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Index Document",
    description="Uploads a PDF or TXT document, splits it into semantic chunks, and creates vector embeddings in FAISS."
)
async def upload_document(file: UploadFile = File(...)):
    allowed_extensions = [".pdf", ".txt", ".md"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {allowed_extensions}"
        )
        
    try:
        # Save temporary file for LangChain loader processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
            
        chunks_created, total_indexed = rag_engine.process_and_index_document(tmp_path, file.filename)
        os.remove(tmp_path)
        
        return DocumentUploadResponse(
            filename=file.filename,
            chunks_created=chunks_created,
            total_indexed_documents=total_indexed,
            status="SUCCESS",
            message=f"Successfully processed '{file.filename}' into {chunks_created} vector chunks."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document '{file.filename}': {str(e)}"
        )

@router.post(
    "/rag/query", 
    response_model=QueryResponse,
    summary="Execute RAG Vector Search & Synthesis QA",
    description="Performs dense vector similarity search over indexed store and generates context-aware LLM response."
)
async def query_rag(request: QueryRequest):
    try:
        response = rag_engine.query(request.query, top_k=request.top_k or settings.TOP_K_RESULTS)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG pipeline: {str(e)}"
        )

@router.get(
    "/vectorstore/stats",
    response_model=VectorStoreStats,
    summary="Get Vector Database Metrics",
    description="Returns telemetry regarding vector store document counts, embedding dimensions, and chunk settings."
)
async def get_vectorstore_stats():
    return VectorStoreStats(
        total_vector_indexes=rag_engine.total_docs_indexed,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        vector_db_backend=settings.VECTOR_DB_TYPE,
        status="ONLINE",
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
