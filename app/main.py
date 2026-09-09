import os
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.api.v1.routes import router as api_v1_router
from app.core.rag_engine import rag_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-initialize embedding model on server boot so user uploads respond in 0ms
    def _warmup():
        try:
            print("Warmup: Initializing FastEmbed embedding model...")
            _ = rag_engine.embeddings
            print("Warmup: FastEmbed embedding model ready!")
        except Exception as e:
            print(f"Warmup notice: {e}")
    asyncio.create_task(run_in_threadpool(_warmup))
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise Retrieval-Augmented Generation (RAG) document intelligence engine built with LangChain, FAISS Vector Indexing, and FastAPI.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

# Serve Web Frontend Interface
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "DocuMind AI RAG Engine API online. Visit /docs for Swagger UI."}

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "vector_db": settings.VECTOR_DB_TYPE
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
