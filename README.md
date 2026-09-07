# ⚡ Enterprise RAG Knowledge Engine (FastAPI + LangChain + FAISS)

A high-performance, asynchronous Retrieval-Augmented Generation (RAG) backend engine designed for domain-specific document intelligence. Built using **FastAPI**, **LangChain**, **Sentence-Transformers Embeddings**, **FAISS Vector Indexing**, and **Pydantic v2**.

---

## 🌟 Key Features

- 📄 **Multi-Format Document Ingestion**: Ingests PDF, TXT, and Markdown files asynchronously.
- 🧩 **Semantic Text Chunking**: Leverages LangChain's `RecursiveCharacterTextSplitter` with customizable chunk overlap.
- 🔍 **Dense Vector Search**: Powered by `sentence-transformers/all-MiniLM-L6-v2` dense embeddings and `FAISS` vector similarity store.
- ⚡ **Asynchronous REST API**: High-throughput FastAPI endpoints with complete Pydantic data validation.
- 📍 **Source Attribution**: Retains source filename, page number, and similarity score citations for every retrieved answer.
- 🐳 **Production Ready**: Fully containerized with Docker & Docker-Compose.

---

## 🛠️ Architecture Workflow

```
[User Document Upload] ──> [LangChain Document Loader] ──> [Recursive Text Splitter]
                                                                  │
                                                                  ▼
                                                      [Dense Vector Embeddings]
                                                                  │
                                                                  ▼
                                                       [FAISS Vector Store]
                                                                  │
[User Natural Query]   ──> [Vector Search (Top-K)] ───> [Contextual QA Generation]
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/YuvarajuMannem/Enterprise-RAG-FastAPI.git
cd Enterprise-RAG-FastAPI

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/documents/upload` | Upload PDF/TXT and generate FAISS vector index |
| `POST` | `/api/v1/rag/query` | Submit natural language query & retrieve answer + citations |
| `GET`  | `/api/v1/vectorstore/stats` | Retrieve total vectors indexed and store telemetry |
| `GET`  | `/health` | Service health check status |

---

## 🐳 Docker Deployment

```bash
docker build -t enterprise-rag-fastapi .
docker run -p 8000:8000 enterprise-rag-fastapi
```

---

## 👨‍💻 Developer & Author

**Yuvaraju Mannem**  
*B.Tech Computer Science & Engineering (Specialization in AI & Machine Learning)*  
*VIT-AP University* | Oracle Certified Generative AI Professional  
- GitHub: [@YuvarajuMannem](https://github.com/YuvarajuMannem)
- Portfolio: [yuvaraju-portfolio.vercel.app](https://yuvaraju-portfolio.vercel.app/)
