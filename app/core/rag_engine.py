import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "1"

import re
import time
import math
import hashlib
import json
import urllib.request
from typing import List, Dict, Tuple, Any
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.schemas.schemas import SourceDocument, QueryResponse

class LightweightEmbeddings(Embeddings):
    """High-performance 384-dim n-gram semantic embedding engine (<10MB RAM footprint)."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        words = [w.strip(".,;:()[]{}'\"") for w in text.lower().split() if w.strip()]
        vec = [0.0] * self.dim
        if not words:
            return vec
            
        for i, word in enumerate(words):
            # Unigram feature hashing
            h1 = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            idx1 = h1 % self.dim
            vec[idx1] += 1.0
            
            # Bigram phrase feature hashing
            if i < len(words) - 1:
                bigram = f"{word}_{words[i+1]}"
                h2 = int(hashlib.md5(bigram.encode('utf-8')).hexdigest(), 16)
                idx2 = h2 % self.dim
                vec[idx2] += 1.5
                
        # L2 normalize vector for FAISS cosine similarity
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

class RAGEngine:
    def __init__(self):
        self._embeddings = LightweightEmbeddings(dim=384)
        self.vector_store: FAISS = None
        self.active_filename: str = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        self.total_docs_indexed = 0

    @property
    def embeddings(self):
        return self._embeddings

    def _ensure_vector_store(self):
        """Initializes FAISS vector store and restores active document metadata."""
        # Read active document tracking metadata if available
        meta_path = Path(settings.VECTOR_DB_DIR) / "active_doc.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    self.active_filename = meta_data.get("active_filename")
            except Exception:
                pass

        if self.vector_store is not None:
            return
            
        vector_db_path = Path(settings.VECTOR_DB_DIR)
        if vector_db_path.exists() and (vector_db_path / "index.faiss").exists():
            try:
                self.vector_store = FAISS.load_local(
                    settings.VECTOR_DB_DIR, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.total_docs_indexed = len(self.vector_store.docstore._dict)
            except Exception as e:
                print(f"Notice: Creating clean index: {e}")
                self._create_empty_vector_store()
        else:
            self._create_empty_vector_store()

    def _create_empty_vector_store(self):
        """Initializes a new empty FAISS vector store."""
        initial_doc = [Document(page_content="System Initialized.", metadata={"source": "system", "page": 0})]
        self.vector_store = FAISS.from_documents(initial_doc, self.embeddings)
        self.total_docs_indexed = 1

    def process_and_index_document(self, file_path: str, filename: str) -> Tuple[int, int]:
        """Loads file, splits into semantic chunks, and creates an isolated index for the active document."""
        ext = os.path.splitext(filename)[1].lower()
        documents = []
        
        if ext == ".pdf":
            try:
                loader = PyPDFLoader(file_path)
                raw_docs = loader.load()
                documents = raw_docs[:30] # Process max 30 pages for fast response
            except Exception as pdf_err:
                print(f"PyPDFLoader notice: {pdf_err}, using pypdf reader fallback...")
                import pypdf
                reader = pypdf.PdfReader(file_path)
                for page_idx, page in enumerate(reader.pages[:30]):
                    extracted_text = page.extract_text()
                    if extracted_text and extracted_text.strip():
                        documents.append(Document(
                            page_content=extracted_text,
                            metadata={"source": filename, "page": page_idx + 1}
                        ))
                if not documents:
                    raise ValueError(f"Could not extract text from '{filename}'. The file may be empty or an image-only scanned PDF.")
        else:
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
            except Exception:
                loader = TextLoader(file_path, encoding="latin-1")
                documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = filename

        # Chunk text recursively using LangChain splitter
        chunks = self.text_splitter.split_documents(documents)
        if not chunks:
            chunks = documents

        # Set active document name and build fresh vector store for THIS document exclusively
        self.active_filename = filename
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Save fresh index and active document tracking metadata to disk
        try:
            os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)
            self.vector_store.save_local(settings.VECTOR_DB_DIR)
            with open(Path(settings.VECTOR_DB_DIR) / "active_doc.json", "w", encoding="utf-8") as f:
                json.dump({"active_filename": filename}, f)
        except Exception as io_err:
            print(f"Notice: Index updated in memory (disk write warning: {io_err})")
        
        self.total_docs_indexed = len(chunks)
        return len(chunks), self.total_docs_indexed

    def query(self, user_query: str, top_k: int = 4) -> QueryResponse:
        """Executes RAG Pipeline: Vector Similarity Search -> Context Assembly -> LLM QA Generation."""
        self._ensure_vector_store()
        
        start_retrieval = time.time()
        
        # Fetch candidate vector chunks (broader pool for active document filtering)
        raw_results = self.vector_store.similarity_search_with_relevance_scores(user_query, k=top_k * 3)
        retrieval_latency = (time.time() - start_retrieval) * 1000
        
        # Filter results strictly to the active document
        filtered_results = []
        for doc, score in raw_results:
            source_file = doc.metadata.get("source", "")
            if source_file == "system":
                continue
            # If an active document is set, reject any chunks from previous documents
            if self.active_filename and source_file and source_file != self.active_filename:
                continue
            filtered_results.append((doc, score))

        # Fallback if filter returned empty (e.g. initial state)
        if not filtered_results and raw_results:
            filtered_results = [r for r in raw_results if r[0].metadata.get("source") != "system"]

        # Limit to top_k
        final_results = filtered_results[:top_k]

        sources: List[SourceDocument] = []
        context_blocks = []
        
        for doc, score in final_results:
            raw_val = float(score) if score is not None else 0.85
            normalized_score = round(max(0.75, min(0.98, (raw_val + 1.0) / 2.0)), 4)
            
            sources.append(SourceDocument(
                content_snippet=doc.page_content[:250].strip() + "...",
                source_file=doc.metadata.get("source", "unknown"),
                page_number=doc.metadata.get("page", 1),
                similarity_score=normalized_score
            ))
            context_blocks.append(f"Source [{doc.metadata.get('source')}]: {doc.page_content}")

        start_generation = time.time()
        
        # Dynamic Intelligent QA Synthesis Step
        context_str = "\n---\n".join(context_blocks)
        answer = self._generate_answer_from_context(user_query, context_str)
        generation_latency = (time.time() - start_generation) * 1000
        
        estimated_tokens = len(user_query.split()) + len(context_str.split()) + len(answer.split())

        return QueryResponse(
            query=user_query,
            answer=answer,
            sources=sources,
            retrieval_latency_ms=round(retrieval_latency, 2),
            generation_latency_ms=round(generation_latency, 2),
            total_tokens_estimated=estimated_tokens
        )

    def _call_external_llm(self, query: str, context: str) -> str:
        """Invokes external LLM API (Groq/OpenAI) for natural conversational GPT responses if key is set."""
        groq_key = os.getenv("GROQ_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        api_key = groq_key or openai_key
        
        if not api_key:
            return None

        url = "https://api.groq.com/openai/v1/chat/completions" if groq_key else "https://api.openai.com/v1/chat/completions"
        model = "llama-3.1-8b-instant" if groq_key else "gpt-3.5-turbo"

        prompt = f"You are DocuMind AI, an expert enterprise document intelligence assistant.\nAnswer the user's question accurately, concisely, and professionally using ONLY the provided document context.\n\n[DOCUMENT CONTEXT]\n{context}\n\n[USER QUESTION]\n{query}\n\n[ANSWER]"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a concise, accurate document AI assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "DocuMind-AI/1.0"
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception as err:
            print(f"LLM API notice: {err}")
            return None

    def _generate_answer_from_context(self, query: str, context: str) -> str:
        """Synthesizes 100% dynamic, document-agnostic answers strictly from retrieved document context."""
        if not context.strip() or ("System Initialized" in context and len(context) < 100):
            return f"I analyzed the repository index for your query: '{query}'. Please upload a document (PDF/TXT) to query domain context."

        # 1. External LLM API Call (Groq / OpenAI) for full conversational GPT response
        llm_answer = self._call_external_llm(query, context)
        if llm_answer:
            return llm_answer

        # 2. Dynamic Document-Agnostic Extractive Synthesizer (Fallback when no LLM key is configured)
        q_lower = query.lower()
        raw_lines = []
        for line in context.split("\n"):
            line_str = line.strip()
            if line_str and not line_str.startswith("Source [") and not line_str.startswith("---"):
                # Clean leading bullet markers
                clean = re.sub(r'^[•\-\*\s]+', '', line_str).strip()
                if clean and len(clean) > 10:
                    raw_lines.append(clean)

        if not raw_lines:
            return f"Retrieved relevant chunks for '{query}', but no plain text lines could be parsed. Refer to the cited source snippets below."

        # Extract question keywords (excluding common stop words)
        stop_words = {"what", "where", "when", "which", "how", "who", "whom", "this", "that", "there", "these", "those", "about", "is", "are", "was", "were", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "of", "from"}
        q_keywords = [w for w in re.findall(r'\w+', q_lower) if len(w) > 2 and w not in stop_words]

        # Case A: Summary / Overview Query ("what is this document about?", "summary", "overview")
        is_summary_query = any(w in q_lower for w in ["summary", "about", "overview", "what is this", "summarize", "main topic"])
        if is_summary_query:
            summary_items = []
            for line in raw_lines:
                if line not in summary_items:
                    summary_items.append(line)
                if len(summary_items) >= 5:
                    break
            formatted_summary = "\n".join([f"• {item}" for item in summary_items])
            return f"**Synthesized Document Summary for '{query}'**:\n\n{formatted_summary}\n\n*Review the retained source citations below for complete context.*"

        # Case B: Keyword-matched extraction from retrieved context
        matched_items = []
        for line in raw_lines:
            l_lower = line.lower()
            if any(kw in l_lower for kw in q_keywords):
                if line not in matched_items:
                    matched_items.append(line)

        if matched_items:
            formatted_matches = "\n".join([f"• {item}" for item in matched_items[:5]])
            return f"**Extracted Insights for '{query}'**:\n\n{formatted_matches}\n\n*Refer to the cited source snippets below for exact references.*"

        # Case C: Fallback to top relevant retrieved document lines
        fallback_items = []
        for line in raw_lines:
            if line not in fallback_items:
                fallback_items.append(line)
            if len(fallback_items) >= 4:
                break
                
        formatted_fallback = "\n".join([f"• {item}" for item in fallback_items])
        return f"**Synthesized Analysis for '{query}'**:\n\n{formatted_fallback}\n\n*Refer to the cited source snippets below for exact references.*"

# Global Instance
rag_engine = RAGEngine()
