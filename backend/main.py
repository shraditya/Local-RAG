import gc
import os
import re
import uuid
import shutil
import tempfile
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from config import CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_K, MAX_K, K_PER_FILE
from core.embeddings import load_embeddings
from core.llamacpp import LlamaCppServerLLM
from core.reranker import load_reranker
from core.retriever import HybridRetriever
from utils.pdf_utils import load_pdf_as_markdown
from utils.text_utils import tokenize, parse_cot_response, save_conversation

# Re-use prompt templates and helpers from query_engine
from core.query_engine import (
    get_prompt_for_model,
    _format_docs,
    strip_thinking,
)

app = FastAPI(title="RAG API", description="LangChain + llama.cpp RAG backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
PERSIST_DIR_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db_api")
CONVERSATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR_BASE, exist_ok=True)
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.persist_dir = os.path.join(PERSIST_DIR_BASE, session_id)
        self.bm25 = None
        self.all_splits: list = []
        self.indexed_files: list[str] = []
        self.vectorstore = None
        self.query_engine = None
        self.messages: list[dict] = []
        self.selected_model: str = "default"
        self.use_reranker: bool = True
        self.last_sources: list = []
        self.conversation_file: Optional[str] = None

    def get_chat_history(self, window: int = 6) -> str:
        history = []
        for msg in self.messages[-window:]:
            if msg["role"] == "assistant":
                _, answer = parse_cot_response(msg["content"])
                history.append(f"Assistant: {answer}")
            else:
                history.append(f"User: {msg['content']}")
        return "\n".join(history)

    def reset_chat(self):
        self.messages = []
        self.conversation_file = None

    def release_vectorstore(self):
        """Release ChromaDB file handles (important on Windows)."""
        if self.vectorstore is not None:
            try:
                client = self.vectorstore._client
                if hasattr(client, "close"):
                    client.close()
            except Exception:
                pass
            self.vectorstore = None
        self.bm25 = None
        self.all_splits = []
        self.query_engine = None
        gc.collect()


# In-memory session store: session_id -> SessionState
_sessions: dict[str, SessionState] = {}


def get_session(session_id: str) -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState(session_id)
    return _sessions[session_id]

class ChatRequest(BaseModel):
    session_id: str
    prompt: str
    model: str = "default"
    use_reranker: bool = True


class ClearRequest(BaseModel):
    session_id: str


class DeleteDocRequest(BaseModel):
    session_id: str
    filename: str


def _build_index(session: SessionState) -> dict:
    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return {"message": "No PDF files found.", "engine_initialized": False}

    embed_model = load_embeddings()
    if embed_model is None:
        return {"message": "Embedding model not found.", "engine_initialized": False}

    all_docs = []
    for filename in pdf_files:
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            docs = load_pdf_as_markdown(file_path, filename)
            all_docs.extend(docs)
            print(f"Loaded {filename} ({len(docs)} pages)")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    if not all_docs:
        return {"message": "No content extracted from PDFs.", "engine_initialized": False}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_documents(all_docs)
    for doc in splits:
        doc.metadata.setdefault("type", "text")

    # Clear old index and rotate to fresh persist dir
    session.release_vectorstore()
    session.persist_dir = os.path.join(PERSIST_DIR_BASE, session.session_id)
    if os.path.exists(session.persist_dir):
        shutil.rmtree(session.persist_dir, ignore_errors=True)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embed_model,
        persist_directory=session.persist_dir,
    )
    session.vectorstore = vectorstore
    session.indexed_files = pdf_files
    session.all_splits = splits

    if BM25Okapi is not None:
        tokenized_corpus = [tokenize(doc.page_content) for doc in splits]
        session.bm25 = BM25Okapi(tokenized_corpus)

    # Build query engine
    _init_query_engine(session)

    return {
        "message": f"Indexed {len(splits)} chunks from {len(pdf_files)} files.",
        "engine_initialized": session.query_engine is not None,
        "files": pdf_files,
    }


def _init_query_engine(session: SessionState):
    """Build the LCEL chain for this session and store it."""
    try:
        num_files = len(session.indexed_files)
        k_value = max(DEFAULT_K, min(K_PER_FILE * max(num_files, 1), MAX_K))

        vector_retriever = session.vectorstore.as_retriever(
            search_kwargs={"k": k_value}
        )

        reranker = load_reranker() if session.use_reranker else None

        if session.bm25 is not None and session.all_splits:
            retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25=session.bm25,
                all_splits=session.all_splits,
                k=k_value,
                reranker=reranker,
            )
        else:
            retriever = vector_retriever

        def retrieve_and_track(query: str) -> str:
            """Retrieve docs, track sources, format context."""
            docs = (
                retriever.invoke(query)
                if hasattr(retriever, "invoke")
                else retriever.get_relevant_documents(query)
            )
            seen = set()
            sources = []
            for doc in docs:
                key = (doc.metadata.get("source_file"), doc.metadata.get("page"))
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "file": doc.metadata.get("source_file", "unknown"),
                        "page": doc.metadata.get("page"),
                        "type": doc.metadata.get("type", "text"),
                    })
            session.last_sources = sources
            return _format_docs(docs)

        llm = LlamaCppServerLLM()
        prompt_template = get_prompt_for_model(session.selected_model)

        session.query_engine = (
            {
                "context": retrieve_and_track,
                "input": RunnablePassthrough(),
                "chat_history": lambda _: session.get_chat_history(),
                "num_docs": lambda _: len(session.indexed_files),
                "doc_names": lambda _: ", ".join(session.indexed_files) or "No documents",
            }
            | prompt_template
            | llm
            | StrOutputParser()
            | strip_thinking
        )

    except Exception as e:
        print(f"Failed to init query engine: {e}")
        session.query_engine = None


def _save_conversation(session: SessionState):
    """Save the session's conversation to a JSON file."""
    import json
    from datetime import datetime

    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

    if session.conversation_file is None:
        timestamp = session.session_id[:8]
        session.conversation_file = os.path.join(
            CONVERSATIONS_DIR, f"conversation_{timestamp}.json"
        )

    payload = {
        "saved_at": datetime.now().isoformat(),
        "session_id": session.session_id,
        "model": session.selected_model,
        "documents": session.indexed_files,
        "messages": [
            {
                "role": m["role"],
                "content": parse_cot_response(m["content"])[1]
                if m["role"] == "assistant"
                else m["content"],
            }
            for m in session.messages
        ],
    }

    with open(session.conversation_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

@app.get("/api/session/new")
def new_session():
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = SessionState(session_id)
    return {"session_id": session_id}


@app.post("/api/documents/upload")
async def upload_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Upload one or more PDFs and rebuild the index for this session."""
    session = get_session(session_id)

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")
        dest = os.path.join(UPLOAD_DIR, file.filename)
        with open(dest, "wb") as f:
            f.write(await file.read())

    result = _build_index(session)
    return result


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str, session_id: str):
    """Delete a PDF and rebuild the index."""
    session = get_session(session_id)
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"{filename} not found.")

    try:
        os.remove(file_path)
    except PermissionError:
        # Windows file lock fallback
        os.rename(file_path, file_path + ".deleted")

    result = _build_index(session)
    return result


@app.get("/api/documents")
def list_documents():
    """List all uploaded PDFs."""
    files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    return {"files": files}


@app.post("/api/chat")
def chat(request: ChatRequest):
    """Send a message and get a response from the RAG chain."""
    session = get_session(request.session_id)
    session.selected_model = request.model
    session.use_reranker = request.use_reranker
    session.last_sources = []

    # Rebuild chain if model changed or not yet built
    if session.query_engine is None or session.selected_model != request.model:
        if session.vectorstore is not None:
            _init_query_engine(session)

    session.messages.append({"role": "user", "content": request.prompt})

    if session.query_engine is None:
        answer = "Please upload and process documents first."
    else:
        try:
            answer = session.query_engine.invoke(request.prompt)
        except Exception as e:
            answer = f"Error: {e}"

    session.messages.append({"role": "assistant", "content": answer})

    # Auto-save conversation
    try:
        _save_conversation(session)
    except Exception as e:
        print(f"Could not save conversation: {e}")

    return {
        "response": answer,
        "sources": session.last_sources,
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in session.messages
        ],
    }


@app.post("/api/chat/clear")
def clear_chat(request: ClearRequest):
    """Clear chat history for a session."""
    session = get_session(request.session_id)
    session.reset_chat()
    return {"message": "Chat cleared."}


@app.get("/api/chat/history")
def get_history(session_id: str):
    """Get full chat history for a session."""
    session = get_session(session_id)
    return {
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in session.messages
        ]
    }


@app.get("/api/status")
def status(session_id: str):
    """Get pipeline status for a session."""
    session = get_session(session_id)
    return {
        "session_id": session_id,
        "indexed_files": session.indexed_files,
        "engine_ready": session.query_engine is not None,
        "bm25_ready": session.bm25 is not None,
        "use_reranker": session.use_reranker,
        "model": session.selected_model,
    }


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    """Clean up a session and free resources."""
    if session_id in _sessions:
        _sessions[session_id].release_vectorstore()
        del _sessions[session_id]
    return {"message": f"Session {session_id} deleted."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)