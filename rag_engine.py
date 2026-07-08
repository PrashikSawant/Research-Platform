"""
rag_engine.py — RAG pipeline as a reusable engine.
Extracts text from PDFs, chunks it, embeds it, and stores it in ChromaDB
tagged with a user_id so each user only ever searches their own documents.
"""

import uuid

import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# ---------- Setup (runs once, on import) ----------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")


# ---------- Step 1: Extract text from a PDF's raw bytes ----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


# ---------- Step 2: Chunk text into overlapping pieces ----------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ---------- Step 3: Embed + store chunks, tagged with user_id ----------
def add_document(user_id: str, filename: str, file_bytes: bytes) -> int:
    text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(text)

    if not chunks:
        return 0

    embeddings = embedding_model.encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"user_id": user_id, "filename": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


# ---------- Step 4: Search — scoped strictly to one user's documents ----------
def search_documents(user_id: str, query: str, n_results: int = 3) -> list[dict]:
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where={"user_id": user_id},  # <-- this is the per-user isolation guarantee
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        hits.append({"text": doc, "filename": meta.get("filename", "unknown")})
    return hits