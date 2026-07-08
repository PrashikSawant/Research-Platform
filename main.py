"""
main.py — FastAPI backend for the Full Research Platform.
Layers: Auth (JWT) -> Agent (tool-calling) -> RAG (per-user document search)
"""

from dotenv import load_dotenv

load_dotenv()  # must run BEFORE importing agent/auth, since they read env vars at import time

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

import agent
import auth
import rag_engine

app = FastAPI(title="AI Research Platform")

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegisterRequest(BaseModel):
    username: str
    password: str


class AskRequest(BaseModel):
    question: str


# ---------- 1. Register ----------
@app.post("/register")
def register(payload: RegisterRequest):
    created = auth.create_user(payload.username, payload.password)
    if not created:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"message": f"User '{payload.username}' created successfully"}


# ---------- 2. Login (returns JWT token) ----------
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = auth.create_access_token(username=form_data.username)
    return {"access_token": token, "token_type": "bearer"}


# ---------- 3. Upload a document (protected — requires valid JWT) ----------
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: str = Depends(auth.get_current_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    num_chunks = rag_engine.add_document(
        user_id=current_user, filename=file.filename, file_bytes=file_bytes
    )

    if num_chunks == 0:
        raise HTTPException(status_code=400, detail="Could not extract any text from this PDF")

    return {
        "message": f"'{file.filename}' processed and stored",
        "chunks_stored": num_chunks,
        "owner": current_user,
    }


# ---------- 4. Ask a question (protected — agent decides if RAG is needed) ----------
@app.post("/ask")
def ask_question(
    payload: AskRequest,
    current_user: str = Depends(auth.get_current_user),
):
    result = agent.run_agent(user_id=current_user, question=payload.question)
    return result


# ---------- Health check ----------
@app.get("/")
def root():
    return {"status": "AI Research Platform backend is running"}