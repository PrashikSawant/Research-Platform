# 🔬 Day 28 — Full Research Platform (Backend): RAG + Agents + Auth

Part of my **#100DaysOfAIEngineering** challenge — 100 projects, 100 days, building toward becoming job-ready as an AI Engineer.

> **Note:** This project was split across two days since it genuinely combines three previously multi-day topics. This repo covers **Day 28 — the FastAPI backend** (auth + RAG-as-a-tool + agent orchestration). Day 28b adds the Streamlit frontend on top of this same backend.

## 💡 Why I Built This

Every project so far has built RAG, agents, or auth *in isolation*. Real AI products combine all three — an authenticated user asks a question, an agent decides how to answer it, and if it needs your private documents, it searches them safely without ever touching another user's data. This project is my first proof that I can architect multiple systems together instead of building them as separate demos.

## 🧠 Thought Process

1. **Auth layer first** — JWT tokens gate every protected endpoint, reused from Day 18's pattern
2. **RAG becomes a tool, not the whole app** — instead of RAG being the entire pipeline (like Day 12-16), it's now one function the agent can choose to call
3. **Agent decides** — given a question, the LLM decides whether to search documents or answer directly, using Groq's native tool/function calling
4. **Per-user isolation** — every document is tagged with a `user_id` in ChromaDB, and every search is filtered by it, so User A can never accidentally search User B's documents
5. **Resilience** — Groq's tool calling can intermittently fail on malformed output (a known issue); the agent retries once, then falls back to a plain answer instead of crashing

## ⚙️ What This Does

- 🔐 Register + login with JWT-based authentication
- 📄 Upload PDF documents — chunked, embedded, and stored per-user in ChromaDB
- 🤖 Ask questions — an LLM agent decides whether to search your documents or answer directly
- 🔒 Strict per-user document isolation (verified with automated tests)
- 🛡️ Automatic retry + graceful fallback if the AI provider's tool calling has a hiccup

## 🛠️ Tech Stack

- **Python + FastAPI** — REST API backend
- **python-jose + bcrypt** — JWT auth and password hashing
- **Groq API** — LLaMA 3.3 70B for both agent reasoning and tool calling
- **ChromaDB** — vector storage with per-user metadata filtering
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — embeddings
- **PyMuPDF** — PDF text extraction

## 🚀 How to Run

```powershell
git clone https://github.com/PrashikSawant/day28-research-platform.git
cd day28-research-platform
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
JWT_SECRET_KEY=generate_your_own_random_string_here
```

Run it:
```powershell
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API tester.

## 📚 What I Learned

- How to treat RAG as **one tool in an agent's toolbox**, not the entire application — a real architectural shift from earlier RAG-only projects
- Why per-user data isolation has to be deliberately built into the data layer (ChromaDB metadata filters), not just the login screen
- That splitting code across multiple files changes *when* environment variables need to load — I hit and fixed a real bug where `.env` wasn't loaded before a module needed it at import time
- That AI providers themselves can be unreliable in production — Groq's tool calling has a known intermittent failure mode, and handling that gracefully (retry + fallback) is a real production engineering skill, not an edge case to ignore
- Why lowering `temperature` for tool-calling specifically (not just general chat) improves reliability

## ⚠️ A Known Limitation (and how it's handled)

Groq's `llama-3.3-70b-versatile` occasionally returns a malformed tool-call format instead of clean JSON — a documented issue across multiple projects, not specific to this app. This backend retries once automatically, and falls back to a direct answer (with a `note` field flagging it) if the issue persists, rather than returning a 500 error to the user.

## 🔮 What's Next

Day 28b — Streamlit frontend for this backend, connecting register/login/upload/ask into a real usable UI.

---
⭐ Part of my 100-day AI Engineering journey — [follow along on LinkedIn](https://www.linkedin.com/in/prashik-sawant-ds/) or check out [all 100 projects](https://github.com/PrashikSawant).
