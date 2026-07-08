"""
agent.py — Tool-calling agent (same pattern as Day 24), except the "tool"
today is our own RAG search function, scoped per authenticated user.
"""

import json
import os

from groq import BadRequestError, Groq

import rag_engine

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- Tool definition (tells the model this function exists) ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the user's uploaded documents for relevant information to answer their question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the user's documents",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

SYSTEM_PROMPT = (
    "You are a research assistant. If the user's question could be answered from their "
    "uploaded documents, use the search_documents tool. Always cite the filename when you "
    "use information from a document. If no documents are relevant or none exist, answer "
    "from your own knowledge and say so clearly."
)

# Groq's llama-3.3-70b-versatile occasionally emits a malformed tool-call format
# (a known, documented issue — not something in our code). We retry once with
# tool calling, and if it still fails, we fall back to a normal answer without
# tools rather than crashing the whole request.
MAX_TOOL_CALL_ATTEMPTS = 2


def _call_with_tools(messages: list):
    """Attempts the tool-calling API call, retrying on Groq's known tool_use_failed bug.
    Returns the response message, or None if every attempt failed."""
    for attempt in range(MAX_TOOL_CALL_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,  # lower temperature = more reliable, cleaner tool-call formatting
            )
            return response.choices[0].message
        except BadRequestError as e:
            if "tool_use_failed" in str(e) and attempt < MAX_TOOL_CALL_ATTEMPTS - 1:
                continue  # retry once — this specific error is often transient
            return None  # give up on tools, caller will fall back to a plain answer
    return None


def run_agent(user_id: str, question: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # ---- First call: let the model decide whether it needs the tool ----
    response_message = _call_with_tools(messages)
    sources_used = []
    tool_calls = response_message.tool_calls if response_message else None

    if response_message is None:
        # Tool calling failed even after retrying — fall back to a plain answer,
        # so the user still gets a response instead of a 500 error.
        fallback = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
        )
        return {
            "answer": fallback.choices[0].message.content,
            "sources": [],
            "used_documents": False,
            "note": "Answered without document search due to a temporary tool-calling issue.",
        }

    if tool_calls:
        # The model wants to call search_documents — append its request to history
        messages.append(response_message)

        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments)
            query = args.get("query", question)

            # Execute the tool — scoped strictly to this user's documents
            results = rag_engine.search_documents(user_id=user_id, query=query)
            sources_used.extend([r["filename"] for r in results])

            tool_result_text = "\n\n".join(
                [f"[From {r['filename']}]: {r['text']}" for r in results]
            ) or "No relevant documents found."

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result_text,
            })

        # ---- Second call: model writes the final answer using tool results ----
        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
        )
        answer = final_response.choices[0].message.content
    else:
        # Model answered directly, no document search needed
        answer = response_message.content

    return {
        "answer": answer,
        "sources": list(set(sources_used)),
        "used_documents": bool(tool_calls),
    }