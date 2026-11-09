import os
import uuid
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agent.orchestrator import run_agent

load_dotenv(override=True)  # .env must win over a stale/invalid key already in the shell env

app = FastAPI()


@app.exception_handler(Exception)
async def handle_uncaught_exception(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": f"{type(exc).__name__}: {str(exc)[:300]}"},
    )


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


# session_id -> list of {"role": "user"|"assistant", "content": str}. In-memory, single process.
sessions: dict[str, list] = {}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    sessions.setdefault(session_id, [])
    sessions[session_id].append({"role": "user", "content": request.message})

    try:
        response, tool_calls = run_agent(sessions[session_id])
    except Exception as e:
        response, tool_calls = f"Agent run failed: {type(e).__name__}: {str(e)[:300]}", []

    sessions[session_id].append({"role": "assistant", "content": response})
    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    sessions.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    # Cloud Run injects PORT and only routes to 0.0.0.0 -- 127.0.0.1 would
    # make the container unreachable from outside itself.
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
