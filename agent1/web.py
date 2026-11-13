from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any

import uvicorn
from agent import MODEL_NAME, StockAgent, _configure_stdout
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "static" / "index.html"

app = FastAPI(title="MOEX Stock Agent", version="1.0.0")

_agent: StockAgent | None = None
_agent_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


def get_agent() -> StockAgent:
    global _agent
    if _agent is not None:
        return _agent
    with _agent_lock:
        if _agent is None:
            try:
                _agent = StockAgent()
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Agent initialization failed: {exc}",
                ) from exc
    return _agent


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="static/index.html is missing.",
        )
    return FileResponse(INDEX_FILE)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "agent_ready": _agent is not None,
    }


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    agent = get_agent()
    try:
        result = agent.ask(question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent request failed: {exc}",
        ) from exc

    trace = result["trace"]
    return {
        "answer": result["answer"],
        "trace_id": trace["id"],
        "model": trace["model"],
        "steps_used": trace["steps_used"],
        "finish_reason": trace["finish_reason"],
        "warning": trace["warning"],
        "tool_calls": [
            {
                "step": call["step"],
                "name": call["name"],
                "arguments": call["arguments"],
                "result": call["result"],
            }
            for call in trace["tool_calls"]
        ],
        "trace_path": result["trace_path"],
    }


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Run the stock-agent web interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Open http://{args.host}:{args.port} in your browser.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
