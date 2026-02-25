# app/main.py
from __future__ import annotations

import json
import os
from uuid import uuid4
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.graph import build_graph

def _to_jsonable(obj):
    # 基本类型
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # dict / list
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    # LangGraph Interrupt 对象：取它的 value
    if obj.__class__.__name__ == "Interrupt" and hasattr(obj, "value"):
        return {"__type__": "Interrupt", "value": _to_jsonable(getattr(obj, "value"))}

    # pydantic
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump())
    if hasattr(obj, "dict"):
        try:
            return _to_jsonable(obj.dict())
        except Exception:
            pass

    # 兜底：转字符串
    return str(obj)

def sse(event: str, data):
    payload = json.dumps(_to_jsonable(data), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

class RunRequest(BaseModel):
    topic: str
    mode: str = "approval"   # "auto" or "approval"

class ResumeRequest(BaseModel):
    value: Any               # True/False 或更复杂 JSON（以后可扩展）

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("data", exist_ok=True)
    # SQLite checkpointer：适合第一版做 durable execution（可恢复/可审批）:contentReference[oaicite:8]{index=8}
    with SqliteSaver.from_conn_string("data/checkpoints.sqlite") as checkpointer:
        app.state.graph = build_graph(checkpointer)
        yield

app = FastAPI(lifespan=lifespan)

@app.post("/runs/stream")
def run_stream(req: RunRequest):
    graph = app.state.graph
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    inputs = {"topic": req.topic, "mode": req.mode, "status": "init"}

    def gen():
        yield sse("meta", {"thread_id": thread_id})

        try:
            for chunk in graph.stream(...):
                if isinstance(chunk, dict) and "__interrupt__" in chunk:
                    continue
                yield sse("update", chunk)
        except Exception as e:
            yield sse("error", {"message": str(e)})
            return

        # ✅ 不要 invoke 了，改用 get_state
        snapshot = graph.get_state(config)
        if getattr(snapshot, "interrupts", None):
            payloads = []
            for it in snapshot.interrupts:
                payloads.append(getattr(it, "value", it))
            yield sse("interrupt", {"thread_id": thread_id, "interrupts": payloads})
        else:
            yield sse("done", {"thread_id": thread_id, "assets": snapshot.values.get("assets")})

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/runs/{thread_id}/resume/stream")
def resume_stream(thread_id: str, req: ResumeRequest):
    graph = app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    def gen():
        yield sse("meta", {"thread_id": thread_id, "resuming": True})

        try:
            for chunk in graph.stream(Command(resume=req.value), config, stream_mode="updates"):
                if isinstance(chunk, dict) and "__interrupt__" in chunk:
                    continue
                yield sse("update", chunk)
        except Exception as e:
            yield sse("error", {"message": str(e)})
            return

        # ✅ 不要 invoke，改用 get_state
        snapshot = graph.get_state(config)
        if getattr(snapshot, "interrupts", None):
            payloads = []
            for it in snapshot.interrupts:
                payloads.append(getattr(it, "value", it))
            yield sse("interrupt", {"thread_id": thread_id, "interrupts": payloads})
        else:
            yield sse("done", {"thread_id": thread_id, "assets": snapshot.values.get("assets")})

    return StreamingResponse(gen(), media_type="text/event-stream")