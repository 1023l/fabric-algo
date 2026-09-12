"""数据问答 Agent 路由（挂到 web_server 的 FastAPI 上）。

接口：
  POST /api/qa/ask         - 同步问答
  POST /api/qa/ask/stream  - SSE 流式（逐步推送思考链）
  GET  /qa                 - 对话页面
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from utils import ROOT
from qa_agent.llm import LLMConfig, LLMClient
from qa_agent.runtime import AgentRuntime

router = APIRouter()

_STATIC = Path(__file__).resolve().parent / "static"


def _load_env() -> None:
    """加载项目根 .env（QA_LLM_* 等变量；已设置的环境变量优先）。"""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().replace("export ", "")
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env()

# LLM 客户端进程级单例（OpenAI client 线程安全，可并发请求）
_llm_lock = threading.Lock()
_agent: AgentRuntime | None = None
_agent_err: str | None = None


def get_agent() -> AgentRuntime:
    global _agent, _agent_err
    with _llm_lock:
        if _agent is None:
            try:
                _agent = AgentRuntime(LLMClient(LLMConfig.from_env()))
                _agent_err = None
            except Exception as e:  # 未配 Key 时给出可读错误
                _agent_err = str(e)
                raise
    return _agent


class AskRequest(BaseModel):
    question: str


@router.get("/qa")
async def qa_page():
    """问答对话页。"""
    page = _STATIC / "qa.html"
    if page.exists():
        return FileResponse(str(page))
    return FileResponse(str(_STATIC / "index.html"))


@router.post("/api/qa/ask")
async def qa_ask(req: AskRequest):
    """同步问答：阻塞直到 Agent 完成，返回完整结果。"""
    agent = get_agent()
    result = agent.run(req.question)
    return {
        "ok": True,
        "answer": result.answer,
        "steps": result.to_trace(),
        "total_tokens": result.total_tokens,
    }


@router.post("/api/qa/ask/stream")
async def qa_ask_stream(req: AskRequest):
    """SSE 流式问答：逐步推送每个 ReAct 步骤，前端实时展示思考链。"""
    try:
        agent = get_agent()
    except Exception as e:
        err_msg = f"LLM 未配置：{e}\n请在项目根 .env 配置 QA_LLM_API_KEY / QA_LLM_BASE_URL / QA_LLM_MODEL"

        async def cfg_error():
            yield f"data: {json.dumps({'is_final': True, 'answer': err_msg, 'thought': ''}, ensure_ascii=False)}\n\n"
        return StreamingResponse(cfg_error(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def event_stream():
        try:
            for step_data in agent.run_stream(req.question):
                yield f"data: {json.dumps(step_data, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'is_final': True, 'answer': f'Agent 运行出错：{e}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )
