"""VLM 视觉质检模块（切入点 2）。

用视觉语言模型对布片裁剪图做"看图判断"：有无瑕疵、类型、严重程度、中文描述。
通过 OpenAI 兼容接口调用，支持：
  - DashScope 兼容模式（qwen-vl-max / qwen-vl-plus，云端 API）
  - 本地 vLLM / Ollama 等 OpenAI 兼容端点（Qwen-VL 本地部署）

环境变量（或项目根 .env）：
  VLM_API_BASE  - 默认 https://dashscope.aliyuncs.com/compatible-mode/v1
  VLM_API_KEY   - API Key（本地端点随便填非空）
  VLM_MODEL     - 默认 qwen-vl-max
  VLM_TIMEOUT   - 请求超时秒数，默认 60
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-max"

_INSPECT_PROMPT = """你是纺织布片质检员。请仔细检查这张布片图片，判断是否存在瑕疵\
（常见瑕疵：破洞、污渍/油污、断纱、跳纱、色差、破损、异物、褶皱损伤等）。

要求：
1. 只看布面本身，忽略图片边缘的背景
2. 判断不出时如实说判断不出，不要编造
3. 严格只输出一个 JSON 对象，不要输出任何其他内容，格式：
{"has_defect": true或false, "defect_type": "瑕疵类型（无瑕疵填 null）", "severity": "none|light|heavy", "description": "一句话中文描述（是什么、位置、程度）"}"""

_client = None
_client_cfg: tuple[str, str, str] | None = None


def _get_client():
    """懒加载 OpenAI 客户端（配置变更时重建）。"""
    global _client, _client_cfg
    base = os.getenv("VLM_API_BASE", DEFAULT_BASE).strip()
    key = os.getenv("VLM_API_KEY", "").strip()
    model = os.getenv("VLM_MODEL", DEFAULT_MODEL).strip()
    cfg = (base, key, model)
    if _client is None or _client_cfg != cfg:
        if not key:
            raise RuntimeError(
                "VLM_API_KEY 未设置。请在 .env 配置 VLM_API_BASE / VLM_API_KEY / VLM_MODEL"
                "（本地 vLLM 端点 Key 可填任意非空字符串）"
            )
        from openai import OpenAI
        _client = OpenAI(api_key=key, base_url=base, timeout=float(os.getenv("VLM_TIMEOUT", "60")))
        _client_cfg = cfg
    return _client, model


def is_configured() -> bool:
    return bool(os.getenv("VLM_API_KEY", "").strip())


def status() -> dict:
    return {
        "configured": is_configured(),
        "api_base": os.getenv("VLM_API_BASE", DEFAULT_BASE),
        "model": os.getenv("VLM_MODEL", DEFAULT_MODEL),
    }


def encode_jpg(img: np.ndarray, max_side: int = 768, quality: int = 85) -> str:
    """ndarray → JPEG data URL（大图先缩，省 token 和带宽）。"""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _extract_json(text: str) -> dict:
    """从模型回复里提取 JSON（容忍 markdown 围栏和前后杂文）。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            text = m.group(0)
    return json.loads(text)


def inspect(img: np.ndarray, *, extra_prompt: str = "") -> dict:
    """对一张布片裁剪图做质检判断，返回结构化结果。

    返回：{"has_defect": bool, "defect_type": str|None,
           "severity": "none|light|heavy", "description": str, "raw": str}
    """
    client, model = _get_client()
    data_url = encode_jpg(img)
    prompt = _INSPECT_PROMPT + (f"\n\n补充说明：{extra_prompt}" if extra_prompt else "")
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or ""
    try:
        obj = _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        return {
            "has_defect": None, "defect_type": None, "severity": None,
            "description": "模型输出无法解析为 JSON", "raw": raw,
        }
    # 字段规范化
    out = {
        "has_defect": bool(obj.get("has_defect")),
        "defect_type": obj.get("defect_type") or None,
        "severity": obj.get("severity") if obj.get("severity") in ("none", "light", "heavy") else None,
        "description": obj.get("description") or "",
        "raw": raw,
    }
    if not out["has_defect"]:
        out["defect_type"] = None
        out["severity"] = "none"
    return out
