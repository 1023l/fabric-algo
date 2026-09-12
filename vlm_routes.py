"""VLM 质检路由（切入点 2）。

接口：
  GET  /api/vlm/status          - VLM 配置状态
  POST /api/vlm/inspect/json    - 单张布片图质检（image_b64/path/url）
  POST /api/vlm/inspect_frame/json - 整帧复合：fabric 检出 → 逐片裁剪 → VLM 质检

说明：
  - VLM 调用不持 TRT 锁（检测部分持锁，VLM 是 HTTP 调用，避免长时间占锁阻塞检测主流程）
  - 未配置 VLM_API_KEY 时接口返回 400 带配置指引
"""

from __future__ import annotations

from time import time

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import vlm_qc
from algo_routes import ImageInput, _ensure_models, _lock, _ok, _read_from_body

router = APIRouter(prefix="/api/vlm", tags=["vlm"])

_INSPECT_PAD = 6       # 裁剪布片时四周扩边像素（给 VLM 多一点上下文）
_INSPECT_MAX_PIECES = 8  # 单帧最多质检片数（防止 API 费用/时延失控）


class InspectExtra(BaseModel):
    prompt_extra: str = Field("", description="追加给 VLM 的补充说明（如关注特定瑕疵）")
    max_pieces: int = Field(_INSPECT_MAX_PIECES, ge=1, le=32)


@router.get("/status")
def api_vlm_status():
    return vlm_qc.status()


@router.post("/inspect/json")
def api_vlm_inspect_json(body: ImageInput):
    """对单张（已裁剪的）布片图做 VLM 质检。"""
    t0 = time()
    img = _read_from_body(body)
    try:
        result = vlm_qc.inspect(img)
    except RuntimeError as e:  # 未配置
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(502, f"VLM 调用失败：{e}") from e
    return _ok({"result": result}, t0)


@router.post("/inspect_frame/json")
def api_vlm_inspect_frame_json(body: ImageInput):
    """整帧复合：fabric 检测 → 每片裁剪 → VLM 逐片质检。

    返回 pieces[]：{piece_box, conf, qc: {has_defect, defect_type, severity, description}}
    以及 summary：{inspected, defect_cnt, by_type}
    """
    t0 = time()
    img = _read_from_body(body)
    if not vlm_qc.is_configured():
        raise HTTPException(400, "VLM 未配置：请在 .env 设置 VLM_API_BASE / VLM_API_KEY / VLM_MODEL")

    # 1) fabric 检测（持 TRT 锁）
    with _lock:
        m = _ensure_models(1)
        dets = m["fabric"].detect(img)

    # 2) 按置信度取前 N 片，裁剪后逐片 VLM（不持锁）
    dets = sorted(dets, key=lambda d: -float(d["conf"]))[:_INSPECT_MAX_PIECES]
    H, W = img.shape[:2]
    pieces = []
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        x1, y1 = max(0, x1 - _INSPECT_PAD), max(0, y1 - _INSPECT_PAD)
        x2, y2 = min(W, x2 + _INSPECT_PAD), min(H, y2 + _INSPECT_PAD)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        try:
            qc = vlm_qc.inspect(crop)
            qc.pop("raw", None)
        except Exception as e:
            qc = {"has_defect": None, "defect_type": None, "severity": None,
                  "description": f"VLM 调用失败：{e}"}
        pieces.append({
            "piece_box": [round(float(v), 2) for v in d["box"]],
            "conf": round(float(d["conf"]), 4),
            "qc": qc,
        })

    defect_pieces = [p for p in pieces if p["qc"].get("has_defect")]
    by_type: dict[str, int] = {}
    for p in defect_pieces:
        t = p["qc"].get("defect_type") or "unknown"
        by_type[t] = by_type.get(t, 0) + 1
    summary = {
        "inspected": len(pieces),
        "defect_cnt": len(defect_pieces),
        "by_type": by_type,
    }
    return JSONResponse({
        "ok": True,
        "elapsed_ms": int((time() - t0) * 1000),
        "data": {"image_shape": list(img.shape[:2]), "summary": summary, "pieces": pieces},
    })
