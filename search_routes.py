"""以图搜图路由（切入点 4）。

接口：
  GET  /api/search/status       - 图库统计 + CLIP 状态
  POST /api/search/index/json   - 图入库：fabric 检出裁片（可选）或整图 → CLIP 向量 → 图库
  POST /api/search/query/json   - 搜图：以图 / 以文 → top-k 相似布片
  DELETE /api/search/items/{id} - 删除图库条目

图库检索主场景是"以图搜图"（客户发一张布面照片，找相似花型）；
以文搜图依赖 CLIP 文本塔，对英文描述效果较好，中文效果有限。
"""

from __future__ import annotations

from time import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import clip_embed
import search_store
from algo_routes import ImageInput, _ensure_models, _lock, _read_from_body

router = APIRouter(tags=["search"])

_CROP_PAD = 4


class SearchBody(BaseModel):
    """单一 body：图查（image_b64）与文查（text）二选一。"""
    image_b64: str = Field("", description="查询图片（data URL 或裸 b64）")
    text: str = Field("", description="文本查询（与 image_b64 二选一）")
    top_k: int = Field(10, ge=1, le=50)
    min_score: float = Field(0.05, description="相似度下限（余弦）")


@router.get("/api/search/status")
def api_search_status():
    return {**clip_embed.status(), "gallery_count": search_store.count()}


@router.post("/api/search/index/json")
def api_search_index_json(body: ImageInput, crop: bool = True, max_pieces: int = 6):
    """图入库。crop=true 时先跑 fabric 检测逐片入库（推荐，与产线流水线一致）；
    整图/无检出时整图入库一条。"""
    t0 = time()
    img = _read_from_body(body)
    items = []

    if crop:
        with _lock:
            m = _ensure_models(1)
            dets = m["fabric"].detect(img)
        dets = sorted(dets, key=lambda d: -float(d["conf"]))[:max(1, max_pieces)]
        H, W = img.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            x1, y1 = max(0, x1 - _CROP_PAD), max(0, y1 - _CROP_PAD)
            x2, y2 = min(W, x2 + _CROP_PAD), min(H, y2 + _CROP_PAD)
            piece = img[y1:y2, x1:x2]
            if piece.size == 0:
                continue
            try:
                items.append({**search_store.add(piece, source="detect"),
                              "conf": round(float(d["conf"]), 4)})
            except RuntimeError as e:  # CLIP 未就绪
                raise HTTPException(400, str(e)) from e

    if not items:  # 无检出或未要求裁片 → 整图入库
        try:
            items.append({**search_store.add(img, source="whole")})
        except RuntimeError as e:
            raise HTTPException(400, str(e)) from e

    return JSONResponse({
        "ok": True, "elapsed_ms": int((time() - t0) * 1000),
        "indexed": len(items), "gallery_count": search_store.count(), "items": items,
    })


@router.post("/api/search/query/json")
def api_search_query_json(body: SearchBody):
    """搜图：传图（image_b64，自动取主片）或传文本（text）。"""
    t0 = time()
    top_k = body.top_k
    use_text = body.text.strip()

    if body.image_b64.strip():
        img_input = ImageInput(image_b64=body.image_b64)
        img = _read_from_body(img_input)
        H, W = img.shape[:2]
        vec = None
        with _lock:  # 有 fabric 模型就用主片，更聚焦
            try:
                m = _ensure_models(1)
                dets = m["fabric"].detect(img)
            except Exception:
                dets = []
        if dets:
            d = max(dets, key=lambda d: float(d["conf"]))
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            piece = img[max(0, y1 - _CROP_PAD):min(H, y2 + _CROP_PAD),
                        max(0, x1 - _CROP_PAD):min(W, x2 + _CROP_PAD)]
            if piece.size:
                vec = clip_embed.embed_image(piece)
        if vec is None:
            vec = clip_embed.embed_image(img)
        query_type = "image"
    elif use_text:
        vec = clip_embed.embed_text(use_text)
        query_type = "text"
    else:
        raise HTTPException(400, "请提供图片（image_b64）或文本（text）")

    try:
        hits = search_store.search(vec, top_k=top_k, min_score=body.min_score)
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e

    return JSONResponse({
        "ok": True, "elapsed_ms": int((time() - t0) * 1000),
        "query_type": query_type, "count": len(hits), "results": hits,
    })


@router.get("/search")
def search_page():
    """搜图页面（与 /qa 页面风格一致）。"""
    from fastapi.responses import FileResponse
    page = search_store.ROOT / "static" / "search.html"
    if page.exists():
        return FileResponse(str(page))
    raise HTTPException(404, "search.html 不存在")


@router.delete("/api/search/items/{item_id}")
def api_search_delete(item_id: int):
    if not search_store.remove(item_id):
        raise HTTPException(404, f"条目不存在: {item_id}")
    return JSONResponse({"ok": True, "gallery_count": search_store.count()})
