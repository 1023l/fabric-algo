# -*- coding: utf-8 -*-
"""
第三方软件可直接调用的算法 HTTP 接口层（与 Web 实时页面解耦）。

挂载前缀：/api/algo
提供 4 个原子接口 + 1 个复合接口：
  POST /api/algo/fabric         - 目标检测（fabric 模型）
  POST /api/algo/text           - 文字区域检测（text 模型）
  POST /api/algo/ocr            - 单张裁剪图 OCR 识别（rec 模型，返回 text+score）
  POST /api/algo/pieces_ocr     - 复合：目标检测 + 文字区检测 + 识别（每片返回 piece_box+texts+size+lr）
  POST /api/algo/count_frame    - 复合：带计数状态机的一帧处理（需调用方用 session_id 持状态）

输入（4 种格式选 1，按优先级解析）：
  a) multipart/form-data   字段名 image，传文件（jpg/png/bmp）
  b) JSON body: {"image_url": "http://.../a.jpg"}   （HTTP(S) URL，会下载）
  c) JSON body: {"image_path": "C:/abs/or/rel/to/ROOT.jpg"} （本地绝对或相对路径，中文安全读取）
  d) JSON body: {"image_b64": "data:image/jpeg;base64,..." 或 纯 base64 字符串}

线程安全：对每个模型访问加 RLock（TRT session 不能跨线程并发）。
"""

from __future__ import annotations

import base64
import re
import threading
import time
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from utils import ROOT
from onnx_engine import TrtOnnxDetector, RecTrtEngine
from tracker import SimpleTracker
from counter import LineCounter, extract_size
from infer_business import make_ocr_callback


# --------- 默认路径（与 web_server.py 共用同一套模型） ---------
MODEL_DIR = ROOT / "models"

# --------- 默认阈值（与 web_server 同步，可通过 JSON body 覆盖） ---------
DEFAULT_FABRIC_CONF = 0.4
DEFAULT_TEXT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_REC_THRESH = 0.35
DEFAULT_REC_MAX_W = 192
DEFAULT_IMGSZ = 640

router = APIRouter(prefix="/api/algo", tags=["algo"])

# --------- 模型单例 + 锁（RLock 允许同线程重入） ---------
_lock = threading.RLock()
_models: dict[str, Any] = {"fabric": None, "text": None, "rec": None, "rec_n": 1}


def _ensure_models(rec_workers: int = 1):
    """懒加载模型。engine 版本读 models/current.json（模型管理上传后自动热替换）。
    返回 {fabric, text, rec: [engine,...]}。"""
    with _lock:
        if _models["fabric"] is None:
            from model_admin import engine_paths
            eng = engine_paths()
            _models["fabric"] = TrtOnnxDetector(eng["fabric"], imgsz=DEFAULT_IMGSZ,
                                                conf=DEFAULT_FABRIC_CONF, iou=DEFAULT_IOU)
            _models["text"] = TrtOnnxDetector(eng["text"], imgsz=DEFAULT_IMGSZ,
                                              conf=DEFAULT_TEXT_CONF, iou=DEFAULT_IOU)
        if _models["rec"] is None or len(_models["rec"]) < rec_workers:
            # 初始按请求 worker 数扩容；上限 8，避免显存爆炸
            from model_admin import engine_paths
            eng = engine_paths()
            n = max(1, min(8, int(rec_workers)))
            engines = list(_models["rec"] or [])
            while len(engines) < n:
                engines.append(RecTrtEngine.load(eng["rec"],
                                                 score_thresh=DEFAULT_REC_THRESH,
                                                 max_width=DEFAULT_REC_MAX_W))
            _models["rec"] = engines
            _models["rec_n"] = n
        return _models


# ============================================================
# 图像输入解析（4 种方式：上传文件 / URL / 本地路径 / base64）
# ============================================================
class ImageInput(BaseModel):
    image_url: str | None = Field(None, description="公网/内网可访问图像 URL，http(s)://")
    image_path: str | None = Field(None, description="本地图像路径（绝对，或相对 fabric-algo 根 ROOT）")
    image_b64: str | None = Field(None, description="base64 图像；可带 data:image/xx;base64, 前缀")
    # 阈值覆写（可选）
    fabric_conf: float | None = None
    text_conf: float | None = None
    iou: float | None = None
    rec_thresh: float | None = None
    rec_max_w: int | None = None
    # 计数相关
    line_ratio: float | None = None
    line_y: int | None = None
    dead_zone: int | None = None
    session_id: str | None = Field(None, description="计数状态机会话；同一条视频流保持相同 session_id")
    ocr: bool = Field(True, description="是否做文字识别（L/R/鞋码需要，纯计数可关掉更快）")
    rec_workers: int = Field(1, ge=1, le=8, description="OCR 时 rec 并行 worker 数")


def _cv_imread(path: str) -> np.ndarray | None:
    """中文路径安全读图：np.fromfile + cv2.imdecode（避免 cv2.imread 对非 ASCII 路径失败）。"""
    try:
        arr = np.fromfile(path, dtype=np.uint8)
        if arr.size == 0:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def _decode_np(raw_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "图像解码失败（不是合法的 jpg/png/bmp/webp 字节流？）")
    return img


def _read_from_body(body: ImageInput) -> np.ndarray:
    if body.image_b64:
        s = body.image_b64
        m = re.match(r"data:image/[\w+]+;base64,(.*)", s, flags=re.I)
        if m:
            s = m.group(1)
        s = re.sub(r"\s+", "", s)
        try:
            raw = base64.b64decode(s)
        except Exception as e:
            raise HTTPException(400, f"image_b64 不合法: {e}") from e
        return _decode_np(raw)
    if body.image_url:
        if not re.match(r"^https?://", body.image_url, flags=re.I):
            raise HTTPException(400, "image_url 必须是 http(s):// 开头")
        try:
            with urllib.request.urlopen(body.image_url, timeout=15) as resp:
                raw = resp.read()
        except Exception as e:
            raise HTTPException(400, f"下载 image_url 失败: {e}") from e
        return _decode_np(raw)
    if body.image_path:
        p = Path(body.image_path)
        if not p.is_absolute():
            p = ROOT / p
        img = _cv_imread(str(p))
        if img is None:
            raise HTTPException(400, f"image_path 读不到: {p}")
        return img
    raise HTTPException(400, "请提供 image_url / image_path / image_b64 中的至少一个")


def _apply_det_thresholds(m, body: ImageInput, which: str):
    """临时覆写 fabric/text detector 的 conf/iou（with lock 内用，使用完恢复），
    返回 (orig_conf, orig_iou) 供调用方 finally 回写。"""
    det = m[which]
    new_conf = {"fabric": body.fabric_conf, "text": body.text_conf}[which]
    new_iou = body.iou
    old = (det.conf, det.iou)
    if new_conf is not None:
        det.conf = float(new_conf)
    if new_iou is not None:
        det.iou = float(new_iou)
    return old


# ============================================================
# 响应包装：统一 {"ok":bool,"elapsed_ms":int,"data":...}
# ============================================================
def _ok(data: Any, t0: float):
    return JSONResponse({"ok": True, "elapsed_ms": int((time.time() - t0) * 1000), "data": data})


# ============================================================
# 1. 目标检测
# ============================================================
@router.post("/fabric")
async def api_algo_fabric(
    image: UploadFile | None = File(None, description="上传图像文件（jpg/png/bmp），与 JSON body 二选一"),
):
    """返回：[{box:[x1,y1,x2,y2], cls:int, conf:float}]（原图坐标，像素）。"""
    from time import time
    t0 = time()
    # a) 上传文件优先
    if image is not None:
        raw = await image.read()
        img = _decode_np(raw)
    else:
        # 未传 multipart 文件 → 请用 /fabric/json（image_url / image_path / image_b64）
        raise HTTPException(400, "/fabric 目前支持 multipart/form-data 上传 image 文件，或请用 POST /fabric/json")
    with _lock:
        m = _ensure_models(1)
        dets = m["fabric"].detect(img)
    out = [
        {"box": [round(float(v), 2) for v in d["box"]], "cls": int(d["cls"]), "conf": round(float(d["conf"]), 4)}
        for d in dets
    ]
    return _ok({"image_shape": list(img.shape[:2]), "boxes": out}, t0)


class _Body(ImageInput):
    pass


@router.post("/fabric/json")
def api_algo_fabric_json(body: _Body):
    """与 /fabric 等价，但通过 JSON body（image_url / image_path / image_b64）传图。"""
    from time import time
    t0 = time()
    img = _read_from_body(body)
    with _lock:
        m = _ensure_models(1)
        old = _apply_det_thresholds(m, body, "fabric")
        try:
            dets = m["fabric"].detect(img)
        finally:
            m["fabric"].conf, m["fabric"].iou = old
    out = [
        {"box": [round(float(v), 2) for v in d["box"]], "cls": int(d["cls"]), "conf": round(float(d["conf"]), 4)}
        for d in dets
    ]
    return _ok({"image_shape": list(img.shape[:2]), "boxes": out}, t0)


# ============================================================
# 2. 文字区域检测
# ============================================================
@router.post("/text/json")
def api_algo_text_json(body: _Body):
    """文字区域检测。cls=0 -> text_h（横文），cls=1 -> text_v（竖文，后续 OCR 前需 rot90 k=3）。"""
    from time import time
    t0 = time()
    img = _read_from_body(body)
    with _lock:
        m = _ensure_models(1)
        old = _apply_det_thresholds(m, body, "text")
        try:
            dets = m["text"].detect(img)
        finally:
            m["text"].conf, m["text"].iou = old
    class_names = {0: "text_h", 1: "text_v"}
    out = [
        {
            "box": [round(float(v), 2) for v in d["box"]],
            "cls": int(d["cls"]),
            "cls_name": class_names.get(int(d["cls"]), str(d["cls"])),
            "conf": round(float(d["conf"]), 4),
        }
        for d in dets
    ]
    return _ok({"image_shape": list(img.shape[:2]), "boxes": out}, t0)


@router.post("/text")
async def api_algo_text(image: UploadFile | None = File(None)):
    from time import time
    t0 = time()
    if image is None:
        raise HTTPException(400, "请上传 image 文件，或调用 /text/json 用 JSON body")
    raw = await image.read()
    img = _decode_np(raw)
    with _lock:
        m = _ensure_models(1)
        dets = m["text"].detect(img)
    class_names = {0: "text_h", 1: "text_v"}
    out = [
        {
            "box": [round(float(v), 2) for v in d["box"]],
            "cls": int(d["cls"]),
            "cls_name": class_names.get(int(d["cls"]), str(d["cls"])),
            "conf": round(float(d["conf"]), 4),
        }
        for d in dets
    ]
    return _ok({"image_shape": list(img.shape[:2]), "boxes": out}, t0)


# ============================================================
# 3. OCR 单张裁剪图识别（rec 模型直接用）
# ============================================================
@router.post("/ocr/json")
def api_algo_ocr_json(body: _Body):
    """直接识别一张已经裁剪好、摆正的文字小图（横文，竖文请先 rot90 k=3 后再调用）。"""
    from time import time
    t0 = time()
    img = _read_from_body(body)
    thresh = body.rec_thresh if body.rec_thresh is not None else DEFAULT_REC_THRESH
    maxw = int(body.rec_max_w or DEFAULT_REC_MAX_W)
    with _lock:
        m = _ensure_models(1)
        engine = m["rec"][0]
        _old = (engine.score_thresh, engine.max_width)
        try:
            engine.score_thresh = float(thresh)
            engine.max_width = maxw
            txt, sc = engine.recognize(img)
        finally:
            engine.score_thresh, engine.max_width = _old
    size = extract_size(txt) or None
    lr = None
    if txt:
        up = txt.upper()
        flags = []
        if "L" in up: flags.append("L")
        if "R" in up: flags.append("R")
        if flags: lr = "/".join(flags)
    return _ok({
        "image_shape": list(img.shape[:2]),
        "text": txt,
        "score": round(float(sc), 4),
        "shoe_size": size,
        "lr_flag": lr,
    }, t0)


@router.post("/ocr")
async def api_algo_ocr(image: UploadFile | None = File(None),
                       rec_thresh: float = Form(DEFAULT_REC_THRESH),
                       rec_max_w: int = Form(DEFAULT_REC_MAX_W)):
    from time import time
    t0 = time()
    if image is None:
        raise HTTPException(400, "请上传 image 文件，或调用 /ocr/json 用 JSON body")
    raw = await image.read()
    img = _decode_np(raw)
    with _lock:
        m = _ensure_models(1)
        engine = m["rec"][0]
        _old = (engine.score_thresh, engine.max_width)
        try:
            engine.score_thresh = float(rec_thresh)
            engine.max_width = int(rec_max_w)
            txt, sc = engine.recognize(img)
        finally:
            engine.score_thresh, engine.max_width = _old
    size = extract_size(txt) or None
    lr = None
    if txt:
        up = txt.upper()
        flags = []
        if "L" in up: flags.append("L")
        if "R" in up: flags.append("R")
        if flags: lr = "/".join(flags)
    return _ok({
        "image_shape": list(img.shape[:2]),
        "text": txt,
        "score": round(float(sc), 4),
        "shoe_size": size,
        "lr_flag": lr,
    }, t0)


# ============================================================
# 4. 复合：目标检测 + 文字区检测 + OCR（每片独立结果）
# ============================================================
_count_sessions: dict[str, dict] = {}
# 上面 count_frame 共享 session，这里 pieces_ocr 不走 session。


@router.post("/pieces_ocr/json")
def api_algo_pieces_ocr_json(body: _Body):
    """
    整帧调用（与实时页面逻辑一致）：
      先 fabric 检测目标 -> 对每片裁剪 -> text 检测文字区 -> 竖文 rot90 -> rec 识别
    返回 pieces[]：每个 piece = {piece_box, track_id（单帧恒等于索引）,
                          text_regions[{text_box, cls, cls_name, text, score}],
                          text (拼接后), shoe_size, lr_flag}
    """
    from time import time
    t0 = time()
    img = _read_from_body(body)
    with _lock:
        m = _ensure_models(max(1, int(body.rec_workers)))
        # 临时覆写 conf/iou/rec_thresh
        old_fab = _apply_det_thresholds(m, body, "fabric")
        old_txt = _apply_det_thresholds(m, body, "text")
        try:
            # 1) fabric
            fabric_dets = m["fabric"].detect(img)
            # 2) 单帧调用：不需要跨帧 tracker（tracker 首帧只建 track 不输出，会丢目标），
            #    直接用检测结果，track_id = 索引
            tracked = [
                {"track_id": i + 1, "box": d["box"], "cls": d["cls"], "conf": d["conf"]}
                for i, d in enumerate(fabric_dets)
            ]
            pieces: list[dict] = []
            if not body.ocr:
                # 只返回位置，不做 OCR
                for t in tracked:
                    pieces.append({
                        "track_id": int(t["track_id"]),
                        "piece_box": [round(float(v), 2) for v in t["box"]],
                        "conf": round(float(t["conf"]), 4),
                        "text_regions": [],
                        "text": None,
                        "shoe_size": None,
                        "lr_flag": None,
                    })
                return _ok({"image_shape": list(img.shape[:2]), "count": len(pieces), "pieces": pieces}, t0)
            # 3) 用 make_ocr_callback 复用 OCR 流水线（含竖文旋转 + 多 worker rec）
            timing = {"text_s": 0, "text_n": 0, "rec_s": 0, "rec_n": 0}
            thresh = float(body.rec_thresh or DEFAULT_REC_THRESH)
            cb = make_ocr_callback(m["text"], m["rec"], pad=0, rec_thresh=thresh,
                                   timing=timing, draw_boxes=False)
            # 构造 counter 要求的 infos（mock 每片都刚过线触发 OCR）
            infos = [{"track_id": t["track_id"], "box": t["box"], "cls": t["cls"],
                      "conf": t["conf"], "crossed": True} for t in tracked]
            per_tid_texts = cb(img, infos, 0) or {}
            # 补 text_regions 详细信息（为了把每个 text_box 也吐出来给调用方）
            # 这里再跑一次 text 检测，只为给 JSON 返回详细的 text_box 坐标（make_ocr_callback
            # 里只把文字拼接结果返回了，没保留 box）。性能：一张帧的 text det 5ms，可接受。
            per_tid_regions: dict[int, list] = {}
            for t in tracked:
                tid = int(t["track_id"])
                x1, y1, x2, y2 = map(int, t["box"])
                x1, y1 = max(0, x1), max(0, y1)
                piece = img[y1:y2, x1:x2]
                if piece.size == 0:
                    continue
                text_dets = m["text"].detect(piece)
                regs = per_tid_regions.setdefault(tid, [])
                for d in text_dets:
                    bx1, by1, bx2, by2 = map(float, d["box"])
                    cls_name = "text_v" if int(d["cls"]) == 1 else "text_h"
                    # 把 text box 映射回全局坐标，也提供 piece 局部坐标
                    regs.append({
                        "cls": int(d["cls"]),
                        "cls_name": cls_name,
                        "conf": round(float(d["conf"]), 4),
                        "box_local": [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)],
                        "box_global": [round(x1 + bx1, 2), round(y1 + by1, 2),
                                       round(x1 + bx2, 2), round(y1 + by2, 2)],
                    })
            # 组装
            for t in tracked:
                tid = int(t["track_id"])
                txt = per_tid_texts.get(tid, "") or None
                sz = extract_size(txt) if txt else None
                lr = None
                if txt:
                    up = txt.upper()
                    flags = []
                    if "L" in up: flags.append("L")
                    if "R" in up: flags.append("R")
                    if flags: lr = "/".join(flags)
                pieces.append({
                    "track_id": tid,
                    "piece_box": [round(float(v), 2) for v in t["box"]],
                    "conf": round(float(t["conf"]), 4),
                    "text_regions": per_tid_regions.get(tid, []),
                    "text": txt,
                    "shoe_size": sz,
                    "lr_flag": lr,
                })
            return _ok({
                "image_shape": list(img.shape[:2]),
                "count": len(pieces),
                "pieces": pieces,
            }, t0)
        finally:
            m["fabric"].conf, m["fabric"].iou = old_fab
            m["text"].conf, m["text"].iou = old_txt


# ============================================================
# 5. 复合：计数 + 一帧（状态机通过 session_id 持状态）
# ============================================================
def _count_session(sid: str, create: bool, height: int, line_ratio: float, dead_zone: int, line_y: int | None):
    if sid in _count_sessions:
        return _count_sessions[sid]
    if not create:
        return None
    if line_y is None:
        line_y = int(height * line_ratio)
    sess = {
        "tracker": SimpleTracker(max_age=60, iou_thresh=0.3, min_hits=1, next_id=1),
        "counter": LineCounter(line_y=line_y, dead_zone=dead_zone, max_track_age=60),
        "persist_texts": {},
        "persist_sizes": {},
        "size_counts": {},
        "lr_l": 0,
        "lr_r": 0,
        "pending_ocr": {},
    }
    _count_sessions[sid] = sess
    return sess


@router.post("/count_frame/json")
def api_algo_count_frame_json(body: _Body):
    """
    一帧过线计数（与 counter.run_count_video_generic 内部算法 1:1）。
    session_id 必传：调用方保持同一视频流相同 session_id，否则计数从零开始。
    返回同实时页面的 stats（down/up/net/lr_l/lr_r/pairs/single/sizes、per_piece 详细信息）。
    """
    from time import time
    t0 = time()
    if not body.session_id:
        raise HTTPException(400, "count_frame 需要传 session_id（同一条视频用同一个）")
    img = _read_from_body(body)
    H, W = img.shape[:2]
    line_ratio = float(body.line_ratio if body.line_ratio is not None
                       else 0.5)
    line_y = int(body.line_y) if body.line_y is not None else int(H * line_ratio)
    dead_zone = int(body.dead_zone if body.dead_zone is not None else 15)

    with _lock:
        m = _ensure_models(max(1, int(body.rec_workers)))
        old_fab = _apply_det_thresholds(m, body, "fabric")
        old_txt = _apply_det_thresholds(m, body, "text")
        try:
            sess = _count_session(body.session_id, True, H, line_ratio, dead_zone, line_y)
            # 允许后续帧调整 line_y（拖动计数线场景）
            sess["counter"].line_y = line_y

            # 1) fabric 检测 + 跟踪
            fab_dets = m["fabric"].detect(img)
            tracked = sess["tracker"].update(fab_dets)
            if tracked:
                boxes = [t["box"] for t in tracked]
                ids = [t["track_id"] for t in tracked]
                clss = [t["cls"] for t in tracked]
                confs = [t["conf"] for t in tracked]
                infos = sess["counter"].process(boxes, ids, clss, confs)
            else:
                infos = []

            # 2) OCR（仅当过线 triggered 且 body.ocr=True）
            per_tid_texts: dict[int, str] = {}
            if body.ocr:
                # pending_ocr 计数
                new_pending: list[int] = []
                for info in infos:
                    if info.get("crossed"):
                        sess["pending_ocr"][int(info["track_id"])] = 5  # 5 帧窗口
                for tid in list(sess["pending_ocr"]):
                    sess["pending_ocr"][tid] -= 1
                    if sess["pending_ocr"][tid] <= 0:
                        del sess["pending_ocr"][tid]
                ocr_infos = [i for i in infos if int(i["track_id"]) in sess["pending_ocr"]]
                if ocr_infos:
                    thresh = float(body.rec_thresh or DEFAULT_REC_THRESH)
                    timing = {"text_s": 0, "text_n": 0, "rec_s": 0, "rec_n": 0}
                    cb = make_ocr_callback(m["text"], m["rec"], pad=0, rec_thresh=thresh,
                                           timing=timing, draw_boxes=False)
                    per_tid_texts = cb(img, ocr_infos, 0) or {}
                    # 合并进 persist
                    for tid, txt in per_tid_texts.items():
                        if tid not in sess["persist_texts"]:
                            up = txt.upper()
                            if "L" in up: sess["lr_l"] += 1
                            if "R" in up: sess["lr_r"] += 1
                            sz = extract_size(up)
                            if sz is not None:
                                sess["persist_sizes"][tid] = sz
                                sess["size_counts"][sz] = sess["size_counts"].get(sz, 0) + 1
                        sess["persist_texts"][tid] = txt
                        sess["pending_ocr"].pop(tid, None)
            # 3) 每片信息
            pieces = []
            for info in infos:
                tid = int(info["track_id"])
                txt = sess["persist_texts"].get(tid, "") or None
                sz = extract_size(txt) if txt else None
                lr = None
                if txt:
                    up = txt.upper()
                    flags = []
                    if "L" in up: flags.append("L")
                    if "R" in up: flags.append("R")
                    if flags: lr = "/".join(flags)
                pieces.append({
                    "track_id": tid,
                    "piece_box": [round(float(v), 2) for v in info["box"]],
                    "conf": round(float(info["conf"]), 4),
                    "side": info["side"],  # above / below / None(死区)
                    "crossed": bool(info.get("crossed")),
                    "text": txt,
                    "shoe_size": sz,
                    "lr_flag": lr,
                })
            c = sess["counter"]
            stats = {
                "down": int(c.total_down), "up": int(c.total_up), "net": int(c.net),
                "lr_l": int(sess["lr_l"]), "lr_r": int(sess["lr_r"]),
                "pairs": int(min(sess["lr_l"], sess["lr_r"])),
                "single": int(abs(sess["lr_l"] - sess["lr_r"])),
                "sizes": {int(k): int(v) for k, v in sess["size_counts"].items()},
                "line_y": line_y, "line_ratio": line_ratio, "dead_zone": dead_zone,
            }
            # 扁平化摘要：界面显示的那几个字段（正向/反向/累计/L/R/成双/单只/鞋码）
            summary = {
                "正向前": stats["down"], "反向前": stats["up"], "累计": stats["net"],
                "L": stats["lr_l"], "R": stats["lr_r"],
                "成双": stats["pairs"], "单只": stats["single"],
                "鞋码": stats["sizes"],
                "计数线位置": stats["line_ratio"],
            }
            return _ok({
                "session_id": body.session_id,
                "image_shape": list(img.shape[:2]),
                "summary": summary,
                "stats": stats,
                "pieces": pieces,
            }, t0)
        finally:
            m["fabric"].conf, m["fabric"].iou = old_fab
            m["text"].conf, m["text"].iou = old_txt


@router.get("/count_sessions")
def api_algo_count_sessions():
    """列出当前内存里的 count_frame 会话（调试用）。"""
    return JSONResponse({
        "ok": True,
        "sessions": [
            {"session_id": sid,
             "down": s["counter"].total_down, "up": s["counter"].total_up,
             "lr_l": s["lr_l"], "lr_r": s["lr_r"],
             "sizes": {int(k): int(v) for k, v in s["size_counts"].items()}}
            for sid, s in _count_sessions.items()
        ],
    })


@router.post("/count_sessions/reset")
def api_algo_count_session_reset_all():
    """重置所有计数会话（回到零）。"""
    _count_sessions.clear()
    return JSONResponse({"ok": True, "reset_all": True})


@router.delete("/count_sessions/{sid}")
def api_algo_count_session_reset(sid: str):
    """释放/重置一个计数会话（比如视频结束了）。"""
    existed = _count_sessions.pop(sid, None)
    return JSONResponse({"ok": True, "reset": existed is not None})


# ============================================================
# ★ 主接口：一帧进，全部结果出（第三方软件只用这一个就行）
# ============================================================
@router.post("/process")
def api_algo_process(body: _Body):
    """
    算法唯一主接口：输入一帧图像，输出界面显示的全部字段。

    - 需要传 session_id：同一条视频流每帧都用同一个字符串，跨帧自动累计计数；
      换一条新视频/需要清零时，换一个新的 session_id（或调 DELETE /count_sessions/{sid}）。
    - ocr 默认 True（出 L/R/鞋码）；纯计数可传 ocr=false 更快。
    - 输出 data.summary 就是界面那排字段：
        正向前=累计正向过线片数  反向前=反向  累计=net
        L/R=左右脚  成双=成双对数  单只=落单只数  鞋码={尺码:数量}  计数线位置
    """
    body.ocr = True if body.ocr is None else body.ocr
    return api_algo_count_frame_json(body)
