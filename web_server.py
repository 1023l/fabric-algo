# -*- coding: utf-8 -*-
"""通用视觉检测流水线 Web 服务（FastAPI）。

定位：目标检测 + OCR 识别 + 跟踪计数的通用推理端（当前参考实现：纺织布片计数/鞋码识别）。

功能（先以视频文件测试，后续接视频流/SDK）：
- 上传视频 -> 浏览器实时显示检测画面（MJPEG 流）
- 计数线可拖拽实时调整（拖动即生效）
- 开始 / 停止 / 清零 / 导出（导出 = 从头完整跑一遍并输出 mp4，与 CLI 一致）

用法:
    python web_server.py                 # 默认 127.0.0.1:8001
    python web_server.py --port 8001

前端: http://127.0.0.1:8001/
依赖: fastapi uvicorn opencv-python numpy tensorrt cupy（yolo-bench 环境）
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from utils import ROOT, find_chinese_font
from counter import run_count_video_generic
from onnx_engine import RecTrtEngine
from infer_business import make_ocr_callback
from tracker import SimpleTracker
from algo_routes import router as algo_router
from model_admin import router as model_admin_router
from model_admin import engine_paths

UPLOAD_DIR = ROOT / "runs" / "uploads"
EXPORT_DIR = ROOT / "runs" / "infer"
PUSH_MAX_W = 1080      # MJPEG 推送画面最大宽度（缩放省带宽）
OCR_WORKERS = 4
REC_THRESH = 0.35      # 识别阈值（与 v22 实测一致）
REC_MAX_W = 192
FABRIC_CONF = 0.4
TEXT_CONF = 0.25
IOU = 0.45

app = FastAPI(title="fabric-algo 实时检测")
app.include_router(algo_router)
app.include_router(model_admin_router)


@app.middleware("http")
async def _hard_timeout(request, call_next):
    """全局 8s 硬超时：防止某个 API 阻塞导致整页无响应。
    模型包上传（数百 MB）与转换进度轮询放宽到 10 分钟。"""
    import asyncio
    timeout = 600 if request.url.path.startswith("/api/models/") else 8
    try:
        return await asyncio.wait_for(call_next(request), timeout=timeout)
    except asyncio.TimeoutError:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=504,
                            content={"ok": False, "msg": "后端处理超时，请刷新重试"})


# ---- 全局状态（线程安全：RLock 允许同线程重入，避免锁内调助手函数死锁） ----
state = {
    "lock": threading.RLock(),
    "video": None,            # 当前视频绝对路径
    "video_name": None,
    "video_info": None,       # {fps, frames, width, height, size}
    "line_ratio": 0.5,        # 计数线位置 0~1（可拖拽实时生效）
    "running": False,         # 检测线程是否在跑
    "stop_event": None,       # 当前检测线程的停止事件
    "latest_frame": None,     # 最新处理帧 JPEG bytes（MJPEG 流读它）
    "latest_stats": None,     # 最新统计 dict
    "last_result": None,      # 上一次完整运行的结果
    "exporting": False,       # 是否正在导出
    "export_path": None,      # 最近一次导出的文件路径
    "session_cfg": {},        # 暂停/续播上下文（counter 就地更新；空 dict=从头开始）
    "mode": "test",           # 显示模式：test=测试(画文字框+识别文字) / prod=生产(纯净，无文字框/文字)
}
# 模型 lazy 加载全局单例
_models: dict = {"fabric": None, "text": None, "rec": None}


def _publish_jpeg(frame_bgr: np.ndarray) -> None:
    """缩放 + JPEG 编码后写入 state（供 MJPEG 流读取）。"""
    h, w = frame_bgr.shape[:2]
    if w > PUSH_MAX_W:
        scale = PUSH_MAX_W / w
        frame_bgr = cv2.resize(frame_bgr, (PUSH_MAX_W, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return
    with state["lock"]:
        state["latest_frame"] = buf.tobytes()


def build_model_set(paths: dict) -> dict:
    """按给定 engine 路径加载一整套模型（首次加载 / 模型管理热替换共用）。"""
    from onnx_engine import TrtOnnxDetector
    return {
        "fabric": TrtOnnxDetector(paths["fabric"], imgsz=640,
                                  conf=FABRIC_CONF, iou=IOU),
        "text": TrtOnnxDetector(paths["text"], imgsz=640,
                                conf=TEXT_CONF, iou=IOU),
        "rec": [RecTrtEngine.load(paths["rec"],
                                  score_thresh=REC_THRESH,
                                  max_width=REC_MAX_W)
                for _ in range(OCR_WORKERS)],
    }


def get_models() -> dict:
    """按需加载 fabric/text/rec 模型（TRT engine，只加载一次；版本读 models/current.json）。"""
    if _models["fabric"] is None:
        _models.update(build_model_set(engine_paths()))
    return _models


def build_pipeline(draw_text_boxes: bool = True, next_track_id: int = 1):
    """构造检测函数与 OCR 回调（每次运行新建 tracker/timing）。"""
    m = get_models()
    tracker = SimpleTracker(max_age=60, iou_thresh=0.3, next_id=next_track_id)
    timing = {"fabric_s": 0.0, "fabric_n": 0, "text_s": 0.0, "text_n": 0,
              "rec_s": 0.0, "rec_n": 0}

    def detect_fn(frame):
        t0 = time.perf_counter()
        dets = m["fabric"].detect(frame)
        timing["fabric_n"] += 1
        timing["fabric_s"] += time.perf_counter() - t0
        if not dets:
            return None
        tracked = tracker.update(dets)
        if not tracked:
            return None
        boxes = np.array([t["box"] for t in tracked], dtype=np.float32)
        ids = np.array([t["track_id"] for t in tracked], dtype=np.int64)
        clss = np.array([t["cls"] for t in tracked], dtype=np.int64)
        confs = np.array([t["conf"] for t in tracked], dtype=np.float32)
        return boxes, ids, clss, confs

    ocr_cb = make_ocr_callback(m["text"], m["rec"], 0, REC_THRESH, timing,
                               draw_boxes=draw_text_boxes)
    return detect_fn, ocr_cb, timing


def _resume_next_track_id(resume: dict | None) -> int:
    """续播时新 tracker 的起始 id：取旧 persist 键的最大值 +1，避免 id 冲突。"""
    if not resume or "counter" not in resume:
        return 1
    tids = set(resume.get("persist_texts", {}))
    tids.update(resume.get("persist_sizes", {}))
    tids.update(resume.get("pending_ocr", {}))
    return max(tids, default=0) + 1


def _display_flags() -> tuple:
    """按当前模式返回 (draw_text_boxes, show_ocr_text)：
    test=测试（画文字框+识别文字）；prod=生产（纯净，无文字框/文字）。"""
    with state["lock"]:
        return (True, True) if state["mode"] == "test" else (False, False)


def _run_engine(video: str, output: str | None, save_video: bool,
                line_ref, draw_text_boxes: bool, show_ocr_text: bool,
                state_key: str = "running", resume: dict | None = None) -> dict:
    """后台线程统一入口：实时预览（不写视频）或导出（写视频）。
    resume: 暂停/续播上下文。空 dict=从头（counter 就地初始化并写回）；带 counter=续播。"""
    detect_fn, ocr_cb, timing = build_pipeline(draw_text_boxes,
                                               _resume_next_track_id(resume))
    stop_event = threading.Event()
    with state["lock"]:
        state[state_key] = True
        state["stop_event"] = stop_event
        state["exporting"] = state_key == "exporting"
        init_line_ratio = float(state["line_ratio"])

    def frame_cb(frame, frame_idx, stats):
        _publish_jpeg(frame)
        with state["lock"]:
            state["latest_stats"] = stats

    result = None
    try:
        result = run_count_video_generic(
            detect_fn, video, str(engine_paths()["fabric"]),
            line_ratio=init_line_ratio, dead_zone=15, max_track_age=60,
            output=output, font_path=find_chinese_font(), show=False,
            ocr_callback=ocr_cb, show_ocr_text=show_ocr_text,
            save_video=save_video, frame_callback=frame_cb,
            stop_event=stop_event, line_ratio_ref=line_ref,
            resume=resume,
        )
    finally:
        with state["lock"]:
            state[state_key] = False
            state["exporting"] = False
            state["last_result"] = result
            if output:
                state["export_path"] = output
    return result


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    """健康检查（接口文档承诺给第三方软件探活用）。"""
    return {"status": "ok"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """上传视频：保存到 runs/uploads，解析视频信息，抽第一帧预览。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "video.mp4").name
    dest = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    data = await file.read()
    dest.write_bytes(data)

    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        raise HTTPException(400, "无法打开上传的视频（格式不支持？）")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()

    info = {"name": safe_name, "path": str(dest), "size": len(data),
            "width": width, "height": height, "fps": round(fps, 2),
            "frames": frames}
    with state["lock"]:
        state["video"] = str(dest)
        state["video_name"] = safe_name
        state["video_info"] = info
        # 新视频：清掉旧画面/统计/暂停进度，页面回到"原始视频"状态
        state["latest_stats"] = None
        state["last_result"] = None
        state["latest_frame"] = None
        state["session_cfg"] = {}
    if ok:
        _publish_jpeg(first)  # 显示第一帧原图（点开始后才是检测画面）
    return info


@app.get("/api/status")
def status():
    with state["lock"]:
        return {
            "video": state["video_name"],
            "video_info": state["video_info"],
            "line_ratio": state["line_ratio"],
            "running": state["running"],
            "exporting": state["exporting"],
            "has_session": "counter" in state["session_cfg"],
            "mode": state["mode"],
            "export_path": Path(state["export_path"]).name if state["export_path"] else None,
            "stats": state["latest_stats"],
            "result": state["last_result"],
        }


@app.post("/api/line")
async def set_line(req: dict):
    """设置计数线位置（0~1），检测循环每帧读取，实时生效。"""
    ratio = float(req.get("ratio", 0.5))
    ratio = max(0.0, min(1.0, ratio))
    with state["lock"]:
        state["line_ratio"] = ratio
    return {"line_ratio": ratio}


@app.post("/api/mode")
async def set_mode(req: dict):
    """切换显示模式：test=测试（实时/导出都画文字框+识别文字）；prod=生产（都纯净）。"""
    mode = str(req.get("mode", "test")).lower()
    if mode not in ("test", "prod"):
        return {"ok": False, "msg": "mode 只支持 test / prod"}
    with state["lock"]:
        state["mode"] = mode
    return {"ok": True, "mode": mode}


@app.post("/api/start")
def start():
    """开始实时检测：从头播放视频（含 fabric 检测 + OCR + 叠加显示，不写视频）。
    显示内容按当前模式：测试=画文字框+识别文字；生产=纯净（无文字框/文字）。"""
    with state["lock"]:
        if state["running"]:
            return {"ok": False, "msg": "已在运行"}
        video = state["video"]
        draw_tb, show_ocr = _display_flags()
    if not video:
        return {"ok": False, "msg": "请先上传视频"}

    with state["lock"]:
        state["session_cfg"] = {}  # 从头开始：清空续播上下文（空 dict=counter 重新初始化）
    cfg = state["session_cfg"]
    t = threading.Thread(target=_run_engine,
                         args=(video, None, False,
                               SimpleNamespace(value=state["line_ratio"]),
                               draw_tb, show_ocr),
                         kwargs={"resume": cfg},
                         daemon=True)
    t.start()
    return {"ok": True}


@app.post("/api/continue")
def cont():
    """继续：从上次暂停位置续播（计数/文字/帧位置全部恢复）。"""
    with state["lock"]:
        if state["running"]:
            return {"ok": False, "msg": "已在运行"}
        video = state["video"]
        cfg = state["session_cfg"]
        draw_tb, show_ocr = _display_flags()
    if not video:
        return {"ok": False, "msg": "请先上传视频"}
    if "counter" not in cfg:
        return {"ok": False, "msg": "还没有暂停进度，请先点「开始」"}
    t = threading.Thread(target=_run_engine,
                         args=(video, None, False,
                               SimpleNamespace(value=state["line_ratio"]),
                               draw_tb, show_ocr),
                         kwargs={"resume": cfg},
                         daemon=True)
    t.start()
    return {"ok": True}


@app.post("/api/stop")
def stop():
    """停止实时检测（暂停：当前帧处理完即停，进度保留，可点「继续」续播）。"""
    with state["lock"]:
        ev = state["stop_event"]
    if ev is not None:
        ev.set()
    return {"ok": True}


@app.post("/api/export")
def export():
    """导出：从头完整跑一遍当前视频，输出 mp4。
    显示内容按当前模式：测试=带文字框/识别文字；生产=纯净版。"""
    with state["lock"]:
        if state["exporting"]:
            return {"ok": False, "msg": "正在导出"}
        if state["running"]:
            return {"ok": False, "msg": "请先停止实时检测再导出"}
        video = state["video"]
        draw_tb, show_ocr = _display_flags()
    if not video:
        return {"ok": False, "msg": "请先上传视频"}
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = str(EXPORT_DIR / f"{Path(video).stem}_export.mp4")
    t = threading.Thread(target=_run_engine,
                         args=(video, out, True,
                               SimpleNamespace(value=state["line_ratio"]),
                               draw_tb, show_ocr, "exporting"),
                         daemon=True)
    t.start()
    return {"ok": True, "path": out}


@app.get("/api/export_download")
def export_download():
    with state["lock"]:
        p = state["export_path"]
    if not p or not Path(p).is_file():
        raise HTTPException(404, "暂无导出文件")
    return FileResponse(p, filename=Path(p).name, media_type="video/mp4")


@app.get("/video_feed")
def video_feed():
    """MJPEG 流：持续推送 state['latest_frame']。"""
    def gen():
        while True:
            with state["lock"]:
                frame = state["latest_frame"]
            if frame is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


def main() -> None:
    parser = argparse.ArgumentParser(description="通用视觉检测流水线 Web 服务（fabric-algo）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    print(f"视觉检测流水线 Web 服务（fabric-algo）: http://{args.host}:{args.port}/")
    _v = engine_paths()
    print(f"模型: fabric={_v['fabric'].name} "
          f"text={_v['text'].name} rec={_v['rec'].name}")
    print("[init] 预热 fabric/text/rec 模型（TRT engine 首次加载较慢，约 20-30s）...", flush=True)
    t0 = time.time()
    get_models()
    print(f"[init] 模型预热完成 ({time.time()-t0:.1f}s)，服务就绪", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
