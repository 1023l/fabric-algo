"""
业务端推理（ONNX 版）：不依赖 torch / paddle，只用 onnxruntime。

    fabric.onnx 检测布片 + 轻量 IoU 跟踪 + 可逆过线计数
    可选（--ocr）：text.onnx 检测文字区 → 垂直文字旋转 90° → rec.onnx 识别
        → 文字叠加视频（框内黄字黑底，跨帧显示）

性能要点（2026-08-27）：
    - rec 输入宽度可调 --rec-max-w（默认 192，鞋码短文字够用，越小越快）
    - rec 并行识别：每个 worker 独立 ORT session（--ocr-workers），
      避免共享同一 session 并发调用时的内部锁竞争（实测共享 session 并行反而更慢）
    - --provider 可选 cpu / cuda / openvino / trt
      - cuda：onnxruntime CUDA EP（需 onnxruntime-gpu）
      - trt：fabric/text 走 TensorRT engine（--fabric/--text 传 .engine 路径，
        需 tensorrt + cupy），rec 仍走 onnxruntime（CUDA EP）
    - 实测（4060）：fabric TRT 6.7ms / text TRT 5.6ms / rec CUDA 19ms，整帧 ~31ms

用法:
    # 只检测计数
    python infer_business.py --source runs/infer/Video_xxx.avi \
        --fabric models/fabric/fabric20260827V1.onnx

    # 检测计数 + 文字识别（CPU）
    python infer_business.py --source runs/infer/Video_xxx.avi --ocr \
        --fabric models/fabric/fabric20260827V1.onnx \
        --text models/text/text20260827V1.onnx \
        --rec models/rec/rec20260827V1.onnx

    # 全 GPU（fabric/text/rec 走 onnxruntime CUDA）
    python infer_business.py --source runs/infer/Video_xxx.avi --ocr --provider cuda \
        --fabric models/fabric/fabric20260827V1.onnx \
        --text models/text/text20260827V1.onnx \
        --rec models/rec/rec20260827V1.onnx

    # 极致性能：fabric/text 走 TRT engine + rec CUDA（整帧 ~31ms）
    python infer_business.py --source runs/infer/Video_xxx.avi --ocr --provider trt \
        --fabric models/fabric/fabric20260827V1.engine \
        --text models/text/text20260827V1.engine \
        --rec models/rec/rec20260827V1.onnx

依赖: onnxruntime(+onnxruntime-gpu/openvino 可选), opencv-python, numpy,
      trt 模式需 tensorrt + cupy
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import cv2

from utils import ROOT, find_chinese_font
from counter import run_count_video_generic
from onnx_engine import YoloOnnxDetector, TrtOnnxDetector, RecOnnxEngine, RecTrtEngine
from tracker import SimpleTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="fabric 检测计数 + 可选 OCR 识别（ONNX 版）")
    p.add_argument("--source", required=True, help="视频文件 / RTSP 流 / 摄像头编号(0)")
    p.add_argument("--fabric", required=True, help="fabric 检测模型 ONNX 路径")
    p.add_argument("--ocr", action="store_true", help="开启文字识别（需提供 --text 和 --rec）")
    p.add_argument("--text", type=str, default=None, help="text 文字区域检测模型 ONNX")
    p.add_argument("--rec", type=str, default=None, help="rec 识别模型 ONNX")
    p.add_argument("--conf", type=float, default=0.25, help="fabric 检测置信度")
    p.add_argument("--text-conf", type=float, default=None, help="text 检测置信度（默认同 --conf；文字模型通常更低）")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    p.add_argument("--imgsz", type=int, default=640, help="fabric/text 检测输入尺寸（当前 ONNX 固定 640）")
    p.add_argument("--rec-thresh", type=float, default=0.5, help="rec 识别最低置信度")
    p.add_argument("--rec-max-w", type=int, default=192, help="rec 输入最大宽度（鞋码类短文字 192 够用，越小越快；超长文字会截断）")
    p.add_argument("--ocr-workers", type=int, default=4, help="rec 并行识别 worker 数（每个 worker 独立 ORT session）")
    p.add_argument("--hide-text-boxes", action="store_true",
                   help="不绘制 text 检测框（默认绘制：横文青色 / 竖文品红 细框）")
    p.add_argument("--hide-ocr-text", action="store_true",
                   help="不叠加识别文字/鞋码在布片框上（HUD 左上角统计仍显示；开发分析时关掉此开关）")
    p.add_argument("--provider", type=str, default="cpu", choices=["cpu", "cuda", "openvino", "trt"],
                   help="推理后端：cpu / cuda（需 onnxruntime-gpu）/ openvino / trt（fabric·text 走 .engine，需 tensorrt+cupy）")
    p.add_argument("--fabric-provider", type=str, default=None, choices=["cpu", "cuda", "openvino", "trt"],
                   help="fabric 模型专用后端（默认跟随 --provider；轻量 yolo 实测 trt 6.7ms/cuda 13ms/cpu 35ms）")
    p.add_argument("--text-provider", type=str, default=None, choices=["cpu", "cuda", "openvino", "trt"],
                   help="text 模型专用后端（默认跟随 --provider；实测 trt 5.6ms/cuda 25ms/cpu 53ms）")
    p.add_argument("--rec-provider", type=str, default=None, choices=["cpu", "cuda", "openvino"],
                   help="rec 模型专用后端（默认跟随 --provider；LSTM/attention 建议 cuda）")
    p.add_argument("--pad", type=int, default=0, help="裁剪布片时框外扩像素")
    p.add_argument("--line-ratio", type=float, default=0.5, help="计数线位置（0~1）")
    p.add_argument("--dead-zone", type=int, default=15, help="死区像素宽度")
    p.add_argument("--max-track-age", type=int, default=60, help="track 过期清理帧数")
    p.add_argument("--track-iou", type=float, default=0.3, help="跟踪匹配 IoU 阈值")
    p.add_argument("--output", type=str, default=None, help="输出视频路径")
    p.add_argument("--show", action="store_true", help="实时弹窗预览（按 q 退出）")
    return p.parse_args()


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (ROOT / pp).resolve()


def resolve_providers(name: str) -> list[str]:
    """按名称解析 ORT providers（回退 CPU）。trt 模式下 rec(ONNX) 走 CUDA EP。"""
    if name == "cuda" or name == "trt":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if name == "openvino":
        return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_detector(path: Path, imgsz: int, conf: float, iou: float, provider: str):
    """按模型文件与后端加载检测器：
    - .engine → TrtOnnxDetector（TensorRT，需 tensorrt+cupy）
    - 其余 → YoloOnnxDetector（onnxruntime，按 provider 选 EP）
    """
    if provider == "trt":
        if not str(path).lower().endswith(".engine"):
            raise SystemExit(f"TRT 模式需要 .engine 模型文件: {path}")
        return TrtOnnxDetector(path, imgsz=imgsz, conf=conf, iou=iou)
    if str(path).lower().endswith(".engine"):
        return TrtOnnxDetector(path, imgsz=imgsz, conf=conf, iou=iou)
    return YoloOnnxDetector(path, imgsz=imgsz, conf=conf, iou=iou,
                            providers=resolve_providers(provider))


def _run_chunk(chunk, engine):
    """某 worker 独立 session 串行识别自己分到的一批文字区。"""
    return [(tid, by1, bx1, engine.recognize(crop)) for (tid, by1, bx1, crop) in chunk]


def make_ocr_callback(text_det, rec_engines, pad: int, rec_thresh: float, timing: dict,
                      draw_boxes: bool = True):
    """构造 OCR 回调：识别 counter 传入的"过线待识别"布片（窗口内重试），文字结果跨帧显示。

    每个 worker 持有独立 ORT session（共享 session 并发会锁竞争，实测反而慢），
    jobs 按 stride 分片到各 worker 并行 rec 识别。耗时分三段统计：
    text_s/text_n（文字区检测，按片）、rec_s/rec_n（文字识别，按张）。
    draw_boxes=True 时把 text 检测框画回全帧：横文(text_h)青色、竖文(text_v)品红，
    识别成功与否都画，方便检查文字区域检出情况。
    """
    n_workers = len(rec_engines)

    def cb(frame, infos, frame_idx):
        texts: dict[int, str] = {}
        jobs = []  # (tid, by1, bx1, crop)
        for info in infos:
            tid = info["track_id"]
            x1, y1, x2, y2 = map(int, info["box"])
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = x2 + pad, y2 + pad
            piece = frame[y1:y2, x1:x2]
            if piece.size == 0:
                continue
            # 1) text 模型检测文字区域（每片一次）
            t0 = time.perf_counter()
            dets = text_det.detect(piece)
            timing["text_n"] += 1
            timing["text_s"] += time.perf_counter() - t0
            for d in dets:
                bx1, by1, bx2, by2 = map(int, d["box"])
                cls_name = "text_v" if d["cls"] == 1 else "text_h"
                if draw_boxes:
                    # text 检测框画回全帧坐标：横文青色 / 竖文品红 加粗框（不含文字，干净框线）
                    gx1, gy1, gx2, gy2 = x1 + bx1, y1 + by1, x1 + bx2, y1 + by2
                    box_color = (255, 0, 255) if cls_name == "text_v" else (255, 200, 0)
                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), box_color, 3)
                crop = piece[by1:by2, bx1:bx2]
                if crop.size == 0:
                    continue
                # 2) 垂直文字区域旋转 90° 再识别（与标注/训练约定一致）
                if cls_name == "text_v":
                    crop = np.rot90(crop, k=3)  # 顺时针 90°：让竖排文字转为水平
                jobs.append((tid, by1, bx1, crop))
        if not jobs:
            return texts

        # 3) rec 识别：各 worker 独立 session，jobs 按 stride 分片并行
        t0 = time.perf_counter()
        results = []
        if n_workers <= 1 or len(jobs) <= 1:
            results = _run_chunk(jobs, rec_engines[0])
        else:
            chunks = [jobs[i::n_workers] for i in range(n_workers)]
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_run_chunk, chunks[i], rec_engines[i])
                           for i in range(n_workers)]
                for f in futures:
                    results.extend(f.result())
        timing["rec_n"] += len(jobs)
        timing["rec_s"] += time.perf_counter() - t0

        # 按 (y, x) 排序拼接：同一布片多个文字区 → 空格连接
        per_tid: dict[int, list] = {}
        for tid, by1, bx1, (txt, sc) in results:
            if txt and sc >= rec_thresh:
                per_tid.setdefault(tid, []).append((by1, bx1, txt))
        for tid, items in per_tid.items():
            items.sort(key=lambda it: (it[0], it[1]))
            texts[tid] = " ".join(it[2] for it in items)
        return texts

    return cb


def main() -> None:
    args = parse_args()
    if args.ocr and (not args.text or not args.rec):
        raise SystemExit("--ocr 需要同时提供 --text 和 --rec（ONNX 路径）")

    rec_providers = resolve_providers(args.rec_provider or args.provider)
    fabric_onnx = _resolve_path(args.fabric)
    if not fabric_onnx.is_file():
        raise SystemExit(f"fabric 模型不存在: {fabric_onnx}")
    fabric_provider = args.fabric_provider or args.provider

    fabric_det = load_detector(fabric_onnx, args.imgsz, args.conf, args.iou, fabric_provider)
    tracker = SimpleTracker(max_age=args.max_track_age, iou_thresh=args.track_iou)
    timing = {"fabric_s": 0.0, "fabric_n": 0, "text_s": 0.0, "text_n": 0,
              "rec_s": 0.0, "rec_n": 0}
    print(f"[infer] fabric ONNX: {fabric_onnx} (provider={args.fabric_provider or args.provider})")

    def detect_fn(frame):
        t0 = time.perf_counter()
        dets = fabric_det.detect(frame)
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

    text_det = None
    rec_engines = []
    if args.ocr:
        text_onnx = _resolve_path(args.text)
        rec_onnx = _resolve_path(args.rec)
        if not text_onnx.is_file() or not rec_onnx.is_file():
            raise SystemExit(f"text/rec ONNX 不存在: {text_onnx} / {rec_onnx}")
        text_conf = args.text_conf if args.text_conf is not None else args.conf
        text_provider = args.text_provider or args.provider
        text_det = load_detector(text_onnx, args.imgsz, text_conf, args.iou, text_provider)
        # 每个 worker 独立 session/context（避免共享并发锁竞争）；rec 支持 .engine（TRT）
        if str(rec_onnx).lower().endswith(".engine"):
            rec_engines = [RecTrtEngine.load(rec_onnx, score_thresh=args.rec_thresh,
                                             max_width=args.rec_max_w)
                           for _ in range(args.ocr_workers)]
        else:
            rec_engines = [RecOnnxEngine.load(rec_onnx, score_thresh=args.rec_thresh,
                                              max_width=args.rec_max_w, providers=rec_providers)
                           for _ in range(args.ocr_workers)]
        print(f"[ocr] text ONNX: {text_onnx} (provider={args.text_provider or args.provider})")
        print(f"[ocr] rec  ONNX: {rec_onnx} (provider={args.rec_provider or args.provider}, "
              f"charset {len(rec_engines[0].chars)} 字符, "
              f"max_w {rec_engines[0].max_width}, workers {len(rec_engines)})")

    ocr_cb = make_ocr_callback(text_det, rec_engines, args.pad, args.rec_thresh, timing,
                               draw_boxes=not args.hide_text_boxes) if args.ocr else None
    font_path = find_chinese_font()

    result = run_count_video_generic(
        detect_fn, args.source, str(fabric_onnx),
        line_ratio=args.line_ratio, dead_zone=args.dead_zone,
        max_track_age=args.max_track_age, output=args.output,
        font_path=font_path, show=args.show, ocr_callback=ocr_cb,
        show_ocr_text=not args.hide_ocr_text,
    )

    print(f"[infer] 计数结果: down={result['down']} up={result['up']} net={result['net']}"
          f" | L={result.get('lr_l', 0)} R={result.get('lr_r', 0)}")
    if result.get("sizes"):
        top = sorted(result["sizes"].items(), key=lambda kv: -kv[1])
        print("[infer] 鞋码统计: " + ", ".join(f"{k}×{v}" for k, v in top))
    if timing["fabric_n"]:
        print(f"[timing] fabric 检测: {timing['fabric_s'] / timing['fabric_n'] * 1000:.1f} ms/帧 × {timing['fabric_n']} 帧")
    if timing["text_n"]:
        print(f"[timing] text 检测: {timing['text_s'] / timing['text_n'] * 1000:.1f} ms/片 × {timing['text_n']} 片")
    if timing["rec_n"]:
        print(f"[timing] rec 识别: {timing['rec_s'] / timing['rec_n'] * 1000:.1f} ms/张 × {timing['rec_n']} 张"
              f"（并行 {args.ocr_workers} worker）")


if __name__ == "__main__":
    main()
