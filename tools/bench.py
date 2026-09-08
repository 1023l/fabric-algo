# -*- coding: utf-8 -*-
"""TRT engine 测速（合并原 _trt11_bench.py / _trt11_rec_bench.py）。

用法:
    python tools/bench.py --model fabric [--engine X] [--video X] [--n 20]
    python tools/bench.py --model text   [--engine X] [--video X]
    python tools/bench.py --model rec    [--engine X] [--width 64]

说明:
    fabric: 真实视频帧整帧推理（ultralytics YOLO 加载 engine）
    text:   第一帧中央 1200x1200 裁剪模拟布片
    rec:    cupy 裸 engine 循环，主测宽度 + config.bench.rec_widths 对比宽度

需要 yolo-bench 环境（ultralytics/tensorrt/cupy）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cfg, ensure_trt_dlls, make_logger, models_dir, version  # noqa: E402

ensure_trt_dlls()


def _read_frames(video: str, n: int):
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f"无法打开视频: {video}"
    frames = []
    while len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    assert frames, "视频里一帧都没读到"
    return frames


def bench_fabric(engine: str, video: str, n: int, log) -> None:
    from ultralytics import YOLO  # noqa: E402

    frames = _read_frames(video, n)
    log(f"frames {len(frames)} shape {frames[0].shape}")
    m = YOLO(engine)
    m.predict(frames[0], imgsz=640, conf=0.4, verbose=False)  # warmup
    ts = []
    for fr in frames:
        t0 = time.perf_counter()
        m.predict(fr, imgsz=640, conf=0.4, verbose=False)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    log(f"fabric TRT detect: mean {sum(ts)/len(ts):.1f}ms median {ts[len(ts)//2]:.1f}ms (n={len(ts)})")


def bench_text(engine: str, video: str, log) -> None:
    from ultralytics import YOLO  # noqa: E402

    fr = _read_frames(video, 1)[0]
    h, w = fr.shape[:2]
    crop = fr[max(0, h // 2 - 600):h // 2 + 600, max(0, w // 2 - 600):w // 2 + 600]
    log(f"text crop shape {crop.shape}")
    m = YOLO(engine)
    m.predict(crop, imgsz=640, conf=0.05, verbose=False)  # warmup
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        m.predict(crop, imgsz=640, conf=0.05, verbose=False)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    log(f"text TRT detect: mean {sum(ts)/len(ts):.1f}ms median {ts[len(ts)//2]:.1f}ms (n={len(ts)})")


def bench_rec(engine: str, widths: list[int], log) -> None:
    import cupy as cp  # noqa: E402
    import numpy as np  # noqa: E402
    import tensorrt as trt  # noqa: E402

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine, "rb") as fp:
        rt = runtime.deserialize_cuda_engine(fp.read())
    ctx = rt.create_execution_context()
    in_name, out_name = rt.get_tensor_name(0), rt.get_tensor_name(1)
    host_out = np.empty(40000, dtype=np.float32)  # (1,seq,chars) 上限充足

    for nw in widths:
        shape = (1, 3, 48, nw)
        ctx.set_input_shape(in_name, shape)
        d_in = cp.empty(int(np.prod(shape)), dtype=np.float32)
        d_out = cp.empty(40000, dtype=np.float32)
        ctx.set_tensor_address(in_name, d_in.data.ptr)
        ctx.set_tensor_address(out_name, d_out.data.ptr)
        x = np.random.randn(*shape).astype(np.float32)
        d_in.set(x.reshape(-1))
        ctx.execute_v2([d_in.data.ptr, d_out.data.ptr])  # warmup
        ts = []
        for _ in range(50 if nw == widths[0] else 30):
            t0 = time.perf_counter()
            d_in.set(x.reshape(-1))
            ctx.execute_v2([d_in.data.ptr, d_out.data.ptr])
            d_out.get(out=host_out)
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        log(f"rec TRT(engine, w={nw}): mean {sum(ts)/len(ts):.1f}ms median {ts[len(ts)//2]:.1f}ms")


def main() -> None:
    p = argparse.ArgumentParser(description="TRT engine 测速")
    p.add_argument("--model", required=True, choices=["fabric", "text", "rec"])
    p.add_argument("--engine", default=None, help="默认 models/<kind>/<name><默认版本>.engine")
    p.add_argument("--video", default=None, help="fabric/text 用；默认 config.bench.video")
    p.add_argument("--n", type=int, default=None, help="fabric 帧数；默认 config.bench.n_frames")
    p.add_argument("--width", type=int, default=None, help="rec 主测宽度；默认 config.bench.rec_widths[0]")
    a = p.parse_args()
    log = make_logger(f"bench_{a.model}")

    b = cfg()["bench"]
    ver = version(a.model)
    engine = a.engine or str(models_dir(a.model) / f"{a.model}{ver}.engine")
    log(f"engine: {engine}")

    if a.model == "fabric":
        bench_fabric(engine, a.video or b["video"], a.n or b["n_frames"], log)
    elif a.model == "text":
        bench_text(engine, a.video or b["video"], log)
    else:
        widths = [a.width] + [w for w in b["rec_widths"] if w != a.width] if a.width else b["rec_widths"]
        bench_rec(engine, widths, log)
    log("done")


if __name__ == "__main__":
    main()
