# -*- coding: utf-8 -*-
"""模型验证（合并原 _verify_engine.py / _verify_det_onnx.py / _cmp_text_onnx.py）。

用法:
    python tools/verify.py engine [--version X]     # 三个 TRT engine 加载+推理（真实视频帧）
    python tools/verify.py onnx  --model fabric|text|all [--version X]   # ONNX 随机输入结构校验
    python tools/verify.py graph --model text [--version X]              # ONNX 图结构 dump（排查构建问题）

engine 模式需要 yolo-bench 环境（GPU）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # fabric-algo 根（onnx_engine）
from _common import cfg, ensure_trt_dlls, models_dir, version  # noqa: E402


def _first_frame():
    video = cfg()["bench"]["video"]
    import cv2
    cap = cv2.VideoCapture(video)
    ok, fr = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    if ok:
        print(f"[frame] {video} shape {fr.shape}")
        return fr
    print(f"[frame] 视频不可用({video})，用全零帧替代")
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def verify_engine(vers: dict) -> None:
    ensure_trt_dlls()
    from onnx_engine import RecTrtEngine, TrtOnnxDetector  # noqa: E402

    fr = _first_frame()
    t0 = time.perf_counter()
    d1 = TrtOnnxDetector(str(models_dir("fabric") / f"fabric{vers['fabric']}.engine"))
    load1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    r1 = d1.detect(fr)
    print(f"[fabric] load {load1:.1f}s | detect {(time.perf_counter()-t0)*1000:.1f}ms -> {len(r1)} boxes")

    h, w = fr.shape[:2]
    crop = fr[max(0, h // 2 - 600):h // 2 + 600, max(0, w // 2 - 600):w // 2 + 600]
    t0 = time.perf_counter()
    d2 = TrtOnnxDetector(str(models_dir("text") / f"text{vers['text']}.engine"))
    load2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    r2 = d2.detect(crop)
    print(f"[text]   load {load2:.1f}s | detect {(time.perf_counter()-t0)*1000:.1f}ms -> {len(r2)} boxes")

    t0 = time.perf_counter()
    rec = RecTrtEngine.load(str(models_dir("rec") / f"rec{vers['rec']}.engine"), max_width=192)
    load3 = time.perf_counter() - t0
    if r2:
        x1, y1, x2, y2 = map(int, r2[0]["box"])
        tcrop = crop[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
        if tcrop.size == 0:
            tcrop = crop[100:400, 100:400]
    else:
        tcrop = crop[100:400, 100:400]
    t0 = time.perf_counter()
    txt, score = rec.recognize(tcrop)
    print(f"[rec]    load {load3:.1f}s | recognize {(time.perf_counter()-t0)*1000:.1f}ms -> '{txt}' score {score:.2f}")
    print("verify engine done")


def verify_onnx(model: str, ver_by_name: dict) -> None:
    import onnxruntime as ort

    names = ["fabric", "text"] if model == "all" else [model]
    for name in names:
        path = models_dir(name) / f"{name}{ver_by_name[name]}.onnx"
        try:
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inp, out_meta = sess.get_inputs()[0], sess.get_outputs()[0]
            x = np.zeros((1, 3, 640, 640), dtype=np.float32)
            res = sess.run(None, {inp.name: x})[0]
            print(f"[OK] {name}: input {inp.shape} -> output {res.shape}, dtype {res.dtype}, "
                  f"fin={np.isfinite(res).all()}")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name}: {e}")
    print("verify onnx done")


def verify_graph(model: str, ver: str) -> None:
    import onnx

    p = models_dir(model) / f"{model}{ver}.onnx"
    m = onnx.load(str(p))
    ops: dict = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"[{model}{ver}] ir={m.ir_version} opset={[o.version for o in m.opset_import]} "
          f"nodes={len(m.graph.node)} inits={len(m.graph.initializer)}")
    print("   top ops:", dict(sorted(ops.items(), key=lambda x: -x[1])[:12]))
    print("   outputs:", [o.name + str([d.dim_value for d in o.type.tensor_type.shape.dim])
                        for o in m.graph.output])


def main() -> None:
    p = argparse.ArgumentParser(description="engine / onnx 验证")
    p.add_argument("mode", choices=["engine", "onnx", "graph"])
    p.add_argument("--model", default="all", choices=["fabric", "text", "rec", "all"])
    p.add_argument("--version", default=None, help="单个模型版本；不传则各用 config.versions 默认")
    a = p.parse_args()

    vers = {k: a.version or version(k) for k in ("fabric", "text", "rec")}
    if a.mode == "engine":
        verify_engine(vers)
    elif a.mode == "onnx":
        verify_onnx(a.model, vers)
    else:
        assert a.model in ("fabric", "text", "rec"), "graph 模式需要指定 --model"
        verify_graph(a.model, vers[a.model])


if __name__ == "__main__":
    main()
