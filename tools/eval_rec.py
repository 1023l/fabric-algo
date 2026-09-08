# -*- coding: utf-8 -*-
"""rec 模型验证集逐样本评估（合并原 _eval_rec_samples.py / _eval_rec_onnx.py）。

用法:
    python tools/eval_rec.py --backend onnx  [--version X] [--val X] [--dict X]
    python tools/eval_rec.py --backend paddle [--version X] [--arch server|mobile]

说明:
    onnx   : 用 fabric-algo onnx_engine.RecOnnxEngine（yolo-bench 环境）
    paddle : 用 PaddleOCR TextRecognizer 直接加载 inference 模型（paddle-ocr 环境，
             缺依赖时自动用 config.envs.paddle_ocr 重启）
    val/dict 默认取 config.rec（train-center/data/rec_text/）
    输出逐样本 预测 vs 标签 + 总准确率。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cfg, fa_root, models_dir, relaunch_if_needed, tc_root, version  # noqa: E402


def load_val(val_path: Path, data_root: Path) -> list[tuple[str, str]]:
    items = []
    for line in val_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        imgp, label = line.split("\t")
        full = Path(imgp.strip())
        if not full.is_absolute():
            full = data_root / full
        items.append((str(full), label.strip()))
    return items


def eval_onnx(ver: str, val: Path, dict_path: Path) -> None:
    import cv2  # noqa: E402
    from onnx_engine import RecOnnxEngine  # noqa: E402

    chars = [l for l in dict_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[dict] {len(chars)} chars")
    onnx_path = models_dir("rec") / f"rec{ver}_onnx.onnx"
    eng = RecOnnxEngine(str(onnx_path), chars, score_thresh=0.0)

    items = load_val(val, val.parent)
    total = correct = 0
    for full, label in items:
        img = cv2.imread(full)
        if img is None:
            print(f"[READFAIL] {full}")
            continue
        pred, conf = eng.recognize(img)
        ok = pred == label
        total += 1
        correct += int(ok)
        print(f"[{'OK ' if ok else 'FAIL'}] pred={pred!r:<20} conf={conf:.3f} label={label!r}")
    print(f"\n== {correct}/{total} = {correct / max(total, 1):.3f}")


def eval_paddle(ver: str, val: Path) -> None:
    relaunch_if_needed("paddle", "paddle_ocr")
    sys.path.insert(0, str(tc_root() / "PaddleOCR"))
    import cv2  # noqa: E402
    from tools.infer.predict_rec import TextRecognizer  # noqa: E402
    from tools.infer.utility import parse_args  # noqa: E402

    model_dir = tc_root() / "models" / "ocr" / f"rec{ver}" / "inference"
    args = parse_args()
    args.rec_model_dir = str(model_dir)
    args.use_gpu = True
    args.gpu_mem = 500
    args.rec_char_dict_path = str(tc_root() / cfg()["rec"]["dict"])
    rec = TextRecognizer(args)

    items = load_val(val, val.parent)
    total = correct = 0
    for full, label in items:
        img = cv2.imread(full)
        if img is None:
            print(f"[READFAIL] {full}")
            continue
        res = rec([img])[0]
        pred = res[0][0] if res and res[0] else ""
        conf = res[0][1] if res and res[0] else 0
        ok = pred == label
        total += 1
        correct += int(ok)
        print(f"[{'OK ' if ok else 'FAIL'}] pred={pred!r:<20} conf={conf:.3f} label={label!r}")
    print(f"\n== {correct}/{total} = {correct / max(total, 1):.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="rec 验证集逐样本评估")
    p.add_argument("--backend", default="onnx", choices=["onnx", "paddle"])
    p.add_argument("--version", default=None, help="默认取 config.versions.rec")
    p.add_argument("--val", default=None, help="val.txt 路径（默认 config.rec.val）")
    p.add_argument("--dict", default=None, help="dict.txt 路径（默认 config.rec.dict）")
    a = p.parse_args()

    ver = a.version or version("rec")
    val = Path(a.val) if a.val else tc_root() / cfg()["rec"]["val"]
    dict_path = Path(a.dict) if a.dict else tc_root() / cfg()["rec"]["dict"]
    print(f"[eval] backend={a.backend} version={ver} val={val}")

    if a.backend == "onnx":
        eval_onnx(ver, val, dict_path)
    else:
        eval_paddle(ver, val)


if __name__ == "__main__":
    main()
