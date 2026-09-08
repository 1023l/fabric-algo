# -*- coding: utf-8 -*-
"""抽帧布片裁剪（原 _crop_fabric_pieces.py 入库版）：text 文字框标注的数据源。

对视频逐帧跑 fabric engine，把每个布片（外扩 pad）裁成独立图片。
text 检测模型的输入是"布片裁剪图"，所以标注数据必须这样造。

用法:
    python tools/crop_pieces.py --video <视频> [--engine X] --out <目录> [--conf 0.4] [--pad 15]

输出: piece_f<帧号>_n<序号>.jpg（JPG 95）
需要 yolo-bench 环境（engine 推理）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import models_dir, version  # noqa: E402
from onnx_engine import TrtOnnxDetector  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="抽帧布片裁剪（text 标注数据源）")
    p.add_argument("--video", required=True, help="输入视频")
    p.add_argument("--engine", default=None, help="fabric engine（默认当前 config 版本）")
    p.add_argument("--out", required=True, help="裁剪输出目录（自动创建）")
    p.add_argument("--conf", type=float, default=0.4, help="fabric 检测置信度（默认与业务一致 0.4）")
    p.add_argument("--pad", type=int, default=15, help="裁剪外扩像素")
    a = p.parse_args()

    engine = a.engine or str(models_dir("fabric") / f"fabric{version('fabric')}.engine")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    det = TrtOnnxDetector(engine, conf=a.conf)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print(f"[crop] 无法打开视频: {a.video}")
        sys.exit(1)

    idx = saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for j, d in enumerate(det.detect(frame)):
            if d["conf"] < a.conf:
                continue
            x1, y1, x2, y2 = map(int, d["box"])
            x1, y1 = max(0, x1 - a.pad), max(0, y1 - a.pad)
            x2 = min(frame.shape[1], x2 + a.pad)
            y2 = min(frame.shape[0], y2 + a.pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            if cv2.imwrite(str(out / f"piece_f{idx:06d}_n{j:02d}.jpg"), crop,
                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
                saved += 1
                if saved % 50 == 0:
                    print(f"[crop] 已裁剪 {saved} 张 ...")
        idx += 1
    cap.release()
    print(f"[crop] 完成: {saved} 张布片裁剪 -> {out}")


if __name__ == "__main__":
    main()
