# -*- coding: utf-8 -*-
"""模型导出 ONNX（统一入口，合并原 _export_det_onnx.py / _export_rec_onnx.py）。

用法:
    python tools/export_onnx.py det --model fabric|text|all [--version 20260828V2]
        YOLO .pt -> 静态 640 + 端到端 NMS ONNX (opset 17, 输出 [1,300,6])
        需要 yolo-bench 环境（缺依赖时自动用 config.envs.yolo_bench 重启）

    python tools/export_onnx.py rec [--version 20260828V3] [--arch server|mobile]
        PaddleOCR best.pdparams -> 动态宽 ONNX（1x3x48x-1, opset 12）
        需要 paddle-ocr 环境（缺依赖时自动用 config.envs.paddle_ocr 重启）

产物: fabric-algo/models/{name}/{name}{version}.onnx（rec 为 {name}{version}_onnx.onnx）
版本化命名，不覆盖旧版本。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cfg, fa_root, models_dir, relaunch_if_needed, tc_root, version  # noqa: E402


def export_det(model: str, ver: str, pt: str | None = None) -> None:
    """YOLO .pt -> ONNX（需 torch/ultralytics）。pt 直接指定源文件时忽略 train-center 目录。"""
    relaunch_if_needed("ultralytics", "yolo_bench")
    from ultralytics import YOLO  # noqa: E402

    d = cfg()["det"]
    pt_dir = tc_root() / "models" / "det"
    if pt:
        names = [model]
    else:
        names = ["fabric", "text"] if model == "all" else [model]
    for name in names:
        src = Path(pt) if pt else pt_dir / f"{name}{ver}.pt"
        if not src.is_file():
            print(f"[export] 跳过，不存在: {src}")
            continue
        print(f"[export] {src.name} -> ONNX (imgsz={d['imgsz']}, nms={d['nms']}, opset={d['opset']})")
        m = YOLO(str(src))
        exported = Path(m.export(format="onnx", imgsz=d["imgsz"], nms=d["nms"], opset=d["opset"]))
        dst_dir = models_dir(name)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{name}{ver}.onnx"
        shutil.copy2(exported, dst)
        print(f"[export] OK -> {dst} ({dst.stat().st_size / 1e6:.2f} MB)")
        if exported.parent != dst_dir:  # 清理 ultralytics 默认同目录副本
            try:
                exported.unlink()
            except OSError:
                pass
    print("[export] det done")


def export_rec(ver: str, arch: str, src: str | None = None) -> None:
    """PaddleOCR best.pdparams -> 动态宽 ONNX（需 paddlepaddle）。src 直接指定模型目录。"""
    relaunch_if_needed("paddle", "paddle_ocr")
    import yaml as _yaml  # noqa: E402
    sys.path.insert(0, str(tc_root() / "PaddleOCR"))
    import paddle  # noqa: E402
    from ppocr.modeling.architectures import build_model  # noqa: E402

    r = cfg()["rec"]
    yml = tc_root() / "PaddleOCR" / (r["train_yml_server"] if arch == "server" else r["train_yml_mobile"])
    model_dir = Path(src) if src else tc_root() / "models" / "ocr" / f"rec{ver}"
    out = models_dir("rec") / f"rec{ver}_onnx"

    with open(yml, encoding="utf-8") as fp:
        cfg_ = _yaml.safe_load(fp)
    arch_cfg = cfg_["Architecture"]
    print(f"[export] arch: {arch_cfg.get('name')}  yml: {yml.name}")

    # char_num = blank + dict + space（CTCLabelDecode；训练权重通道数必须一致）
    dict_path = tc_root() / r["dict"]
    chars = dict_path.read_text(encoding="utf-8").splitlines()
    char_num = len(chars) + 2
    arch_cfg["Head"]["out_channels_list"] = {
        "CTCLabelDecode": char_num,
        "NRTRLabelDecode": char_num + 3,
    }
    print(f"[export] dict={len(chars)} chars, char_num={char_num}")

    model = build_model(arch_cfg)
    state = paddle.load(str(model_dir / "best.pdparams"))
    model.set_state_dict(state)
    model.eval()

    spec = [paddle.static.InputSpec(shape=[1, 3, 48, -1], dtype="float32", name="x")]
    paddle.onnx.export(model, input_spec=spec, path=str(out), opset_version=12)
    out_onnx = Path(f"{out}.onnx")
    # 产物校验：历史上出现过导出"成功"但产物 0 字节的残留文件
    if not out_onnx.is_file() or out_onnx.stat().st_size == 0:
        raise RuntimeError(f"ONNX 导出失败：产物缺失或为 0 字节: {out_onnx}")
    print(f"[export] rec done -> {out_onnx} ({out_onnx.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="模型导出 ONNX（det / rec 统一入口）")
    p.add_argument("kind", choices=["det", "rec"], help="det=YOLO 检测模型, rec=PaddleOCR 识别模型")
    p.add_argument("--model", default="all", choices=["fabric", "text", "all"], help="det 模式下导出哪些")
    p.add_argument("--version", default=None, help="版本号（默认取 config.versions）")
    p.add_argument("--arch", default="server", choices=["server", "mobile"], help="rec 模式用哪套训练配置")
    p.add_argument("--pt", default=None, help="det: 直接指定 .pt 源文件路径（默认 train_center/models/det/{model}{ver}.pt）")
    p.add_argument("--src", default=None, help="rec: 直接指定含 best.pdparams 的模型目录（默认 train_center/models/ocr/rec{ver}）")
    a = p.parse_args()

    if a.kind == "det":
        if a.pt and a.model == "all":
            p.error("--pt 指定源文件时 --model 不能为 all（fabric/text 版本可能不同）")
        export_det(a.model, a.version or version("fabric"), a.pt)
    else:
        export_rec(a.version or version("rec"), a.arch, a.src)


if __name__ == "__main__":
    main()
