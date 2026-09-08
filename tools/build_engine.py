# -*- coding: utf-8 -*-
"""ONNX -> TRT engine 统一构建（合并原 _trt11_onnx.py / _trt11_rec_onnx.py）。

用法:
    python tools/build_engine.py --model fabric|text|rec|all [--version X] [--force]

说明:
    - fabric/text: 静态 640 输入，直接 OnnxParser 构建
    - rec:        动态宽 profile 1x3x48x{min,opt,max}（config.trt.rec_width）
    - 版本化命名 {name}{version}.engine，已存在则跳过（--force 重建同名）
    - 计时缓存: config.trt.timing_cache（trt_cache 目录），同机重复构建提速
    - OnnxParser 直接构建，不依赖 modelopt/ultralytics
    - eval 版 TRT（tensorrt_cu12）无 FP16/INT8 flag 时自动 FP32 构建

需要 yolo-bench 环境（tensorrt + cupy）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cfg, ensure_trt_dlls, make_logger, models_dir, version  # noqa: E402

ensure_trt_dlls()
import tensorrt as trt  # noqa: E402


def _build(onnx_path: Path, out: Path, log, dynamic: dict | None = None, force: bool = False) -> None:
    """通用构建：parse onnx -> builder config(+timing cache, +dynamic profile) -> serialize。"""
    if out.exists() and not force:
        log(f"== {out.name} ==  already exists, skip (用 --force 重建)")
        return
    if not onnx_path.exists():
        log(f"== {out.name} ==  onnx not found, skip: {onnx_path}")
        return

    log(f"== {onnx_path.name} -> {out.name} ==")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as fp:
        if not parser.parse(fp.read()):
            for i in range(min(parser.num_errors, 10)):
                log(f"  parse err: {parser.get_error(i)}")
            return
    inp = network.get_input(0)
    log(f"  parse ok: input {inp.name} {inp.shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(cfg()["trt"]["workspace"]))

    flags = [a for a in dir(trt.BuilderFlag) if not a.startswith("_")]
    if hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        log(f"  FP16 flag not found, building fp32 (flags={flags})")

    # 计时缓存（trt_cache，同机重建提速；旧版本 TRT 无此 API 时跳过）
    cache = None
    tc_path = cfg()["trt"].get("timing_cache")
    if tc_path and hasattr(config, "create_timing_cache"):
        blob = b""
        if os.path.exists(tc_path):
            blob = open(tc_path, "rb").read()
            log(f"  timing cache loaded: {tc_path} ({len(blob)} B)")
        cache = config.create_timing_cache(blob)
        config.set_timing_cache(cache)

    # rec 动态宽 profile
    if dynamic:
        mn, op, mx = dynamic["min"], dynamic["opt"], dynamic["max"]
        profile = builder.create_optimization_profile()
        profile.set_shape(inp.name, (1, 3, 48, mn), (1, 3, 48, op), (1, 3, 48, mx))
        config.add_optimization_profile(profile)
        log(f"  profile: min(1,3,48,{mn}) opt(1,3,48,{op}) max(1,3,48,{mx})")

    t0 = time.perf_counter()
    log("  building ...")
    eng = builder.build_serialized_network(network, config)
    dt = time.perf_counter() - t0
    if eng:
        out.write_bytes(eng)
        log(f"  OK {dt:.0f}s -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    else:
        log(f"  FAILED {dt:.0f}s")

    # 保存计时缓存供下次构建
    if cache is not None:
        try:
            blob = cache.serialize()
            if blob:
                os.makedirs(os.path.dirname(tc_path), exist_ok=True)
                with open(tc_path, "wb") as w:
                    w.write(blob)
                log(f"  timing cache saved: {tc_path} ({len(blob)} B)")
        except Exception as e:  # noqa: BLE001
            log(f"  timing cache save failed: {e}")
    del network, parser


def main() -> None:
    p = argparse.ArgumentParser(description="ONNX -> TRT engine 统一构建")
    p.add_argument("--model", default="all", choices=["fabric", "text", "rec", "all"])
    p.add_argument("--version", default=None, help="默认取 config.versions；all 模式下各用各的默认版本")
    p.add_argument("--force", action="store_true", help="engine 已存在时强制重建（仅同名版本）")
    a = p.parse_args()
    log = make_logger("build_engine")

    t = cfg()["trt"]
    if a.model in ("fabric", "text", "all"):
        names = ["fabric", "text"] if a.model == "all" else [a.model]
        for name in names:
            ver = a.version or version(name)
            _build(models_dir(name) / f"{name}{ver}.onnx",
                   models_dir(name) / f"{name}{ver}.engine", log, force=a.force)
    if a.model in ("rec", "all"):
        ver = a.version or version("rec")
        onnx = models_dir("rec") / f"rec{ver}_onnx.onnx"
        if not onnx.exists():
            onnx = models_dir("rec") / f"rec{ver}.onnx"  # 兼容无 _onnx 后缀命名
        _build(onnx, models_dir("rec") / f"rec{ver}.engine", log,
               dynamic=t["rec_width"], force=a.force)

    log("done")


if __name__ == "__main__":
    main()
