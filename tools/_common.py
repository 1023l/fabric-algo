# -*- coding: utf-8 -*-
"""tools 公共模块：config.yaml 加载、路径解析、日志、环境自动切换。

所有 tools/*.py 通过 `from _common import ...` 使用（同目录，python 直接运行即可）。
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

TOOLS_DIR = Path(__file__).resolve().parent
_CFG_PATH = TOOLS_DIR / "config.yaml"

_cfg_cache: dict | None = None


def cfg() -> dict:
    """读取 config.yaml（进程内缓存）。"""
    global _cfg_cache
    if _cfg_cache is None:
        with open(_CFG_PATH, encoding="utf-8") as fp:
            _cfg_cache = yaml.safe_load(fp)
    return _cfg_cache


# ---- 常用路径快捷方式 ----

def tc_root() -> Path:
    return Path(cfg()["train_center"])


def fa_root() -> Path:
    return Path(cfg()["fabric_algo"])


def models_dir(kind: str) -> Path:
    """fabric-algo 模型目录：kind in {fabric, text, rec}。"""
    return fa_root() / "models" / kind


def version(name: str) -> str:
    """config 里的默认版本号（fabric/text/rec）。"""
    return cfg()["versions"][name]


def staging_dir() -> Path:
    """trt_export 暂存目录（日志 / 中间产物）。"""
    d = Path(cfg()["staging"]["dir"])
    (d / "logs").mkdir(parents=True, exist_ok=True)
    return d


# ---- 日志：stderr + 暂存目录文件双写（沙箱/重定向环境也能留档）----

def make_logger(name: str) -> "Logger":
    return Logger(name)


class Logger:
    def __init__(self, name: str):
        self.path = staging_dir() / "logs" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        self._fh = open(self.path, "w", encoding="utf-8")

    def __call__(self, m: str) -> None:
        self._fh.write(m + "\n")
        self._fh.flush()
        sys.stderr.write(m + "\n")
        sys.stderr.flush()

    def close(self) -> None:
        self._fh.close()


# ---- TRT DLL 路径（tensorrt_cu12_libs 需要 prepend PATH）----

def ensure_trt_dlls() -> None:
    libs = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "tensorrt_cu12_libs"
    if libs.is_dir():
        os.environ["PATH"] = str(libs) + ";" + os.environ.get("PATH", "")


# ---- 环境自动切换：缺依赖就用配置里的解释器重启自己 ----

def relaunch_if_needed(module: str, env_key: str) -> None:
    """当前解释器 import 不了 module 时，用 config.envs[env_key] 的 python 重跑本脚本。"""
    try:
        importlib.import_module(module)
        return
    except ImportError:
        pass
    py = cfg()["envs"].get(env_key)
    if py and Path(py).is_file() and Path(py).resolve() != Path(sys.executable).resolve():
        print(f"[relaunch] 当前环境缺 {module}，用 {py} 重跑 ...")
        # 注意：重跑目标必须是调用方脚本（sys.argv[0]，如 export_onnx.py），
        # 而不是本库文件 _common.py——否则库文件无入口逻辑会秒退 0（静默"成功"）
        r = subprocess.run([py, sys.argv[0], *sys.argv[1:]])
        sys.exit(r.returncode)
    raise SystemExit(f"[fatal] 缺依赖 {module}，且 config.envs.{env_key} 不可用（请检查 config.yaml）")
