"""CLIP 图像/文本向量提取（切入点 4：以图搜图）。

用 open_clip 的 ViT-B-32（openai 预训练权重，首次使用自动下载 ~350MB）。
向量用途：布片图库建库 + 图搜图 / 文搜图。

环境变量：
  CLIP_DEVICE - 推理设备，默认 cuda（不可用自动回退 cpu）
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

# HF 直连在大陆环境通常不可达，默认走镜像（用户已显式设置时不覆盖）
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "openai"

_lock = threading.Lock()
_model = None
_preprocess = None
_tokenizer = None
_device = "cpu"
_failed = False


def _get_model():
    global _model, _preprocess, _tokenizer, _device, _failed
    if _failed:
        raise RuntimeError("CLIP 模型初始化失败过，不再重试（见启动日志）")
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    import open_clip
                    import torch
                    _device = "cuda" if torch.cuda.is_available() else "cpu"
                    _model, _, _preprocess = open_clip.create_model_and_transforms(
                        _MODEL_NAME, pretrained=_PRETRAINED)
                    _model = _model.to(_device).eval()
                    _tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
                except Exception as e:
                    _failed = True
                    raise RuntimeError(f"CLIP 模型加载失败: {e}") from e
    return _model, _preprocess, _tokenizer, _device


def embed_image(img: np.ndarray) -> np.ndarray:
    """BGR ndarray → 归一化向量（float32, 1-D）。"""
    import torch
    from PIL import Image as PILImage
    model, preprocess, _, device = _get_model()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = preprocess(PILImage.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).float().cpu().numpy()


def embed_text(text: str) -> np.ndarray:
    """查询文本 → 归一化向量（与图像向量同空间）。"""
    import torch
    model, _, tokenizer, device = _get_model()
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).float().cpu().numpy()


def status() -> dict:
    info = {"model": _MODEL_NAME, "pretrained": _PRETRAINED, "device": _device, "ready": _model is not None}
    if _failed:
        info["error"] = "模型初始化失败（上次尝试）"
    return info
