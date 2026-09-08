"""公共工具：路径定位、权重查找、中文字体、中文绘制。"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent          # fabric-algo/
PROJECTS = ROOT.parent                          # c:\projects\
FALLBACK_WEIGHTS = PROJECTS / "detect_deliver" / "model" / "latest.pt"

# Windows 常用中文字体候选（按序探测），Linux 回退 Noto CJK
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",          # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",        # 黑体
    "C:/Windows/Fonts/simsun.ttc",        # 宋体
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


def find_chinese_font() -> str:
    """按优先级探测可用的中文字体路径，找不到返回空串。"""
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return path
    return ""


def find_latest_weights() -> Path:
    """runs/<exp>/weights/best.pt 中最新的，否则回退 detect_deliver/model/latest.pt。"""
    runs_dir = ROOT / "runs"
    if runs_dir.is_dir():
        cands = sorted(
            runs_dir.glob("*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if cands:
            return cands[0]
    return FALLBACK_WEIGHTS


def resolve_weights(weights) -> str:
    """解析权重路径：None 自动找最新，相对路径基于 fabric-algo 根目录。"""
    weights = weights or str(find_latest_weights())
    path = Path(weights)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"权重不存在: {path}\n请先训练（python train.py）或用 --weights 指定。"
        )
    return str(path)


def put_chinese_text(img, text, pos, font_size, color_bgr, font_path,
                     stroke_width=0, stroke_color_bgr=(0, 0, 0)):
    """用 PIL 在 OpenCV 图像上绘制中文文本。color_bgr 为 OpenCV BGR 格式。"""
    if not font_path:
        return img
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    if stroke_width > 0:
        stroke_rgb = (stroke_color_bgr[2], stroke_color_bgr[1], stroke_color_bgr[0])
        draw.text(pos, text, font=font, fill=color_rgb,
                  stroke_width=stroke_width, stroke_fill=stroke_rgb)
    else:
        draw.text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
