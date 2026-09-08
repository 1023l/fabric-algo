"""
ONNX 推理引擎 —— 不依赖 torch / paddle，只用 onnxruntime + opencv + numpy。

- YoloOnnxDetector: 加载 ultralytics 导出的 YOLO ONNX（fabric / text），
  letterbox 前处理 + 置信度过滤 + 非极大值抑制(NMS)，输出原图像素坐标框。
- RecOnnxEngine: 加载 PaddleOCR rec 模型转换的 ONNX，
  预处理（RecResizeImg）+ CTC 解码，输出识别文本与置信度。
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

# CUDA EP 需要 CUDA/cuDNN 运行库：若系统未装 CUDA toolkit，则借用 torch 自带的 DLL。
# 必须在 import onnxruntime 之前把 torch/lib 加入 PATH。
def _prepend_torch_lib_to_path():
    try:
        import site

        for sp in site.getsitepackages():
            tl = os.path.join(sp, "torch", "lib")
            if os.path.isdir(tl):
                os.environ["PATH"] = tl + os.pathsep + os.environ.get("PATH", "")
                return
    except Exception:
        pass


_prepend_torch_lib_to_path()

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """等比缩放 + 灰边填充到 new_shape（与 ultralytics 一致）。返回 (图, 原始尺寸信息)。"""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = round(w * r), round(h * r)
    pad_x = (new_shape - nw) // 2
    pad_y = (new_shape - nh) // 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, (pad_x, pad_y, r)


def nms(boxes, scores, iou_thresh=0.45):
    """numpy 实现的非极大值抑制。boxes: [N,4] (x1y1x2y2), scores: [N]。返回保留索引。"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= iou_thresh]
    return keep


class YoloOnnxDetector:
    """YOLO 检测模型（ultralytics 导出的 ONNX，输出 [1, 4+nc, N]）。"""

    def __init__(self, onnx_path, imgsz=640, conf=0.25, iou=0.45, providers=None):
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.nc = None  # 类别数：从输出形状推断

    def detect(self, img_bgr):
        """对单帧 BGR 图检测，返回 [{box:(x1,y1,x2,y2), cls:int, conf:float}]（原图像素坐标）。

        兼容两种 ONNX 输出：
        - YOLO26 端到端 NMS: [1, Nmax, 6]（每行 x1,y1,x2,y2,conf,cls，已做 NMS）
        - 常规 YOLO: [1, 4+nc, N]（原始预测，需自行 NMS）
        """
        h0, w0 = img_bgr.shape[:2]
        letter, (pad_x, pad_y, r) = letterbox(img_bgr, self.imgsz)
        rgb = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = rgb.transpose(2, 0, 1)[None]  # [1,3,H,W]
        out = self.sess.run(None, {self.input_name: inp})[0]  # [1, N, 6] 或 [1, 4+nc, N]

        if out.ndim == 3:
            pred = out[0]
        else:
            pred = out

        # 端到端 NMS 输出：[N, 6]（x1,y1,x2,y2,conf,cls）
        if pred.ndim == 2 and pred.shape[1] == 6:
            dets = []
            for row in pred:
                x1, y1, x2, y2, conf, cls = row
                if float(conf) < self.conf:
                    continue
                dets.append({"box": (x1, y1, x2, y2), "cls": int(cls), "conf": float(conf)})
            # 端到端输出实测可能未完全去重（TRT 构建后重叠框保留），补一道 NMS
            if len(dets) > 1:
                boxes = np.array([d["box"] for d in dets], dtype=np.float32)
                scores = np.array([d["conf"] for d in dets], dtype=np.float32)
                keep = nms(boxes, scores, self.iou)
                dets = [dets[i] for i in keep]
            # 逆 letterbox 还原到原图坐标
            out_dets = []
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                x1 = (x1 - pad_x) / r
                y1 = (y1 - pad_y) / r
                x2 = (x2 - pad_x) / r
                y2 = (y2 - pad_y) / r
                x1, y1 = max(0.0, min(x1, w0)), max(0.0, min(y1, h0))
                x2, y2 = max(0.0, min(x2, w0)), max(0.0, min(y2, h0))
                out_dets.append({"box": (x1, y1, x2, y2), "cls": d["cls"], "conf": d["conf"]})
            return out_dets

        # 常规原始预测输出：[4+nc, N] 或 [N, 4+nc]
        if pred.shape[0] > pred.shape[1] and pred.shape[1] < 20:
            pred = pred.T  # [N, 4+nc]
        self.nc = pred.shape[1] - 4
        pred = pred.astype(np.float32)

        boxes, scores, clss = [], [], []
        for i in range(pred.shape[0]):
            row = pred[i]
            cls_scores = row[4:]
            c = int(cls_scores.argmax())
            sc = float(cls_scores[c])
            if sc < self.conf:
                continue
            x1, y1, x2, y2 = row[:4]
            boxes.append([x1, y1, x2, y2])
            scores.append(sc)
            clss.append(c)
        if not boxes:
            return []
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # 单类/多类 NMS（多类按类别分别 NMS）
        keep = []
        for c in set(clss):
            idx = [i for i, v in enumerate(clss) if v == c]
            if not idx:
                continue
            sub_boxes = boxes[idx]
            sub_scores = scores[idx]
            keep += [idx[i] for i in nms(sub_boxes, sub_scores, self.iou)]
        keep = sorted(keep)

        out_dets = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            # 逆 letterbox 还原到原图坐标
            x1 = (x1 - pad_x) / r
            y1 = (y1 - pad_y) / r
            x2 = (x2 - pad_x) / r
            y2 = (y2 - pad_y) / r
            x1, y1 = max(0.0, min(x1, w0)), max(0.0, min(y1, h0))
            x2, y2 = max(0.0, min(x2, w0)), max(0.0, min(y2, h0))
            out_dets.append({"box": (x1, y1, x2, y2), "cls": clss[i], "conf": float(scores[i])})
        return out_dets


class TrtOnnxDetector:
    """TensorRT engine 检测器（ultralytics 导出 yolo26 engine，端到端 NMS 输出 [1,N,6]）。

    依赖 tensorrt + cupy（无 torch），detect() 接口与 YoloOnnxDetector 一致。
    输出为已做 NMS 的 [1, N, 6]（x1,y1,x2,y2,conf,cls），逆 letterbox 还原原图坐标。
    """

    def __init__(self, engine_path, imgsz=640, conf=0.25, iou=0.45, providers=None):
        import tensorrt as trt
        import cupy as cp

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.cp = cp
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as fp:
            self.engine = self.runtime.deserialize_cuda_engine(fp.read())
        self.ctx = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        # 动态维度(构建时未固定，可能是 -1)回退默认
        if any(s <= 0 for s in in_shape):
            in_shape = (1, 3, imgsz, imgsz)
        if any(s <= 0 for s in out_shape):
            out_shape = (1, 300, 6)
        self.out_shape = out_shape
        self.d_in = cp.empty(int(np.prod(in_shape)), dtype=np.float32)
        self.d_out = cp.empty(int(np.prod(out_shape)), dtype=np.float32)
        self.ctx.set_tensor_address(self.in_name, self.d_in.data.ptr)
        self.ctx.set_tensor_address(self.out_name, self.d_out.data.ptr)
        self._host_out = np.empty(int(np.prod(out_shape)), dtype=np.float32)

    def detect(self, img_bgr):
        """对单帧 BGR 图检测，返回 [{box:(x1,y1,x2,y2), cls:int, conf:float}]（原图像素坐标）。"""
        cp = self.cp
        h0, w0 = img_bgr.shape[:2]
        letter, (pad_x, pad_y, r) = letterbox(img_bgr, self.imgsz)
        rgb = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = rgb.transpose(2, 0, 1).reshape(-1)
        self.d_in.set(x)  # host→device 拷贝（无临时分配）
        if not self.ctx.execute_v2([self.d_in.data.ptr, self.d_out.data.ptr]):
            return []
        self.d_out.get(out=self._host_out)
        pred = self._host_out.reshape(self.out_shape)[0]

        dets = []
        for row in pred:
            x1, y1, x2, y2, conf, cls = row
            if float(conf) < self.conf:
                continue
            dets.append({"box": (x1, y1, x2, y2), "cls": int(cls), "conf": float(conf)})
        # 端到端输出实测可能未完全去重（TRT 构建后重叠框保留），补一道 NMS
        if len(dets) > 1:
            boxes = np.array([d["box"] for d in dets], dtype=np.float32)
            scores = np.array([d["conf"] for d in dets], dtype=np.float32)
            keep = nms(boxes, scores, self.iou)
            dets = [dets[i] for i in keep]
        out_dets = []
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            x1 = max(0.0, min((x1 - pad_x) / r, w0))
            y1 = max(0.0, min((y1 - pad_y) / r, h0))
            x2 = max(0.0, min((x2 - pad_x) / r, w0))
            y2 = max(0.0, min((y2 - pad_y) / r, h0))
            out_dets.append({"box": (x1, y1, x2, y2), "cls": d["cls"], "conf": d["conf"]})
        return out_dets


class RecOnnxEngine:
    """PaddleOCR rec 模型（转 ONNX 后）识别引擎。

    - 预处理与 PaddleOCR RecResizeImg 一致：等比缩放到高 48，宽自适应（≤320），灰边填充
    - 归一化：/255 - 0.5 再 /0.5（等价 (x/255-0.5)/0.5）
    - 解码：CTC（argmax → 去重连续 → 去 blank）
    """

    def __init__(self, onnx_path, dict_chars, rec_height=48, max_width=320,
                 score_thresh=0.5, providers=None):
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        # PaddleOCR CTCLabelDecode 字符表：['blank'] + dict + (use_space_char 追加 ' ')
        # 输出维度 num_classes = len(character)，blank 在 idx 0
        self.chars = [""] + list(dict_chars) + [" "]
        self.height = rec_height
        self.max_width = max_width
        self.score_thresh = score_thresh

    def _preprocess(self, img_bgr):
        # 与 PaddleOCR RecResizeImg(resize_norm_img) 一致：
        # 等比缩放到高 48，宽=ceil(48*w/h)（≤320），内容区归一化 (x/255-0.5)/0.5，
        # 右侧 padding 保持原始 0.0（不参与归一化）
        h, w = img_bgr.shape[:2]
        nw = int(math.ceil(self.height * w / h))
        nw = min(nw, self.max_width)
        resized = cv2.resize(img_bgr, (nw, self.height), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        x = (x - 0.5) / 0.5
        canvas = np.zeros((3, self.height, self.max_width), dtype=np.float32)
        canvas[:, :, :nw] = x
        return canvas, nw  # (3,48,max_width) + 内容宽

    def recognize(self, img_bgr):
        """识别一张文字图（BGR），返回 (text, score)。识别失败返回 ("", 0.0)。"""
        if img_bgr is None or img_bgr.size == 0:
            return "", 0.0
        x, _nw = self._preprocess(img_bgr)
        out = self.sess.run(None, {self.input_name: x[None]})[0]  # [1, seq, num_cls]
        logits = out[0]  # [seq, num_cls]
        pred_idx = logits.argmax(axis=1)
        confs = logits.max(axis=1)
        # CTC 解码：去重连续相同 + 去 blank(idx 0)
        chars = []
        scores = []
        prev = -1
        for idx, conf in zip(pred_idx.tolist(), confs.tolist()):
            if idx == prev:
                continue
            prev = idx
            if idx == 0 or idx >= len(self.chars):
                continue
            chars.append(self.chars[idx])
            scores.append(float(conf))
        text = "".join(chars)
        score = float(np.mean(scores)) if scores else 0.0
        if not text or score < self.score_thresh:
            return "", 0.0
        return text, score

    @staticmethod
    def _read_chars(onnx_path, rec_dir=None):
        """从 onnx 同目录 / rec_dir / 其子目录的 inference.yml 读字符集。"""
        chars = None
        for d in (rec_dir, Path(onnx_path).parent):
            if d is None:
                continue
            root = Path(d)
            # 目录本身或其一级子目录下的 inference.yml
            for yml in (root / "inference.yml", *sorted(root.glob("*/inference.yml"))):
                if yml.is_file():
                    txt = yml.read_text(encoding="utf-8")
                    m = re.search(r"character_dict:\s*\n((?:\s*-\s*.*\n?)+)", txt)
                    if m:
                        chars = [ln.strip().lstrip("-").strip().strip("'\"")
                                 for ln in m.group(1).splitlines() if ln.strip()]
                        break
            if chars:
                break
        if not chars:
            raise ValueError(f"未找到 character_dict（inference.yml 缺失或格式不符）: {onnx_path}")
        return chars

    @staticmethod
    def load(onnx_path, rec_dir=None, score_thresh=0.5, max_width=192, providers=None):
        """便捷加载：从 onnx 同目录 / rec_dir / 其子目录的 inference.yml 读字符集。"""
        chars = RecOnnxEngine._read_chars(onnx_path, rec_dir)
        return RecOnnxEngine(onnx_path, chars, score_thresh=score_thresh,
                             max_width=max_width, providers=providers)


class RecTrtEngine(RecOnnxEngine):
    """TensorRT 版 rec 识别（ultralytics/Paddle 导出的动态宽 engine）。

    与 RecOnnxEngine 同接口（recognize / load / chars / max_width），
    依赖 tensorrt + cupy。实测（4060，w=64）2.1ms vs onnxruntime CUDA 19ms。
    """

    def __init__(self, engine_path, dict_chars, rec_height=48, max_width=320,
                 score_thresh=0.5):
        import tensorrt as trt
        import cupy as cp

        self.chars = [""] + list(dict_chars) + [" "]
        self.height = rec_height
        self.max_width = max_width
        self.score_thresh = score_thresh
        self.cp = cp
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as fp:
            self.engine = self.runtime.deserialize_cuda_engine(fp.read())
        self.ctx = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        # 输入 (1,3,48,动态宽)，d_in 按 max_width 上限分配（地址不变，用 slice set）
        self.d_in = cp.empty(3 * rec_height * max_width, dtype=np.float32)
        # 输出 (1,seq,24)，seq 动态；分配充足上限，执行后按实际 shape slice
        self.d_out = cp.empty(1 << 20, dtype=np.float32)  # 4MB，足够
        self.ctx.set_tensor_address(self.in_name, self.d_in.data.ptr)
        self.ctx.set_tensor_address(self.out_name, self.d_out.data.ptr)

    def recognize(self, img_bgr):
        """识别一张文字图（BGR），返回 (text, score)。识别失败返回 ("", 0.0)。"""
        if img_bgr is None or img_bgr.size == 0:
            return "", 0.0
        cp = self.cp
        x, nw = self._preprocess(img_bgr)  # (3,48,max_width) + 内容宽
        in_shape = (1, 3, self.height, nw)
        self.ctx.set_input_shape(self.in_name, in_shape)
        n = 3 * self.height * nw
        self.d_in[:n].set(np.ascontiguousarray(x[:, :, :nw]).reshape(-1))
        if not self.ctx.execute_v2([self.d_in.data.ptr, self.d_out.data.ptr]):
            return "", 0.0
        out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        need = int(np.prod(out_shape))
        host = np.empty(need, dtype=np.float32)
        self.d_out[:need].get(out=host)
        logits = host.reshape(out_shape)[0]  # [seq, num_cls]
        pred_idx = logits.argmax(axis=1)
        confs = logits.max(axis=1)
        # CTC 解码：去重连续相同 + 去 blank(idx 0)
        chars = []
        scores = []
        prev = -1
        for idx, conf in zip(pred_idx.tolist(), confs.tolist()):
            if idx == prev:
                continue
            prev = idx
            if idx == 0 or idx >= len(self.chars):
                continue
            chars.append(self.chars[idx])
            scores.append(float(conf))
        text = "".join(chars)
        score = float(np.mean(scores)) if scores else 0.0
        if not text or score < self.score_thresh:
            return "", 0.0
        return text, score

    @staticmethod
    def load(engine_path, rec_dir=None, score_thresh=0.5, max_width=192):
        """便捷加载（字符集读取同 RecOnnxEngine）。"""
        chars = RecOnnxEngine._read_chars(engine_path, rec_dir)
        return RecTrtEngine(engine_path, chars, score_thresh=score_thresh,
                            max_width=max_width)
