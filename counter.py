"""
可逆过线计数器 —— 算法移植自 deploy/fabric_piece_counter.py（领导提供的计数脚本）。

核心思路:
- 在画面 y 轴 LINE_RATIO 处画一条水平计数线
- 检测框质心 上->下 穿过计数线 -> +1（正向过一片）
- 检测框质心 下->上 穿过计数线 -> -1（反向回退一片）
- 死区 ±dead_zone px：质心在死区内不更新状态，防止布机停止时框体抖动误计数
- last_crossed 记录上一次穿越方向，防止同一次穿越被重复计数
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2

from utils import ROOT, put_chinese_text


def extract_size(txt: str):
    """从 OCR 文本提取鞋码："- 前面的数字"（T 可选），如 '30-30TR' -> 30。
    货号类（WS5840 / -W-10 / WSH1084B 等）无此结构返回 None。"""
    if not txt:
        return None
    m = re.search(r"(\d+)\s*T?\s*-\s*\d", txt.upper())
    return int(m.group(1)) if m else None


class LineCounter:
    """逐 track 的可逆过线状态机（纯逻辑，不涉及绘制）。"""

    def __init__(self, line_y: int, dead_zone: int = 15, max_track_age: int = 60):
        self.line_y = line_y
        self.dead_zone = dead_zone
        self.max_track_age = max_track_age
        # track_id -> {clear_side: above/below/None, last_crossed: down/up/None, age: int}
        self.states: dict[int, dict] = {}
        self._down = 0
        self._up = 0

    @property
    def total_down(self) -> int:
        return self._down

    @property
    def total_up(self) -> int:
        return self._up

    @property
    def net(self) -> int:
        return self._down - self._up

    def process(self, boxes, track_ids, clss, confs) -> list:
        """处理一帧的全部检测框，更新计数并返回每框信息（含 side 用于着色）。"""
        # 老化所有 track
        for s in self.states.values():
            s["age"] += 1

        infos = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            track_id = int(track_ids[i])
            centroid_y = (y1 + y2) / 2.0

            # 当前处于死区上方 / 死区内 / 死区下方
            if centroid_y < self.line_y - self.dead_zone:
                side = "above"
            elif centroid_y > self.line_y + self.dead_zone:
                side = "below"
            else:
                side = None  # 死区，不参与计数

            self._update(track_id, side, centroid_y)

            # 本次是否刚过线（跨线当帧为 True，取走后重置，避免重复触发）
            st = self.states.get(track_id, {})
            crossed = bool(st.pop("crossed", False))

            infos.append({
                "box": (x1, y1, x2, y2),
                "track_id": track_id,
                "cls": int(clss[i]),
                "conf": float(confs[i]),
                "side": side,
                "crossed": crossed,
            })

        # 清理过期 track
        stale = [t for t, s in self.states.items() if s["age"] > self.max_track_age]
        for tid in stale:
            del self.states[tid]
        return infos

    def _update(self, track_id: int, side, centroid_y: float) -> None:
        if track_id in self.states:
            st = self.states[track_id]
            prev_side = st["clear_side"]
            last_crossed = st["last_crossed"]

            # 从上方"干净区域"进入下方"干净区域" -> 正向过线 +1
            if prev_side == "above" and side == "below" and last_crossed != "down":
                self._down += 1
                st["last_crossed"] = "down"
                st["crossed"] = True
            # 从下方"干净区域"进入上方"干净区域" -> 反向过线 -1
            elif prev_side == "below" and side == "above" and last_crossed != "up":
                self._up += 1
                st["last_crossed"] = "up"
                st["crossed"] = True

            # 死区内不更新所在侧，保持上一次"干净侧"记录
            if side is not None:
                st["clear_side"] = side
            st["age"] = 0
        else:
            # 新 track：若首次出现在死区内，按质心相对计数线推断初始侧
            init_side = side
            if init_side is None:
                if centroid_y < self.line_y:
                    init_side = "above"
                elif centroid_y > self.line_y:
                    init_side = "below"
            self.states[track_id] = {
                "clear_side": init_side,
                "last_crossed": (
                    "down" if init_side == "below" else
                    "up" if init_side == "above" else None
                ),
                "age": 0,
            }


def run_count_video(model, source: str, weights: str, *, imgsz: int = 640,
                    conf: float = 0.25, device: str = "0", line_ratio: float = 0.5,
                    dead_zone: int = 15, max_track_age: int = 60,
                    output=None, font_path: str = "", show: bool = False,
                    ocr_callback=None) -> dict:
    """ultralytics YOLO 版入口（兼容 infer.py）：内部包一层 detect_fn 转通用主循环。"""
    def detect_fn(frame):
        results = model.track(frame, persist=True, imgsz=imgsz, conf=conf,
                              device=device, verbose=False)
        result = results[0]
        if result.boxes is not None and result.boxes.is_track:
            return (
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.id.int().cpu().numpy(),
                result.boxes.cls.int().cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            )
        return None

    return run_count_video_generic(
        detect_fn, source, weights,
        imgsz=imgsz, conf=conf, line_ratio=line_ratio, dead_zone=dead_zone,
        max_track_age=max_track_age, output=output, font_path=font_path,
        show=show, ocr_callback=ocr_callback,
    )


def run_count_video_generic(detect_fn, source: str, weights: str = "", *,
                            line_ratio: float = 0.5, dead_zone: int = 15,
                            max_track_age: int = 60, output=None,
                            font_path: str = "", show: bool = False,
                            ocr_callback=None, show_ocr_text: bool = True,
                            save_video: bool = True, frame_callback=None,
                            stop_event=None, line_ratio_ref=None,
                            resume: dict | None = None) -> dict:
    """通用视频计数主循环。

    detect_fn(frame) -> (boxes, ids, clss, confs) 或 None（None=本帧无检测）。
        boxes: [N,4] float (x1,y1,x2,y2)；ids/clss/confs: [N]。
    ocr_callback: 可选。每帧调用 ocr_callback(frame, infos, frame_idx)，
        应返回 {track_id: 识别文字}，会叠加在对应布片框下方（黄色）。
    show_ocr_text: False 时不在框上叠加识别文字/鞋码（HUD 统计仍显示），
        用于输出给领导的纯净版。
    save_video: False 时不写输出视频（实时预览模式，output 传 None 即可）。
    frame_callback: 可选。每帧绘制完成后调用 frame_callback(frame, frame_idx, stats)，
        可用于实时推送画面；stats 为 dict(正向/反向/累计/L/R/成双/单只/鞋码)。
    stop_event: 可选 threading.Event，置位后当前帧结束即停止循环。
    line_ratio_ref: 可选可变引用（如 SimpleNamespace(value=...)），每帧读取，
        计数线位置可实时拖拽调整（与 line_ratio 二选一）。
    resume: 可选 dict，暂停/续播上下文（调用方持有引用，counter 就地更新）：
        - 传空 dict 表示"从头开始"，本函数会初始化并写回全部状态
        - 传上次暂停留下的 dict 表示"续播"，会恢复计数/文字/帧位置继续跑
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    line_y = int(height * line_ratio)
    hud_font = max(32, height // 36)      # 叠加字随分辨率放大
    line_thick = max(2, height // 400)
    dz_thick = max(1, height // 700)

    if output:
        output_path = output
    elif save_video:
        # 默认统一输出到 runs/infer/
        out_dir = ROOT / "runs" / "infer"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{Path(source).stem}_counted.mp4")
    else:
        output_path = None  # 实时预览模式，不写视频
    out = None
    if output_path:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # 过线识别重试窗口（帧数）：布片过线当帧登记，后续最多尝试这么多帧，成功即停
    OCR_RETRY_FRAMES = 12

    # ---- 暂停/续播上下文（resume: dict，就地更新，调用方持有引用） ----
    # 首次（resume 为空 dict）：初始化并写回；续播（resume 已有 counter）：恢复全部状态并定位帧
    if resume is not None:
        if "counter" in resume:
            counter = resume["counter"]
            persist_texts: dict = resume["persist_texts"]
            persist_sizes: dict = resume["persist_sizes"]
            size_counts: dict = resume["size_counts"]
            lr_l = resume["lr_l"]
            lr_r = resume["lr_r"]
            pending_ocr: dict[int, int] = resume["pending_ocr"]
            start_idx = resume.get("frame_idx", 0)
            if start_idx > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        else:
            counter = LineCounter(line_y, dead_zone, max_track_age)
            persist_texts = {}
            persist_sizes = {}
            size_counts = {}
            lr_l = 0
            lr_r = 0
            pending_ocr = {}
            resume.update(counter=counter, persist_texts=persist_texts,
                          persist_sizes=persist_sizes, size_counts=size_counts,
                          lr_l=0, lr_r=0, pending_ocr=pending_ocr, frame_idx=0)
    else:
        counter = LineCounter(line_y, dead_zone, max_track_age)
        persist_texts = {}
        persist_sizes = {}
        size_counts = {}
        lr_l = 0
        lr_r = 0
        pending_ocr = {}

    print(f"[count] 权重: {weights or '(detect_fn)'}")
    print(f"[count] 视频: {source} ({width}x{height} @{fps:.1f}fps, {total_frames}帧)")
    print(f"[count] 计数线 y={line_y} (高度{height}的{line_ratio:.0%}处) | 死区 ±{dead_zone}px")
    print("[count] 开始处理...")

    frame_idx = 0
    while cap.isOpened():
        # 停止信号：当前帧处理完后立即退出（用于实时预览的"停止"按钮）
        if stop_event is not None and stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if resume is not None:
            resume["frame_idx"] = frame_idx
            # lr_l/lr_r 是 int（不可变），必须每帧显式写回，暂停续播才能恢复
            resume["lr_l"] = lr_l
            resume["lr_r"] = lr_r

        # 计数线位置动态读取（拖拽实时生效）：line_ratio_ref.value 每帧重新计算 line_y
        if line_ratio_ref is not None:
            lr = float(getattr(line_ratio_ref, "value", None) or line_ratio)
            line_y = int(height * lr)
            counter.line_y = line_y

        det = detect_fn(frame)

        # ---- 计数线与死区绘制 ----
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), line_thick)
        frame = put_chinese_text(frame, "计数线", (10, line_y - hud_font - 4),
                                 hud_font, (0, 0, 255), font_path)
        overlay = frame.copy()
        for dz_y in (line_y - dead_zone, line_y + dead_zone):
            cv2.line(overlay, (0, dz_y), (width, dz_y), (0, 255, 255), dz_thick)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # ---- 计数状态机 ----
        ocr_texts: dict = {}
        if det is not None:
            boxes, ids, clss, confs = det
            infos = counter.process(boxes, ids, clss, confs)

            # 过线识别重试：过线当帧登记，窗口内持续尝试，成功即停
            for tid in list(pending_ocr):
                pending_ocr[tid] -= 1
                if pending_ocr[tid] <= 0:
                    del pending_ocr[tid]
            for info in infos:
                if info.get("crossed"):
                    pending_ocr[info["track_id"]] = OCR_RETRY_FRAMES
            ocr_infos = [i for i in infos if i["track_id"] in pending_ocr]
            if ocr_callback is not None and ocr_infos:
                ocr_texts = ocr_callback(frame, ocr_infos, frame_idx) or {}
                if ocr_texts:
                    for tid, txt in ocr_texts.items():
                        if tid not in persist_texts:  # 只对本次新识别成功的布片统计
                            up = txt.upper()
                            if "L" in up:
                                lr_l += 1
                            if "R" in up:
                                lr_r += 1
                            sz = extract_size(up)
                            if sz is not None:
                                persist_sizes[tid] = sz
                                size_counts[sz] = size_counts.get(sz, 0) + 1
                    persist_texts.update(ocr_texts)
                    for tid in ocr_texts:
                        pending_ocr.pop(tid, None)
            for info in infos:
                x1, y1, x2, y2 = info["box"]
                if info["side"] == "above":
                    box_color = (0, 255, 0)        # 绿 = 线上方
                elif info["side"] == "below":
                    box_color = (255, 165, 0)      # 橙 = 线下方
                else:
                    box_color = (128, 128, 128)    # 灰 = 死区内
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                label = f"ID:{info['track_id']} {info['conf']:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                # OCR 识别文字叠加在框内顶部（黄字黑底，跨帧持续显示），字号自适应框宽且保证可读
                txt = persist_texts.get(info["track_id"]) if show_ocr_text else None
                if txt:
                    base = 1.0
                    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, base, 2)
                    scale = base * min(1.0, (x2 - x1 - 8) / max(tw, 1))
                    scale = max(scale, 0.55)  # 最小字号，保证可读
                    (tw2, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
                    x0, y0 = int(x1) + 4, int(y1) + th2 + 6
                    cv2.rectangle(frame, (x0 - 3, y0 - th2 - 3), (x0 + tw2 + 3, y0 + 3), (0, 0, 0), -1)
                    cv2.putText(frame, txt, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX,
                                scale, (0, 255, 255), 2, cv2.LINE_AA)
                # 鞋码显示（"- 前数字"提取结果），叠加在 OCR 文字下方（绿字黑底）
                sz = persist_sizes.get(info["track_id"]) if show_ocr_text else None
                if sz is not None:
                    sz_txt = f"鞋码:{sz}"
                    sz_font = max(16, hud_font - 4)
                    frame = put_chinese_text(
                        frame, sz_txt, (int(x1) + 4, y0 + sz_font + 6), sz_font,
                        (0, 255, 0), font_path, stroke_width=2, stroke_color_bgr=(0, 0, 0),
                    )

        # ---- HUD 计数信息（白描边绿字，无黑底框） ----
        # 布局（OCR 模式）：鞋码在最上面一行，然后正向/反向/累计（含左右）
        text_lines = []
        if ocr_callback is not None:
            if size_counts:
                top = sorted(size_counts.items(), key=lambda kv: -kv[1])[:6]
                text_lines.append("鞋码: " + " ".join(str(k) for k, _ in top))
            text_lines.append(f"正向鞋布片数: {counter.total_down}")
            text_lines.append(f"反向鞋布片数: {counter.total_up}")
            text_lines.append(f"累计 {counter.net} 只 | L: {lr_l} | R: {lr_r}")
            # 成双统计：一只 L + 一只 R = 一双；多出的单只不计双
            text_lines.append(f"成双 {min(lr_l, lr_r)} 双 | 单只 {abs(lr_l - lr_r)} 只")
        else:
            text_lines = [
                f"正向鞋布片数: {counter.total_down}",
                f"反向鞋布片数: {counter.total_up}",
                f"已检鞋布片数: {counter.net}",
            ]
        y_off = hud_font
        for line_text in text_lines:
            frame = put_chinese_text(
                frame, line_text, (12, y_off), hud_font, (0, 255, 0), font_path,
                stroke_width=max(1, hud_font // 12), stroke_color_bgr=(0, 0, 0),
            )
            y_off += int(hud_font * 1.15)

        if out is not None:
            out.write(frame)

        if show:
            cv2.imshow("fabric-count", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_callback is not None:
            frame_callback(frame, frame_idx, {
                "down": counter.total_down,
                "up": counter.total_up,
                "net": counter.net,
                "lr_l": lr_l, "lr_r": lr_r,
                "pairs": min(lr_l, lr_r), "single": abs(lr_l - lr_r),
                "sizes": dict(size_counts),
            })

        if frame_idx % 100 == 0:
            print(f"  帧 {frame_idx}/{total_frames} | 正向={counter.total_down} "
                  f"反向={counter.total_up} 已检={counter.net}")

    cap.release()
    if out is not None:
        out.release()

    print(f"\n[count] 完成！输出视频: {output_path}")
    print(f"正向过线(下): {counter.total_down}")
    print(f"反向过线(上): {counter.total_up}")
    print(f"已检鞋布片数: {counter.net}")
    if ocr_callback is not None:
        print(f"累计 {counter.net} 只 | 含L: {lr_l} | 含R: {lr_r}")
        print(f"成双 {min(lr_l, lr_r)} 双 | 单只 {abs(lr_l - lr_r)} 只")
        if size_counts:
            top = sorted(size_counts.items(), key=lambda kv: -kv[1])
            print("鞋码统计: " + ", ".join(f"{k}×{v}" for k, v in top))
    return {"output": output_path, "down": counter.total_down,
            "up": counter.total_up, "net": counter.net,
            "lr_l": lr_l, "lr_r": lr_r, "pairs": min(lr_l, lr_r),
            "single": abs(lr_l - lr_r), "sizes": size_counts}
