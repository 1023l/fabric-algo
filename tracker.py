"""
轻量 IoU 跟踪器 —— 替代 ultralytics 内置 ByteTrack（ONNX 推理无跟踪器）。

匹配策略：每帧检测框与现存 track 按 IoU 贪心匹配（IoU 最大优先）。
- 匹配上的 track 更新位置、age 归零
- 未匹配的检测框 → 新建 track（连续命中 min_hits 帧才算稳定，可配置）
- 未匹配的 track → age+1，超过 max_age 自动删除
对布片过线计数场景（目标移动连续、画面内目标少）足够稳定。
"""

from __future__ import annotations


def iou(box1, box2) -> float:
    """两个 [x1,y1,x2,y2] 框的 IoU。"""
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)


class SimpleTracker:
    """极简 IoU 跟踪器：维护 track_id 跨帧稳定，供过线计数使用。"""

    def __init__(self, max_age: int = 60, iou_thresh: float = 0.3,
                 min_hits: int = 1, next_id: int = 1):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.min_hits = min_hits
        # track_id -> {"box": (x1,y1,x2,y2), "age": int, "hits": int}
        self.tracks: dict[int, dict] = {}
        self._next_id = next_id  # 续播时传入旧最大 id+1，避免与暂停前 persist 键冲突

    def update(self, dets: list):
        """dets: [{box:(x1,y1,x2,y2), cls:int, conf:float}]，返回匹配结果列表。

        返回: [{box, track_id, cls, conf, side_hint}]，side_hint 无意义（兼容旧接口）。
        """
        if not dets:
            for t in self.tracks.values():
                t["age"] += 1
            self._cleanup()
            return []

        # 1) 贪心匹配：按 IoU 从大到小分配 det -> track
        matches = []  # (iou, det_idx, track_id)
        for di, det in enumerate(dets):
            for tid, tr in self.tracks.items():
                v = iou(det["box"], tr["box"])
                if v >= self.iou_thresh:
                    matches.append((v, di, tid))
        matches.sort(reverse=True, key=lambda m: m[0])
        used_det, used_tr = set(), set()
        assigned = []  # (det_idx, track_id)
        for v, di, tid in matches:
            if di in used_det or tid in used_tr:
                continue
            used_det.add(di)
            used_tr.add(tid)
            assigned.append((di, tid))

        # 2) 更新匹配上的 track
        for di, tid in assigned:
            tr = self.tracks[tid]
            tr["box"] = dets[di]["box"]
            tr["age"] = 0
            tr["hits"] += 1

        # 3) 未匹配 track 老化
        for tid, tr in self.tracks.items():
            if tid not in used_tr:
                tr["age"] += 1
        self._cleanup()

        # 4) 未匹配 det 新建 track（跨帧连续命中后才输出，减少闪烁误跟踪）
        for di in range(len(dets)):
            if di in used_det:
                continue
            det = dets[di]
            self.tracks[self._next_id] = {"box": det["box"], "age": 0, "hits": 1}
            self._next_id += 1

        # 5) 组装输出：只输出命中次数足够的稳定 track
        out = []
        for di, tid in assigned:
            tr = self.tracks[tid]
            if tr["hits"] < self.min_hits:
                continue
            d = dets[di]
            out.append({
                "box": d["box"],
                "track_id": tid,
                "cls": d["cls"],
                "conf": d["conf"],
            })
        return out

    def _cleanup(self) -> None:
        stale = [t for t, s in self.tracks.items() if s["age"] > self.max_age]
        for t in stale:
            del self.tracks[t]
