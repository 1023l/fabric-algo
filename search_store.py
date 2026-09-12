"""布片图库（以图搜图数据层）：CLIP 向量 + 元数据，numpy 余弦检索。

存储结构（search_db/，运行时数据，不入 git）：
  meta.jsonl      每行一条 {"id", "file", "source", "ts"}
  embeddings.npy  N×512 float32，行序与 meta.jsonl 一致
  images/<id>.jpg 入库图（裁剪后的布片缩略图）

检索规模万级以内直接 numpy 暴力余弦（<10ms），无需 faiss。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DB_DIR = ROOT / "search_db"
IMG_DIR = DB_DIR / "images"
META_PATH = DB_DIR / "meta.jsonl"
EMB_PATH = DB_DIR / "embeddings.npy"

_lock = threading.Lock()

# 进程内缓存（与磁盘同步：每次写盘后刷新）
_meta: list[dict] = []
_emb: np.ndarray | None = None


def _ensure_dirs() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)


def _load() -> tuple[list[dict], np.ndarray | None]:
    global _meta, _emb
    if _emb is not None:
        return _meta, _emb
    if META_PATH.is_file() and EMB_PATH.is_file():
        _meta = [json.loads(l) for l in META_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        _emb = np.load(EMB_PATH)
        if len(_meta) != len(_emb):  # 数据不一致时以 meta 为准截断/报错
            n = min(len(_meta), len(_emb))
            _meta, _emb = _meta[:n], _emb[:n]
    else:
        _meta, _emb = [], None
    return _meta, _emb


def _flush() -> None:
    _ensure_dirs()
    META_PATH.write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in _meta) + ("\n" if _meta else ""),
        encoding="utf-8",
    )
    if _emb is not None and len(_emb):
        np.save(EMB_PATH, _emb)
    elif EMB_PATH.is_file():
        EMB_PATH.unlink()


def _next_id() -> int:
    return (max((int(m["id"]) for m in _meta), default=0)) + 1


def add(img: np.ndarray, *, source: str = "upload", max_side: int = 512) -> dict:
    """入库一张布片图：存缩略图 + 向量。返回该条 meta。"""
    import clip_embed

    global _emb
    feat = clip_embed.embed_image(img)
    with _lock:
        _load()
        _ensure_dirs()
        item_id = _next_id()
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        rel = f"images/{item_id}.jpg"
        cv_imwrite_ok = cv2.imwrite(str(DB_DIR / rel), img)
        if not cv_imwrite_ok:
            raise RuntimeError("缩略图写入失败")
        meta = {"id": item_id, "file": rel, "source": source,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _meta.append(meta)
        _emb = feat[None, :] if _emb is None else np.vstack([_emb, feat[None, :]])
        _flush()
    return meta


def remove(item_id: int) -> bool:
    global _emb
    with _lock:
        _load()
        idx = next((i for i, m in enumerate(_meta) if int(m["id"]) == item_id), None)
        if idx is None:
            return False
        f = DB_DIR / _meta[idx]["file"]
        if f.is_file():
            try:
                f.unlink()
            except OSError:
                pass
        _meta.pop(idx)
        _emb = np.delete(_emb, idx, axis=0) if _emb is not None and len(_emb) else None
        _flush()
    return True


def search(vec: np.ndarray, *, top_k: int = 10, min_score: float = 0.0) -> list[dict]:
    """余弦相似检索 top-k（vec 需已归一化；返回含 meta + score + 图 URL）。"""
    with _lock:
        meta, emb = _load()
        if emb is None or not len(meta):
            return []
        scores = (emb @ vec.astype(np.float32)).tolist()
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order[:top_k]:
            if scores[i] < min_score:
                break
            out.append({**meta[i], "score": round(scores[i], 4),
                        "url": f"/files/{DB_DIR.name}/{meta[i]['file']}"})
        return out


def count() -> int:
    with _lock:
        _load()
        return len(_meta)


if __name__ == "__main__":
    # 自检：真实帧图入库 → 查询图 A 的 top1 应为 A 自己 → 清理
    import numpy as np
    import clip_embed
    from utils import ROOT

    # 幂等：先清掉残留
    for m in list(_load()[0]):
        remove(int(m["id"]))

    imgs = sorted((ROOT / "runs" / "infer" / "_compare_frames" / "det").glob("frame_0000[01]*.jpg"))
    if len(imgs) < 2:
        raise SystemExit("自检需要 runs/infer/_compare_frames/det 下至少 2 张帧图")
    m1 = add(cv_imread(imgs[0]), source="selftest")
    m2 = add(cv_imread(imgs[1]), source="selftest")
    print("count:", count())

    q = clip_embed.embed_image(cv_imread(imgs[0]))
    hits = search(q, top_k=2)
    print("search img0 ->", [(h["id"], h["score"]) for h in hits])
    assert hits[0]["id"] == m1["id"], "top1 应为查询图自身"
    assert hits[0]["score"] >= hits[1]["score"], "自身相似度应最高"
    remove(m1["id"]); remove(m2["id"])
    print("cleaned, count:", count())
    print("SELFTEST OK")
