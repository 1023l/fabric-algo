# -*- coding: utf-8 -*-
"""算法接口冒烟测试（原 _algo_api_smoke.py 通用化）。

对运行中的 fabric-algo web_server 逐个打算法接口：
    fabric/json -> text/json -> ocr/json -> pieces_ocr/json -> count_frame/json -> 清理会话

用法:
    python tools/api_smoke.py [--base http://127.0.0.1:8001] [--image 某帧.jpg]

结果 JSON 落到 trt_export/logs/api_smoke_<时间戳>.json。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cfg, fa_root, staging_dir  # noqa: E402


def b64(p: Path) -> str:
    if not p.is_file():
        raise SystemExit(f"image not found: {p}")
    return base64.b64encode(p.read_bytes()).decode("ascii")


def post(base: str, path: str, body: dict):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode("utf-8"),
                                 method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8")) if e.fp else {"detail": str(e)}


def pick_image(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    cand = Path(cfg()["smoke"]["image"])
    if cand.is_file():
        return cand
    hits = sorted(fa_root().glob("runs/**/*.jpg"))
    if not hits:
        raise SystemExit("找不到测试图：请用 --image 指定一张整帧图")
    return hits[0]


def main() -> None:
    p = argparse.ArgumentParser(description="算法接口冒烟测试")
    p.add_argument("--base", default=None)
    p.add_argument("--image", default=None, help="整帧测试图（默认 config.smoke.image）")
    a = p.parse_args()
    base = a.base or cfg()["smoke"]["base"]
    img = pick_image(a.image)
    print(f"[smoke] base={base} image={img}")
    enc = b64(img)
    results: dict = {}

    t0 = time.time()
    st, r = post(base, "/fabric/json", {"image_b64": enc, "fabric_conf": 0.4, "iou": 0.45})
    n_fab = len((r.get("data") or {}).get("boxes", []))
    results["fabric"] = {"status": st, "n_boxes": n_fab}
    print(f"[fabric] status={st} n_boxes={n_fab} took={time.time()-t0:.2f}s")

    t0 = time.time()
    st, r = post(base, "/text/json", {"image_b64": enc})
    n_txt = len((r.get("data") or {}).get("boxes", []))
    results["text"] = {"status": st, "n_boxes": n_txt}
    print(f"[text]   status={st} n_boxes={n_txt} took={time.time()-t0:.2f}s")

    # ocr：从 fabric 最可信框里裁右上 1/4 当小图
    import cv2
    import numpy as np

    ocr_res: dict = {"skipped": True}
    fab_boxes = (r.get("data") or {}).get("boxes", [])
    if results["fabric"]["n_boxes"] > 0:
        t0 = time.time()
        st, r = post(base, "/fabric/json", {"image_b64": enc, "fabric_conf": 0.4})
        boxes = (r.get("data") or {}).get("boxes", []) or fab_boxes
        img_arr = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_COLOR)
        x1, y1, x2, y2 = [int(v) for v in boxes[0]["box"]]
        crop = img_arr[y1 + int((y2 - y1) * 0.15):y1 + int((y2 - y1) * 0.35),
                       x1 + int((x2 - x1) * 0.55):x2 - int((x2 - x1) * 0.05)]
        ok2, buf = cv2.imencode(".jpg", crop)
        if ok2 and crop.size:
            small = base64.b64encode(buf.tobytes()).decode("ascii")
            st, r = post(base, "/ocr/json", {"image_b64": small, "rec_thresh": 0.3})
            ocr_res = {"status": st, "text": (r.get("data") or {}).get("text"),
                       "score": (r.get("data") or {}).get("score")}
            print(f"[ocr]    status={st} text={ocr_res['text']!r} score={ocr_res['score']} took={time.time()-t0:.2f}s")
    results["ocr"] = ocr_res

    t0 = time.time()
    st, r = post(base, "/pieces_ocr/json", {"image_b64": enc, "ocr": True, "rec_workers": 2})
    pieces = (r.get("data") or {}).get("pieces", [])
    results["pieces_ocr"] = {"status": st, "count": (r.get("data") or {}).get("count"),
                             "texts": [(p["shoe_size"], p["lr_flag"], p["text"]) for p in pieces]}
    print(f"[pieces] status={st} count={results['pieces_ocr']['count']} took={time.time()-t0:.2f}s")

    sid = f"smoke_{int(time.time())}"
    last = None
    for i in range(3):
        st, r = post(base, "/count_frame/json", {"image_b64": enc, "session_id": sid,
                                                 "line_ratio": 0.55, "ocr": True, "rec_workers": 2})
        last = (r.get("data") or {}).get("stats")
        print(f"[count {i}] status={st} stats={last}")
    results["count_frame"] = {"status": st, "last_stats": last, "session_id": sid}

    req = urllib.request.Request(base + f"/count_sessions/{sid}", method="DELETE")
    with urllib.request.urlopen(req, timeout=10) as resp:
        results["session_reset"] = json.loads(resp.read().decode("utf-8"))

    out = staging_dir() / "logs" / f"api_smoke_{int(time.time())}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SMOKE DONE ->", out)


if __name__ == "__main__":
    main()
