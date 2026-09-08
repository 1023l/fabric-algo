# -*- coding: utf-8 -*-
"""det 数据集图片找回（原 _restore_det_text.py 通用化）。

按 labels/<split>/*.txt 的文件名，从 --src（默认 train-center/data/_uploads）递归找回
同名 jpg，复制回 train-center/data/<dataset>/images/<split>/。找不到的打印 MISS。
顺带清理 images/ 下误残留的 txt。

用法:
    python tools/restore_det_images.py --dataset det_text [--src X]

适用任何 det 数据集（det_text / det_fabric / ...），标错删图后自救用。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import tc_root  # noqa: E402


def restore(dataset: str, src: Path, root: Path) -> None:
    det = root / "data" / dataset
    for split in ("train", "val"):
        label_dir = det / "labels" / split
        if not label_dir.is_dir():
            print(f"[restore] 跳过（无 labels/{split}）: {label_dir}")
            continue
        img_dir = det / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        n_found = 0
        miss: list[str] = []
        total = 0
        for txt in sorted(label_dir.glob("*.txt")):
            total += 1
            jpg_name = txt.stem + ".jpg"
            dst = img_dir / jpg_name
            if dst.is_file():
                n_found += 1
                continue
            hits = list(src.rglob(jpg_name))
            if hits:
                shutil.copy2(hits[0], dst)
                n_found += 1
            else:
                miss.append(jpg_name)
        print(f"[restore] {dataset}/{split}: 找回 {n_found} / 共 {total}")
        if miss:
            print(f"[restore] {dataset}/{split} MISS（源图不存在）: {', '.join(sorted(set(miss)))}")

        # 清理 images/ 下误残留 txt（训练只读 labels/，但残留会干扰打包/核对）
        for t in img_dir.glob("*.txt"):
            t.unlink()
    print("[restore] done")


def main() -> None:
    p = argparse.ArgumentParser(description="det 数据集图片找回")
    p.add_argument("--dataset", required=True, help="如 det_text / det_fabric")
    p.add_argument("--src", default=None, help="搜索源目录（默认 train-center/data/_uploads）")
    a = p.parse_args()

    root = tc_root()
    src = Path(a.src) if a.src else root / "data" / "_uploads"
    if not src.is_dir():
        raise SystemExit(f"源目录不存在: {src}（用 --src 指定图片所在目录）")
    restore(a.dataset, src, root)


if __name__ == "__main__":
    main()
