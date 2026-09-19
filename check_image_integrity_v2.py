#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_image_integrity_v2.py
=============================
Ban nang cap cua check_image_integrity.py: kiem tra tung file anh bang
CA HAI thu vien cv2 va Pillow (PIL), vi 2 thu vien co the khong thong nhat
ve viec 1 file da bi hong hay chua (02_pointcloud_and_mesh.py dung cv2,
02_mesh_tsdf.py dung Pillow - can kiem tra ca 2 de khong bo sot).

CACH DUNG
---------
    python check_image_integrity_v2.py "E:/Thucdia-18May2026/cay_0003_1805"

Script chi DOC, khong sua/xoa gi ca.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Can cai opencv-python: pip install opencv-python")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Can cai pillow: pip install pillow")
    sys.exit(1)


def readable_cv2(path: Path) -> bool:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return img is not None
    except Exception:
        return False


def readable_pil(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def check_folder(folder: Path, exts: tuple[str, ...], label: str) -> list[str]:
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    if not files:
        print(f"  ({label}: khong co file nao trong {folder})")
        return []

    sizes = [p.stat().st_size for p in files]
    median_size = statistics.median(sizes)

    bad = []
    print(f"  Dang kiem tra {len(files)} file trong {folder} (cv2 + PIL) ...")
    for i, p in enumerate(files, 1):
        if i % 500 == 0:
            print(f"    ... {i}/{len(files)}")
        ok_cv2 = readable_cv2(p)
        ok_pil = readable_pil(p)
        if not (ok_cv2 and ok_pil):
            size = p.stat().st_size
            ratio = size / median_size if median_size else 0
            which = []
            if not ok_cv2:
                which.append("cv2")
            if not ok_pil:
                which.append("PIL")
            bad.append(
                f"{p.name}  ({size:,} bytes, ~{ratio:.0%} so voi trung vi "
                f"{median_size:,.0f} bytes)  -- hong theo: {'+'.join(which)}"
            )
    return bad


def main() -> None:
    if len(sys.argv) != 2:
        print('Cach dung: python check_image_integrity_v2.py "duong/dan/toi/thu_muc_scan"')
        sys.exit(1)

    scan_dir = Path(sys.argv[1]).resolve()
    if not scan_dir.exists():
        print(f"Khong tim thay thu muc: {scan_dir}")
        sys.exit(1)

    print("=" * 70)
    print(f"  Kiem tra tinh toan ven anh (cv2 + PIL) trong: {scan_dir}")
    print("=" * 70)

    all_bad: dict[str, list[str]] = {}

    depth_dir = scan_dir / "depth"
    if depth_dir.is_dir():
        all_bad["depth/"] = check_folder(depth_dir, (".png",), "depth")

    rgb_dir = scan_dir / "rgb"
    if rgb_dir.is_dir():
        all_bad["rgb/"] = check_folder(rgb_dir, (".jpg", ".jpeg", ".png"), "rgb")

    conf_dir = scan_dir / "confidence"
    if conf_dir.is_dir():
        all_bad["confidence/"] = check_folder(conf_dir, (".png",), "confidence")

    print()
    print("=" * 70)
    total_bad = sum(len(v) for v in all_bad.values())
    if total_bad == 0:
        print("  KET QUA: Khong phat hien file nao bi hong (ca cv2 va PIL deu doc duoc). Du lieu OK.")
    else:
        print(f"  KET QUA: Phat hien {total_bad} file bi hong / khong doc duoc:")
        for folder_label, bad_list in all_bad.items():
            if not bad_list:
                continue
            print(f"\n  --- {folder_label} ({len(bad_list)} file hong) ---")
            for line in bad_list:
                print(f"    - {line}")
    print("=" * 70)


if __name__ == "__main__":
    main()