#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_detect_lichen_diagnose3.py
==================================
Va THEM 05_detect_lichen.py: in ra bounding box (vung khong gian) cua:
  1. Toan bo diem trong trunk_pointcloud.ply
  2. Rieng cac diem duoc gan nhan "trang" (nghi la dia y)
  3. Cac vertex cua mesh (dung de phat hien dia y)

Neu bounding box cua (2) va (3) khong giao nhau (hoac giao rat it), day la
bang chung truc tiep cho thay trunk_pointcloud.ply va mesh KHONG khop vi
tri khong gian voi nhau (co the do 2 pipeline dung hinh doc lap - TSDF cho
mesh, RGBD/Poisson cho pointcloud - bi lech he toa do o mot so scan).

CACH DUNG (chay SAU cac patch chan doan truoc)
---------------------------------------------------
    python patch_detect_lichen_diagnose3.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_detect_lichen.py"

OLD_SNIPPET = '''    print(f"    PCD: white={( pcd_labels==2).sum():,} ({(pcd_labels==2).sum()/n*100:.1f}%)"
          f"  green={(pcd_labels==1).sum():,} ({(pcd_labels==1).sum()/n*100:.1f}%)",
          flush=True)'''

NEW_SNIPPET = '''    print(f"    PCD: white={( pcd_labels==2).sum():,} ({(pcd_labels==2).sum()/n*100:.1f}%)"
          f"  green={(pcd_labels==1).sum():,} ({(pcd_labels==1).sum()/n*100:.1f}%)",
          flush=True)

    # --- CHAN DOAN 3: so sanh vung khong gian (bounding box) ---
    def _bbox(arr):
        if len(arr) == 0:
            return None
        return arr.min(axis=0), arr.max(axis=0)

    bb_all = _bbox(pts)
    bb_white = _bbox(pts[pcd_labels == 2])
    bb_verts = _bbox(verts.astype(np.float32))
    print(f"    [chan doan 3] Bounding box toan bo PCD     : "
          f"min={bb_all[0]}  max={bb_all[1]}")
    if bb_white is not None:
        print(f"    [chan doan 3] Bounding box diem TRANG    : "
              f"min={bb_white[0]}  max={bb_white[1]}")
    else:
        print(f"    [chan doan 3] Bounding box diem TRANG    : (khong co diem nao)")
    print(f"    [chan doan 3] Bounding box mesh vertices : "
          f"min={bb_verts[0]}  max={bb_verts[1]}")
    if bb_white is not None:
        overlap = np.minimum(bb_white[1], bb_verts[1]) - np.maximum(bb_white[0], bb_verts[0])
        print(f"    [chan doan 3] Do giao nhau (am = khong giao): {overlap}")'''


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print(f"Hay dat file nay cung thu muc voi {TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content:
        print(f"{TARGET_NAME} (chan doan 3) da duoc va truoc do roi. Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_diagnose3")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong (chan doan 3): {target}")
    print("Chay lai buoc detect lichen de xem ket qua chan doan moi.")


if __name__ == "__main__":
    main()