#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trunk_isolation_downsample.py
=====================================
Va BO SUNG cho 05_trunk_isolation.py (chay SAU patch_trunk_isolation.py):
them buoc voxel-downsample NGAY SAU khi loc diem bat thuong, TRUOC khi chay
RANSAC ground-plane removal / KDTree statistical-outlier-removal / DBSCAN -
day la 3 buoc nang nhat, de bi cham hoac treo may khi point cloud co hang
trieu diem tren may RAM han che.

Voxel 1cm van du chi tiet de xac dinh hinh dang/truc than cay - chi tiet
dia y lay tu TSDF mesh o buoc rieng (02_mesh_tsdf.py), khong phu thuoc vao
file pointcloud.ply nay.

CACH DUNG (chay SAU khi da chay patch_trunk_isolation.py it nhat 1 lan)
------------------------------------------------------------------------
    python patch_trunk_isolation_downsample.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_trunk_isolation.py"

OLD_SNIPPET = '''        if n_orig == 0:
            print("  SKIP: point cloud rong sau khi loc diem bat thuong")
            return False

    # Step 1: Loại bỏ ground plane
    pcd = remove_ground_plane(pcd)'''

NEW_SNIPPET = '''        if n_orig == 0:
            print("  SKIP: point cloud rong sau khi loc diem bat thuong")
            return False

    # --- Downsample som de giam RAM/thoi gian cho RANSAC/KDTree/DBSCAN ---
    # Voi may RAM han che, point cloud qua nhieu diem (hang trieu) lam cac
    # buoc phia sau cham hoac treo do swap bo nho. Voxel 1cm van du chi
    # tiet de xac dinh hinh dang than cay (chi tiet dia y lay tu TSDF mesh
    # o buoc rieng, khong phu thuoc vao day).
    if n_orig > 1_000_000:
        before = n_orig
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        n_orig = len(pcd.points)
        print(f"  Downsample (voxel 1cm): {before:,} -> {n_orig:,} pts")

    # Step 1: Loại bỏ ground plane
    pcd = remove_ground_plane(pcd)'''


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print("Hay dat file nay cung thu muc voi "
              f"{TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content:
        print(f"{TARGET_NAME} da duoc va downsample truoc do roi. "
              "Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Kiem tra xem da chay patch_trunk_isolation.py truoc chua "
              "(patch nay can chay SAU patch do). Neu roi ma van loi, "
              "co the file da bi sua tay - bao lai cho tro ly kiem tra.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak2")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc (truoc patch downsample): {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va downsample thanh cong: {target}")
    print("Chay lai pipeline binh thuong.")


if __name__ == "__main__":
    main()