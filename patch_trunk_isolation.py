#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trunk_isolation.py
==========================
Tu dong va file 05_trunk_isolation.py: them buoc loc bo diem co toa do
bat thuong (NaN / Inf / qua xa - vd do file .ply bi loi ghi/doc) NGAY SAU
KHI doc pointcloud.ply, truoc khi chay bat ky phep tinh KDTree/RANSAC nao.

Day la nguyen nhan gay crash STATUS_STACK_BUFFER_OVERRUN (0xC0000409) khi
mot vai diem trong pointcloud.ply co toa do "rac" (vd 1e260 met) khien cac
ham native cua Open3D xu ly sai.

Script se:
  1. Tu dong tim file 05_trunk_isolation.py (cung thu muc voi script nay)
  2. Sao luu ban goc thanh 05_trunk_isolation.py.bak (neu chua co)
  3. Chen doan code loc vao dung vi tri
  4. Bao loi ro rang neu khong tim thay dung doan can thay (vi du file da
     bi sua tay truoc do) - KHONG bao gio ghi de neu khong chac chan dung.

CACH DUNG
---------
    python patch_trunk_isolation.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_trunk_isolation.py"

OLD_SNIPPET = '''    pcd = o3d.io.read_point_cloud(pcd_path)
    n_orig = len(pcd.points)
    print(f"  Loaded {n_orig} points")

    if n_orig == 0:
        print("  SKIP: empty point cloud")
        return False

    # Step 1: Loại bỏ ground plane
    pcd = remove_ground_plane(pcd)'''

NEW_SNIPPET = '''    pcd = o3d.io.read_point_cloud(pcd_path)
    n_orig = len(pcd.points)
    print(f"  Loaded {n_orig} points")

    if n_orig == 0:
        print("  SKIP: empty point cloud")
        return False

    # --- Loc phong ve: bo diem co toa do bat thuong (NaN/Inf/qua xa) ---
    # File .ply doi khi bi loi ghi/doc, tao ra vai diem co toa do "rac"
    # (vi du ~1e260 met) khien cac ham KDTree/RANSAC cua Open3D bi crash
    # (STATUS_STACK_BUFFER_OVERRUN). Loc som o day de tranh hoan toan.
    raw_points = np.asarray(pcd.points)
    SANE_LIMIT = 50.0  # met - khong scan Stray Scanner nao rong hon muc nay
    sane_mask = (
        np.isfinite(raw_points).all(axis=1)
        & (np.abs(raw_points) <= SANE_LIMIT).all(axis=1)
    )
    n_bad = n_orig - int(sane_mask.sum())
    if n_bad > 0:
        print(f"  CANH BAO: loai {n_bad} diem co toa do bat thuong "
              f"(NaN/Inf/qua {SANE_LIMIT}m) truoc khi xu ly.")
        pcd = pcd.select_by_index(np.where(sane_mask)[0])
        n_orig = len(pcd.points)
        if n_orig == 0:
            print("  SKIP: point cloud rong sau khi loc diem bat thuong")
            return False

    # Step 1: Loại bỏ ground plane
    pcd = remove_ground_plane(pcd)'''


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print("Hay dat file patch_trunk_isolation.py nay cung thu muc voi "
              f"{TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content:
        print(f"{TARGET_NAME} da duoc va truoc do roi. Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "de tranh lam hong file. Bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong: {target}")
    print("Chay lai pipeline binh thuong.")


if __name__ == "__main__":
    main()