#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_pointcloud.py
====================
Chan doan nhanh file output/pointcloud.ply cua 1 scan: dem so diem, kiem tra
co toa do NaN/Inf hay khong, va uoc luong RAM can de xu ly - de biet truoc
05_trunk_isolation.py bi crash do DU LIEU HONG hay do HET RAM.

Script CHI DOC, khong tinh toan nang (khong goi KDTree/DBSCAN), nen chay
nhanh (vai giay den vai chuc giay) va an toan, khong lam may bi treo.

CACH DUNG
---------
    python check_pointcloud.py "E:/Thucdia-18May2026/cay_0003_1805"
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import numpy as np
    import open3d as o3d
except ImportError as e:
    print(f"Thieu thu vien: {e}")
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        print('Cach dung: python check_pointcloud.py "duong/dan/toi/thu_muc_scan"')
        sys.exit(1)

    scan_dir = Path(sys.argv[1]).resolve()
    pcd_path = scan_dir / "output" / "pointcloud.ply"

    if not pcd_path.exists():
        print(f"Khong tim thay file: {pcd_path}")
        sys.exit(1)

    size_mb = pcd_path.stat().st_size / (1024 * 1024)
    print("=" * 70)
    print(f"  File: {pcd_path}")
    print(f"  Kich thuoc file: {size_mb:.1f} MB")
    print("=" * 70)

    print("\n  Dang doc file (chi doc, khong tinh toan)...")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)
    n = len(points)

    print(f"\n  So diem: {n:,}")
    if n == 0:
        print("  CANH BAO: point cloud rong!")
        return

    # Uoc luong RAM can (Open3D + numpy + cac buffer trung gian khi loc/KDTree
    # thuong can gap 5-10 lan kich thuoc du lieu goc)
    bytes_per_point = 3 * 8  # float64 x,y,z trong numpy
    raw_mb = n * bytes_per_point / (1024 * 1024)
    print(f"  RAM toi thieu cho mang toa do (numpy): ~{raw_mb:.0f} MB")
    print(f"  RAM uoc tinh khi chay KDTree/SOR/DBSCAN: ~{raw_mb*8:.0f}-{raw_mb*15:.0f} MB "
          f"(Open3D thuong can gap 8-15 lan)")

    # Kiem tra NaN / Inf
    nan_mask = np.isnan(points).any(axis=1)
    inf_mask = np.isinf(points).any(axis=1)
    n_nan = int(nan_mask.sum())
    n_inf = int(inf_mask.sum())

    print(f"\n  Diem co toa do NaN : {n_nan:,} ({100*n_nan/n:.4f}%)")
    print(f"  Diem co toa do Inf : {n_inf:,} ({100*n_inf/n:.4f}%)")

    # Bounding box cua cac diem HOP LE (bo NaN/Inf) de xem co outlier khong
    valid_mask = ~(nan_mask | inf_mask)
    valid_pts = points[valid_mask]
    if len(valid_pts) > 0:
        mins = valid_pts.min(axis=0)
        maxs = valid_pts.max(axis=0)
        print(f"\n  Bounding box (diem hop le, met):")
        print(f"    X: {mins[0]:.3f} -> {maxs[0]:.3f}  (rong {maxs[0]-mins[0]:.3f} m)")
        print(f"    Y: {mins[1]:.3f} -> {maxs[1]:.3f}  (rong {maxs[1]-mins[1]:.3f} m)")
        print(f"    Z: {mins[2]:.3f} -> {maxs[2]:.3f}  (rong {maxs[2]-mins[2]:.3f} m)")
        # Canh bao neu bounding box qua lon bat thuong cho 1 than cay
        # (than cay thuong chi rong vai chuc cm, cao vai met)
        if (maxs[0]-mins[0]) > 50 or (maxs[1]-mins[1]) > 50 or (maxs[2]-mins[2]) > 50:
            print("\n  CANH BAO: bounding box qua 50m o mot chieu - co diem outlier"
                  " o rat xa, du lieu depth/pose co the bi loi o mot so khung hinh.")

    print("\n" + "=" * 70)
    print("  KET LUAN:")
    if n_nan > 0 or n_inf > 0:
        print(f"  -> CO {n_nan + n_inf:,} diem toa do khong hop le (NaN/Inf)."
              " Day co the la nguyen nhan lam crash 05_trunk_isolation.py.")
    if n > 5_000_000:
        print(f"  -> Point cloud RAT LON ({n:,} diem). Voi may 8GB RAM,"
              " cac buoc KDTree/SOR/DBSCAN co nguy co het bo nho.")
    if n_nan == 0 and n_inf == 0 and n <= 5_000_000:
        print("  -> Du lieu toa do hop le, so luong diem o muc binh thuong."
              " Nguyen nhan crash co the khac (RAM tai thoi diem chay, dung"
              " luong Open3D/DBSCAN cu the).")
    print("=" * 70)


if __name__ == "__main__":
    main()