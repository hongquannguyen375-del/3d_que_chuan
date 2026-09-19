#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_spatial_mismatch.py
==============================
Script chan doan DOC LAP, chi doc du lieu, khong sua file nao trong pipeline.

So sanh vung khong gian (bounding box + trong tam) cua 4 thu:
  1. pointcloud.ply       - point cloud GOC, chua co lap than cay (Buoc 2)
  2. trunk_pointcloud.ply - point cloud SAU khi co lap than cay (Buoc 3)
  3. Quy dao camera       - vi tri camera tu odometry.csv (camera luon o
                            gan than cay dang quet, trong pham vi ~1-3m)
  4. trunk_mesh_final.ply (hoac trunk_mesh_detected.ply neu co) - mesh cuoi

Neu trunk_pointcloud.ply (2) nam XA quy dao camera (3) trong khi mesh (4)
lai o GAN quy dao camera - day la bang chung truc tiep cho thay buoc co
lap than cay (05_trunk_isolation.py) da chon NHAM cum diem (vd nhat nham
1 cay khac/vat the nen o gan do), khong phai loi xoay he truc.

CACH DUNG
---------
    python diagnose_spatial_mismatch.py "D:/Backup/Thucdia-18May2026/cay_0031_1805"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Can cai open3d: pip install open3d==0.19.0")
    sys.exit(1)


def bbox_center(points: np.ndarray):
    if len(points) == 0:
        return None, None, None
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = (mn + mx) / 2
    return mn, mx, center


def report(label: str, points: np.ndarray):
    mn, mx, center = bbox_center(points)
    if mn is None:
        print(f"  {label:28s}: (khong co diem nao)")
        return None
    size = mx - mn
    print(f"  {label:28s}: {len(points):>10,} diem")
    print(f"    {'':28s}  min={mn}")
    print(f"    {'':28s}  max={mx}")
    print(f"    {'':28s}  tam={center}  kich_thuoc={size}")
    return center


def main() -> None:
    if len(sys.argv) != 2:
        print('Cach dung: python diagnose_spatial_mismatch.py "duong/dan/toi/thu_muc_scan"')
        sys.exit(1)

    scan_dir = Path(sys.argv[1]).resolve()
    out_dir = scan_dir / "output"

    print("=" * 70)
    print(f"  Chan doan vung khong gian: {scan_dir}")
    print("=" * 70)
    print()

    centers = {}

    # 1. pointcloud.ply goc
    p1 = out_dir / "pointcloud.ply"
    if p1.exists():
        pcd = o3d.io.read_point_cloud(str(p1))
        pts = np.asarray(pcd.points)
        centers["pointcloud.ply (goc)"] = report("pointcloud.ply (goc, chua co lap)", pts)
    else:
        print(f"  (khong tim thay {p1})")
    print()

    # 2. trunk_pointcloud.ply
    p2 = out_dir / "trunk_pointcloud.ply"
    if p2.exists():
        pcd = o3d.io.read_point_cloud(str(p2))
        pts = np.asarray(pcd.points)
        centers["trunk_pointcloud.ply"] = report("trunk_pointcloud.ply (da co lap)", pts)
    else:
        print(f"  (khong tim thay {p2})")
    print()

    # 3. Quy dao camera tu odometry.csv
    odo_path = scan_dir / "odometry.csv"
    if odo_path.exists():
        import csv
        cam_pos = []
        with open(odo_path, "r", newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                try:
                    cam_pos.append([float(row["x"]), float(row["y"]), float(row["z"])])
                except (KeyError, ValueError):
                    continue
        cam_pos = np.array(cam_pos)
        centers["Quy dao camera"] = report("Quy dao camera (odometry.csv)", cam_pos)
    else:
        print(f"  (khong tim thay {odo_path})")
    print()

    # 4. Mesh cuoi cung
    for mesh_name in ["trunk_mesh_final.ply", "trunk_mesh_detected.ply",
                       "trunk_mesh_trimmed.ply", "trunk_mesh_tsdf.ply"]:
        p4 = out_dir / mesh_name
        if p4.exists():
            mesh = o3d.io.read_triangle_mesh(str(p4))
            verts = np.asarray(mesh.vertices)
            centers[mesh_name] = report(f"{mesh_name} (mesh)", verts)
            break
    else:
        print("  (khong tim thay file mesh nao)")
    print()

    # So sanh khoang cach giua cac trong tam
    print("=" * 70)
    print("  Khoang cach giua cac trong tam (met):")
    print("=" * 70)
    names = list(centers.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c1, c2 = centers[names[i]], centers[names[j]]
            if c1 is None or c2 is None:
                continue
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            flag = "  <-- XA BAT THUONG!" if dist > 1.0 else ""
            print(f"  {names[i]:28s} <-> {names[j]:28s}: {dist:.3f} m{flag}")

    print()
    print("  Ghi chu: camera quet thuong cach than cay dang quet ~0.3-1.5m.")
    print("  Neu trong tam 'trunk_pointcloud.ply' cach 'Quy dao camera' xa hon")
    print("  ~1.5-2m trong khi trong tam mesh lai gan quy dao camera hon nhieu,")
    print("  day la bang chung 05_trunk_isolation.py co the da chon nham cum")
    print("  diem (vd nham sang cay khac/vat the nen) chu khong phai loi xoay.")
    print("=" * 70)


if __name__ == "__main__":
    main()