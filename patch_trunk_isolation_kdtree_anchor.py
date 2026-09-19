#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trunk_isolation_kdtree_anchor.py
========================================
Va 05_trunk_isolation.py: THAY THE khoi neo camera hien tai (tam trung binh
XY + dai Z rong, tu patch_trunk_isolation_camera_anchor_z.py) bang mot cach
neo dang huong hon: tinh khoang cach Euclid 3D THAT SU tu moi diem den
CAMERA POSE GAN NHAT (dung cKDTree tren toan bo quy dao camera), thay vi
tach rieng "ngang" (XY, tam trung binh) va "cao" (dai Z).

Ly do: cach cu tach truc "ngang" va "cao" theo dung 2 cot co dinh (0,1 cho
tam ngang; 2 cho dai cao) - day chinh la nguyen nhan can 05_trunk_isolation.py
bi loi nham truc len/xuong that su cua du lieu (da xac nhan: Y=cot 1, khong
phai Z=cot 2 - xem patch_trunk_isolation_up_axis.py). Cach moi (cKDTree 3D)
khong tach truc nao ca - tinh khoang cach that trong khong gian 3D toi
camera pose gan nhat - nen KHONG PHU THUOC vao truc nao la "cao", tranh hoan
toan lop loi nay o buoc neo camera. Thuat toan giong het ham
filter_by_camera_distance() da co san (nhung khong dung) trong
01_clean_pointcloud.py.

Da duoc ghi nhan (CLAUDE.md) la sua duoc 2 ca loi thuc te:
  - cay_0031_1805: chon nham cum theo chieu ngang
  - cay_0007_1805: chon nham cum cung XY nhung khac do cao (Z)
Ca hai deu duoc xu ly bang cach nay vi no khong tach rieng "ngang" va "cao".

CACH DUNG
---------
    python patch_trunk_isolation_kdtree_anchor.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_trunk_isolation.py"

OLD_SNIPPET = '''    # --- Cat so bo theo quy dao camera (odometry.csv) ---
    # Da xac nhan qua chan doan: camera luon o gan than cay that su dang
    # quet (~0.5-0.8m), trong khi buoc uoc tinh truc than cay phia sau co
    # the bi keo lech sang cum diem SAI (vd cay khac/vat the nen) neu cum
    # do chiem phan lon vung "cao nhat" cua toan canh. Cat truoc theo ban
    # kinh quanh tam quy dao camera de loai tru hoan toan nguy co nay.
    import csv as _csv
    CAMERA_ANCHOR_RADIUS = 2.0  # met - rong rai, chi de loai cum o xa han
    Z_MARGIN = 0.5  # met - bien do them ngoai pham vi Z camera da di qua
    odo_path = os.path.join(dataset_dir, "odometry.csv")
    if os.path.exists(odo_path):
        cam_xyz = []
        try:
            with open(odo_path, "r", newline="") as _f:
                _reader = _csv.DictReader(_f, skipinitialspace=True)
                for _row in _reader:
                    try:
                        cam_xyz.append([float(_row["x"]), float(_row["y"]), float(_row["z"])])
                    except (KeyError, ValueError):
                        continue
        except Exception as e:
            cam_xyz = []
            print(f"  CANH BAO: khong doc duoc odometry.csv de neo camera: {e}")
        if cam_xyz:
            import numpy as _np
            cam_xyz = _np.array(cam_xyz)
            cam_center = cam_xyz[:, :2].mean(axis=0)
            z_min_cam = cam_xyz[:, 2].min() - Z_MARGIN
            z_max_cam = cam_xyz[:, 2].max() + Z_MARGIN
            all_pts = np.asarray(pcd.points)
            pts_xy = all_pts[:, :2]
            pts_z = all_pts[:, 2]
            dist_to_cam = np.sqrt(((pts_xy - cam_center) ** 2).sum(axis=1))
            xy_mask = dist_to_cam <= CAMERA_ANCHOR_RADIUS
            z_mask = (pts_z >= z_min_cam) & (pts_z <= z_max_cam)
            anchor_mask = xy_mask & z_mask
            n_before_anchor = len(pcd.points)
            n_kept_anchor = int(anchor_mask.sum())
            if n_kept_anchor > 0:
                pcd = pcd.select_by_index(np.where(anchor_mask)[0])
                n_orig = len(pcd.points)
                print(f"  Neo theo quy dao camera (r={CAMERA_ANCHOR_RADIUS}m quanh "
                      f"tam XY={cam_center}, Z trong [{z_min_cam:.3f}, "
                      f"{z_max_cam:.3f}]): {n_before_anchor:,} -> {n_orig:,} pts")
            else:
                print(f"  CANH BAO: khong con diem nao trong pham vi neo camera "
                      f"(XY r={CAMERA_ANCHOR_RADIUS}m, Z margin={Z_MARGIN}m) - "
                      f"giu nguyen point cloud, bo qua buoc neo camera.")
    else:
        print(f"  CANH BAO: khong tim thay {odo_path}, bo qua buoc neo "
              f"theo quy dao camera.")'''

NEW_SNIPPET = '''    # --- Neo theo quy dao camera bang cKDTree 3D dang huong ---
    # Tinh khoang cach Euclid 3D that su tu moi diem den camera pose GAN
    # NHAT (khong phai mot tam trung binh don, khong tach rieng "ngang" va
    # "cao"), dua tren toan bo quy dao camera. Vi khong tach truc nao ca,
    # cach nay KHONG PHU THUOC vao truc nao la chieu cao trong du lieu -
    # tranh hoan toan lop loi da gap voi cach cu (xem patch_trunk_isolation
    # _up_axis.py). Giong het filter_by_camera_distance() trong
    # 01_clean_pointcloud.py. Da xac nhan qua 2 ca loi thuc te:
    # cay_0031_1805 (chon nham cum theo chieu ngang) va cay_0007_1805
    # (chon nham cum cung XY nhung khac do cao) - ca hai deu duoc xu ly.
    from scipy.spatial import cKDTree as _cKDTree
    import csv as _csv
    CAMERA_ANCHOR_RADIUS = 1.0  # met - toi camera pose gan nhat
    odo_path = os.path.join(dataset_dir, "odometry.csv")
    if os.path.exists(odo_path):
        cam_xyz = []
        try:
            with open(odo_path, "r", newline="") as _f:
                _reader = _csv.DictReader(_f, skipinitialspace=True)
                for _row in _reader:
                    try:
                        cam_xyz.append([float(_row["x"]), float(_row["y"]), float(_row["z"])])
                    except (KeyError, ValueError):
                        continue
        except Exception as e:
            cam_xyz = []
            print(f"  CANH BAO: khong doc duoc odometry.csv de neo camera: {e}")
        if cam_xyz:
            cam_xyz = np.array(cam_xyz, dtype=np.float32)
            all_pts = np.asarray(pcd.points, dtype=np.float32)
            tree = _cKDTree(cam_xyz)
            dist_to_cam, _ = tree.query(all_pts, k=1, workers=-1)
            anchor_mask = dist_to_cam <= CAMERA_ANCHOR_RADIUS
            n_before_anchor = len(pcd.points)
            n_kept_anchor = int(anchor_mask.sum())
            if n_kept_anchor > 0:
                pcd = pcd.select_by_index(np.where(anchor_mask)[0])
                n_orig = len(pcd.points)
                print(f"  Neo theo quy dao camera (cKDTree 3D, r={CAMERA_ANCHOR_RADIUS}m "
                      f"toi camera pose gan nhat): {n_before_anchor:,} -> {n_orig:,} pts")
            else:
                print(f"  CANH BAO: khong con diem nao trong ban kinh "
                      f"{CAMERA_ANCHOR_RADIUS}m toi bat ky camera pose nao - "
                      f"giu nguyen point cloud, bo qua buoc neo camera.")
    else:
        print(f"  CANH BAO: khong tim thay {odo_path}, bo qua buoc neo "
              f"theo quy dao camera.")'''


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print(f"Hay dat file nay cung thu muc voi {TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content:
        print(f"{TARGET_NAME} (kdtree anchor) da duoc va truoc do roi. Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Co the file da bi sua tay truoc do, hoac cac patch neo camera "
              "cu chua duoc ap dung dung thu tu. Dung lai, khong sua gi, de "
              "tranh lam hong file. Bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_kdtreeanchor")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong (kdtree anchor): {target}")
    print("Chay lai buoc co lap than cay (buoc 3/8) de kiem tra ket qua.")


if __name__ == "__main__":
    main()
