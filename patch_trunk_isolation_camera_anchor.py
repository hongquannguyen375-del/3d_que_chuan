#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trunk_isolation_camera_anchor.py
========================================
Va 05_trunk_isolation.py: them buoc cat so bo point cloud theo ban kinh
quanh TRONG TAM QUY DAO CAMERA (tu odometry.csv), thuc hien SOM (truoc
ca RANSAC ground removal va uoc tinh truc than cay).

Ly do: da xac nhan qua chan doan (diagnose_spatial_mismatch.py) rang
"estimate_trunk_axis" (dung trung vi 40% diem cao nhat) co the bi keo
lech sang mot cum diem SAI (vi du cay khac/vat the nen o gan do) neu
cum do vo tinh chiem phan lon vung "cao nhat" cua toan canh. Trong khi
do, camera LUON o gan than cay dang quet that su (da do duoc: mesh that
chi cach quy dao camera ~0.5-0.8m, trong khi cum sai co the cach toi
~2m). Cat truoc theo ban kinh quanh camera loai bo hoan toan nguy co
chon nham cum o xa.

CACH DUNG (chay SAU patch_trunk_isolation.py va
patch_trunk_isolation_downsample.py)
------------------------------------------------------------------------
    python patch_trunk_isolation_camera_anchor.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_trunk_isolation.py"

OLD_SNIPPET = '''        if n_orig == 0:
            print("  SKIP: point cloud rong sau khi loc diem bat thuong")
            return False

    # --- Downsample som de giam RAM/thoi gian cho RANSAC/KDTree/DBSCAN ---'''

NEW_SNIPPET = '''        if n_orig == 0:
            print("  SKIP: point cloud rong sau khi loc diem bat thuong")
            return False

    # --- Cat so bo theo quy dao camera (odometry.csv) ---
    # Da xac nhan qua chan doan: camera luon o gan than cay that su dang
    # quet (~0.5-0.8m), trong khi buoc uoc tinh truc than cay phia sau co
    # the bi keo lech sang cum diem SAI (vd cay khac/vat the nen) neu cum
    # do chiem phan lon vung "cao nhat" cua toan canh. Cat truoc theo ban
    # kinh quanh tam quy dao camera de loai tru hoan toan nguy co nay.
    import csv as _csv
    CAMERA_ANCHOR_RADIUS = 2.0  # met - rong rai, chi de loai cum o xa han
    odo_path = os.path.join(dataset_dir, "odometry.csv")
    if os.path.exists(odo_path):
        cam_xy = []
        try:
            with open(odo_path, "r", newline="") as _f:
                _reader = _csv.DictReader(_f, skipinitialspace=True)
                for _row in _reader:
                    try:
                        cam_xy.append([float(_row["x"]), float(_row["y"])])
                    except (KeyError, ValueError):
                        continue
        except Exception as e:
            cam_xy = []
            print(f"  CANH BAO: khong doc duoc odometry.csv de neo camera: {e}")
        if cam_xy:
            import numpy as _np
            cam_xy = _np.array(cam_xy)
            cam_center = cam_xy.mean(axis=0)
            pts_xy = np.asarray(pcd.points)[:, :2]
            dist_to_cam = np.sqrt(((pts_xy - cam_center) ** 2).sum(axis=1))
            anchor_mask = dist_to_cam <= CAMERA_ANCHOR_RADIUS
            n_before_anchor = len(pcd.points)
            n_kept_anchor = int(anchor_mask.sum())
            if n_kept_anchor > 0:
                pcd = pcd.select_by_index(np.where(anchor_mask)[0])
                n_orig = len(pcd.points)
                print(f"  Neo theo quy dao camera (r={CAMERA_ANCHOR_RADIUS}m quanh "
                      f"tam camera XY={cam_center}): {n_before_anchor:,} -> "
                      f"{n_orig:,} pts")
            else:
                print(f"  CANH BAO: khong con diem nao trong ban kinh "
                      f"{CAMERA_ANCHOR_RADIUS}m quanh camera - giu nguyen "
                      f"point cloud, bo qua buoc neo camera.")
    else:
        print(f"  CANH BAO: khong tim thay {odo_path}, bo qua buoc neo "
              f"theo quy dao camera.")

    # --- Downsample som de giam RAM/thoi gian cho RANSAC/KDTree/DBSCAN ---'''


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print(f"Hay dat file nay cung thu muc voi {TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content:
        print(f"{TARGET_NAME} (neo camera) da duoc va truoc do roi. Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Kiem tra da chay patch_trunk_isolation.py va "
              "patch_trunk_isolation_downsample.py truoc chua (patch nay "
              "can chay SAU 2 patch do). Neu roi ma van khong khop, co the "
              "file da bi sua tay - bao lai cho tro ly kiem tra.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_cameraanchor")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong (neo camera): {target}")
    print("Chay lai buoc co lap than cay (buoc 3/8) de kiem tra ket qua.")


if __name__ == "__main__":
    main()