#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trunk_isolation_camera_anchor_z.py
===========================================
Mo rong patch_trunk_isolation_camera_anchor.py: bo sung gioi han theo
TRUC Z (chieu cao), khong chi XY (ngang).

Ly do: da phat hien qua du lieu that (cay_0007_1805) mot dang loi khac -
cum diem SAI nam CUNG vi tri ngang (XY) voi than cay that, nhung o DO CAO
(Z) hoan toan khac (vd canh cay o tren cao, hoac vat the khac o do cao
khac). Ban vá XY-only truoc do khong bat duoc truong hop nay.

Theo cach quay cua nguoi dung (camera di quanh than cay, DONG THOI dua
len xuong doc chieu dai than), pham vi Z (chieu cao) ma camera da di qua
chinh la pham vi chieu cao that cua doan than cay duoc quet. Dung thong
tin nay de loai bo diem o do cao qua xa pham vi camera.

CACH DUNG (chay SAU patch_trunk_isolation_camera_anchor.py)
-----------------------------------------------------------------
    python patch_trunk_isolation_camera_anchor_z.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_trunk_isolation.py"

OLD_SNIPPET = '''    import csv as _csv
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
              f"theo quy dao camera.")'''

NEW_SNIPPET = '''    import csv as _csv
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


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print(f"Hay dat file nay cung thu muc voi {TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content:
        print(f"{TARGET_NAME} (neo camera + Z) da duoc va truoc do roi. Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Kiem tra da chay patch_trunk_isolation_camera_anchor.py truoc "
              "chua (patch nay can chay SAU patch do, thay the hoan toan "
              "logic cu). Neu roi ma van khong khop, co the file da bi sua "
              "tay - bao lai cho tro ly kiem tra.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_cameraanchor_z")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong (neo camera + Z): {target}")
    print("Chay lai buoc co lap than cay (buoc 3/8) de kiem tra ket qua.")


if __name__ == "__main__":
    main()
