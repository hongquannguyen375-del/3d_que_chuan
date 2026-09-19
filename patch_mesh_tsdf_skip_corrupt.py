#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_mesh_tsdf_skip_corrupt.py
=================================
Va file 02_mesh_tsdf.py: boc try/except quanh phan doc + tich hop moi khung
hinh, de neu 1 khung bi hong (anh PNG/JPG khong doc duoc - OSError tu PIL,
loi tu cv2, v.v.) thi TU DONG BO QUA khung do va chay tiep, thay vi dung
toan bo pipeline.

Voi so luong khung hong nho (vai khung tren tong hang nghin khung), viec
bo qua khong anh huong dang ke den chat luong mesh cuoi cung, vi du lieu
quet co do chong lap (overlap) rat cao giua cac khung lan can.

CACH DUNG
---------
    python patch_mesh_tsdf_skip_corrupt.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "02_mesh_tsdf.py"

OLD_SNIPPET = '''    n_ok = n_skip = 0
    for k, fid in enumerate(fids):
        depth_path = os.path.join(depth_dir, f"{fid:06d}.png")
        rgb_path   = os.path.join(rgb_dir,   f"{fid:06d}.jpg")
        conf_path  = os.path.join(conf_dir,  f"{fid:06d}.png")

        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            n_skip += 1
            continue

        # confidence map (optional)
        conf = None
        if os.path.isdir(conf_dir) and os.path.exists(conf_path):
            conf = np.array(Image.open(conf_path))

        depth_img = load_depth_img(depth_path, conf, conf_thresh)
        rgb_img   = load_rgb_img(rgb_path)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img, depth_img,
            depth_scale=1.0, depth_trunc=max_depth,
            convert_rgb_to_intensity=False)

        T_CW = np.linalg.inv(poses[fid])
        volume.integrate(rgbd, intrinsics, T_CW)
        n_ok += 1

        if verbose and (k + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    [{k+1}/{len(fids)}] {elapsed:.0f}s", flush=True)

    if verbose:
        print(f"  Integrated {n_ok} frames ({n_skip} skip)", flush=True)'''

NEW_SNIPPET = '''    n_ok = n_skip = n_corrupt = 0
    for k, fid in enumerate(fids):
        depth_path = os.path.join(depth_dir, f"{fid:06d}.png")
        rgb_path   = os.path.join(rgb_dir,   f"{fid:06d}.jpg")
        conf_path  = os.path.join(conf_dir,  f"{fid:06d}.png")

        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            n_skip += 1
            continue

        # Boc try/except: neu anh depth/rgb/confidence bi hong (khong doc
        # duoc), bo qua khung nay va chay tiep thay vi dung ca pipeline.
        try:
            # confidence map (optional)
            conf = None
            if os.path.isdir(conf_dir) and os.path.exists(conf_path):
                conf = np.array(Image.open(conf_path))

            depth_img = load_depth_img(depth_path, conf, conf_thresh)
            rgb_img   = load_rgb_img(rgb_path)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_img, depth_img,
                depth_scale=1.0, depth_trunc=max_depth,
                convert_rgb_to_intensity=False)

            T_CW = np.linalg.inv(poses[fid])
            volume.integrate(rgbd, intrinsics, T_CW)
            n_ok += 1
        except Exception as e:
            n_corrupt += 1
            n_skip += 1
            if n_corrupt <= 30:
                print(f"    CANH BAO: bo qua khung {fid:06d} (anh hong: {e})",
                      flush=True)
            continue

        if verbose and (k + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    [{k+1}/{len(fids)}] {elapsed:.0f}s", flush=True)

    if verbose:
        print(f"  Integrated {n_ok} frames ({n_skip} skip, {n_corrupt} corrupt)",
              flush=True)'''


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print(f"Hay dat file nay cung thu muc voi {TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content:
        print(f"{TARGET_NAME} da duoc va truoc do roi. Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Co the file da bi sua tay truoc do, hoac chua chay "
              "patch_mesh_tsdf_odometry.py truoc (khong sao, patch nay "
              "doc lap voi patch do). Neu van khong khop, bao lai cho tro "
              "ly de kiem tra thu cong - KHONG tu sua tay.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_skipcorrupt")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong: {target}")
    print("Chay lai pipeline binh thuong.")


if __name__ == "__main__":
    main()