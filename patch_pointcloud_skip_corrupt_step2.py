#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_pointcloud_skip_corrupt_step2.py
========================================
Va THEM vao 02_pointcloud_and_mesh.py: file nay co 2 giai doan doc anh
rieng biet - STEP 1 "Generating Point Cloud" (ham create_rgbd_pointcloud,
da duoc va boi patch_pointcloud_skip_corrupt.py) va STEP 2 "Generating
TSDF Mesh" (ham generate_mesh_tsdf, CHUA duoc va) - moi ham co vong lap
doc depth/rgb rieng, phai va ca 2 thi moi khong bi dung khi gap anh hong.

Patch nay xu ly rieng ham generate_mesh_tsdf (STEP 2).

CACH DUNG (nen chay SAU patch_pointcloud_skip_corrupt.py)
------------------------------------------------------------
    python patch_pointcloud_skip_corrupt_step2.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "02_pointcloud_and_mesh.py"

OLD_SNIPPET = '''    processed = 0
    for i, pose in enumerate(poses):
        if i % skip_every != 0:
            continue

        frame_id = pose["frame"]
        depth_path = os.path.join(dataset_dir, "depth", f"{frame_id}.png")
        rgb_path = os.path.join(dataset_dir, "rgb", f"{frame_id}.jpg")
        conf_path = os.path.join(dataset_dir, "confidence", f"{frame_id}.png")

        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            continue

        depth = load_depth(depth_path)
        confidence = load_confidence(conf_path)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        mask = (confidence >= confidence_threshold) & (depth > 0) & (depth < max_depth)
        depth[~mask] = 0

        # Upscale depth + downscale RGB to working resolution
        depth_up = cv2.resize(depth, (work_w, work_h),
                              interpolation=cv2.INTER_NEAREST)
        rgb_work = cv2.resize(rgb, (work_w, work_h),
                              interpolation=cv2.INTER_LANCZOS4)

        color_o3d = o3d.geometry.Image(rgb_work.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_up.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )

        T_cam2world = pose_to_extrinsic(pose["position"], pose["quaternion"])
        T_world2cam = np.linalg.inv(T_cam2world)
        volume.integrate(rgbd, intrinsic, T_world2cam)

        # Free per-frame memory immediately
        del depth, confidence, rgb, depth_up, rgb_work
        del color_o3d, depth_o3d, rgbd

        processed += 1
        if processed % 5 == 0 or processed == 1:
            print(f"  Integrated frame {frame_id} ({processed}/{total_to_process})",
                  flush=True)
        # Periodic GC to keep resident memory in check
        if processed % 20 == 0:
            gc.collect()'''

NEW_SNIPPET = '''    processed = 0
    n_corrupt_tsdf = 0
    for i, pose in enumerate(poses):
        if i % skip_every != 0:
            continue

        frame_id = pose["frame"]
        depth_path = os.path.join(dataset_dir, "depth", f"{frame_id}.png")
        rgb_path = os.path.join(dataset_dir, "rgb", f"{frame_id}.jpg")
        conf_path = os.path.join(dataset_dir, "confidence", f"{frame_id}.png")

        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            continue

        # Boc try/except: neu anh depth/rgb/confidence bi hong, bo qua
        # khung nay va chay tiep thay vi dung ca script.
        try:
            depth = load_depth(depth_path)
            confidence = load_confidence(conf_path)
            rgb = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            mask = (confidence >= confidence_threshold) & (depth > 0) & (depth < max_depth)
            depth[~mask] = 0

            # Upscale depth + downscale RGB to working resolution
            depth_up = cv2.resize(depth, (work_w, work_h),
                                  interpolation=cv2.INTER_NEAREST)
            rgb_work = cv2.resize(rgb, (work_w, work_h),
                                  interpolation=cv2.INTER_LANCZOS4)

            color_o3d = o3d.geometry.Image(rgb_work.astype(np.uint8))
            depth_o3d = o3d.geometry.Image(depth_up.astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0,
                depth_trunc=max_depth,
                convert_rgb_to_intensity=False,
            )

            T_cam2world = pose_to_extrinsic(pose["position"], pose["quaternion"])
            T_world2cam = np.linalg.inv(T_cam2world)
            volume.integrate(rgbd, intrinsic, T_world2cam)

            # Free per-frame memory immediately
            del depth, confidence, rgb, depth_up, rgb_work
            del color_o3d, depth_o3d, rgbd
        except Exception as e:
            n_corrupt_tsdf += 1
            if n_corrupt_tsdf <= 30:
                print(f"  CANH BAO: bo qua khung {frame_id} (anh hong: {e})",
                      flush=True)
            continue

        processed += 1
        if processed % 5 == 0 or processed == 1:
            print(f"  Integrated frame {frame_id} ({processed}/{total_to_process})",
                  flush=True)
        # Periodic GC to keep resident memory in check
        if processed % 20 == 0:
            gc.collect()

    if n_corrupt_tsdf > 0:
        print(f"\\nTong so khung bi bo qua vi anh hong (TSDF): {n_corrupt_tsdf}",
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
        print(f"{TARGET_NAME} (STEP 2 - TSDF) da duoc va truoc do roi. "
              "Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_skipcorrupt_step2")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong (STEP 2 - TSDF): {target}")
    print("Chay lai pipeline binh thuong.")


if __name__ == "__main__":
    main()