#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_pointcloud_skip_corrupt.py
==================================
Va file 02_pointcloud_and_mesh.py: boc try/except quanh phan doc + xu ly
moi khung hinh, de neu 1 khung bi hong (anh depth/rgb/confidence khong doc
duoc) thi TU DONG BO QUA khung do va chay tiep, thay vi dung ca script
(loi goc: FileNotFoundError "Cannot read depth image" tu cv2.imread tra
ve None khi anh bi hong).

CACH DUNG
---------
    python patch_pointcloud_skip_corrupt.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "02_pointcloud_and_mesh.py"

OLD_SNIPPET = '''    combined_pcd = o3d.geometry.PointCloud()
    camera_positions = []

    for i, pose in enumerate(poses):
        if i % skip_every != 0:
            continue

        frame_id = pose["frame"]
        depth_path = os.path.join(dataset_dir, "depth", f"{frame_id}.png")
        rgb_path = os.path.join(dataset_dir, "rgb", f"{frame_id}.jpg")
        conf_path = os.path.join(dataset_dir, "confidence", f"{frame_id}.png")

        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            continue

        # Load data
        depth = load_depth(depth_path)
        confidence = load_confidence(conf_path)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Mask out low-confidence and far pixels at native depth res
        mask = (confidence >= confidence_threshold) & (depth > 0) & (depth < max_depth)
        depth[~mask] = 0

        # Upscale depth & mask to working resolution (NEAREST to avoid
        # blending depth values across edges)
        depth_up = cv2.resize(depth, (work_w, work_h),
                              interpolation=cv2.INTER_NEAREST)
        # Downscale RGB to working resolution (LANCZOS for sharp colour)
        rgb_work = cv2.resize(rgb, (work_w, work_h),
                              interpolation=cv2.INTER_LANCZOS4)

        # Create RGBD image at working resolution
        color_o3d = o3d.geometry.Image(rgb_work.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_up.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        T_cam2world = pose_to_extrinsic(pose["position"], pose["quaternion"])
        pcd.transform(T_cam2world)

        combined_pcd += pcd
        camera_positions.append(pose["position"])

        del depth, confidence, rgb, depth_up, rgb_work
        del color_o3d, depth_o3d, rgbd, pcd

        if (i // skip_every) % 20 == 0:
            print(f"  Processed frame {frame_id} ({i+1}/{len(poses)})")
            gc.collect()

    # Downsample — use a finer voxel to keep more detail
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.003)
    print(f"\\nCombined point cloud: {len(combined_pcd.points)} points")'''

NEW_SNIPPET = '''    combined_pcd = o3d.geometry.PointCloud()
    camera_positions = []
    n_corrupt_frames = 0

    for i, pose in enumerate(poses):
        if i % skip_every != 0:
            continue

        frame_id = pose["frame"]
        depth_path = os.path.join(dataset_dir, "depth", f"{frame_id}.png")
        rgb_path = os.path.join(dataset_dir, "rgb", f"{frame_id}.jpg")
        conf_path = os.path.join(dataset_dir, "confidence", f"{frame_id}.png")

        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            continue

        # Boc try/except: neu anh depth/rgb/confidence bi hong (khong doc
        # duoc), bo qua khung nay va chay tiep thay vi dung ca script.
        try:
            # Load data
            depth = load_depth(depth_path)
            confidence = load_confidence(conf_path)
            rgb = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # Mask out low-confidence and far pixels at native depth res
            mask = (confidence >= confidence_threshold) & (depth > 0) & (depth < max_depth)
            depth[~mask] = 0

            # Upscale depth & mask to working resolution (NEAREST to avoid
            # blending depth values across edges)
            depth_up = cv2.resize(depth, (work_w, work_h),
                                  interpolation=cv2.INTER_NEAREST)
            # Downscale RGB to working resolution (LANCZOS for sharp colour)
            rgb_work = cv2.resize(rgb, (work_w, work_h),
                                  interpolation=cv2.INTER_LANCZOS4)

            # Create RGBD image at working resolution
            color_o3d = o3d.geometry.Image(rgb_work.astype(np.uint8))
            depth_o3d = o3d.geometry.Image(depth_up.astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0,
                depth_trunc=max_depth,
                convert_rgb_to_intensity=False,
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

            T_cam2world = pose_to_extrinsic(pose["position"], pose["quaternion"])
            pcd.transform(T_cam2world)

            combined_pcd += pcd
            camera_positions.append(pose["position"])

            del depth, confidence, rgb, depth_up, rgb_work
            del color_o3d, depth_o3d, rgbd, pcd
        except Exception as e:
            n_corrupt_frames += 1
            if n_corrupt_frames <= 30:
                print(f"  CANH BAO: bo qua khung {frame_id} (anh hong: {e})")
            continue

        if (i // skip_every) % 20 == 0:
            print(f"  Processed frame {frame_id} ({i+1}/{len(poses)})")
            gc.collect()

    if n_corrupt_frames > 0:
        print(f"\\nTong so khung bi bo qua vi anh hong: {n_corrupt_frames}")

    # Downsample — use a finer voxel to keep more detail
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.003)
    print(f"\\nCombined point cloud: {len(combined_pcd.points)} points")'''


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
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "bao lai cho tro ly de kiem tra thu cong.")
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