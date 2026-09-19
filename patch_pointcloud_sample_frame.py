#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_pointcloud_sample_frame.py
==================================
Va 02_pointcloud_and_mesh.py: doan dau ham create_rgbd_pointcloud() doc
1 khung "mau" (mac dinh la khung DAU TIEN, poses[0]) de xac dinh do phan
giai depth/rgb, chay TRUOC vong lap chinh (da duoc va boi
patch_pointcloud_skip_corrupt.py) nen KHONG duoc bao ve - neu dung khung
dau tien bi hong, script se dung ngay lap tuc du cac khung sau van tot.

Patch nay sua thanh: thu lan luot tung khung cho den khi tim duoc 1 khung
doc duoc ca depth va rgb, moi dung de xac dinh do phan giai.

CACH DUNG
---------
    python patch_pointcloud_sample_frame.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "02_pointcloud_and_mesh.py"

OLD_SNIPPET = '''    # Determine native resolutions
    sample_depth = load_depth(os.path.join(dataset_dir, "depth", f"{poses[0]['frame']}.png"))
    depth_h, depth_w = sample_depth.shape[:2]

    sample_rgb_path = os.path.join(dataset_dir, "rgb", f"{poses[0]['frame']}.jpg")
    sample_rgb = cv2.imread(sample_rgb_path)
    if sample_rgb is None:
        raise FileNotFoundError(
            f"No RGB frame found at {sample_rgb_path}. "
            "Did you run 01_extract_rgb_frames.py first?"
        )
    rgb_h, rgb_w = sample_rgb.shape[:2]'''

NEW_SNIPPET = '''    # Determine native resolutions - thu lan luot tung khung cho den khi
    # tim duoc 1 khung doc duoc ca depth va rgb (khung dau tien co the bi
    # hong, khong nen lam dung ca script chi vi 1 khung mau).
    sample_depth = None
    sample_rgb = None
    for _p in poses:
        try:
            _d = load_depth(os.path.join(dataset_dir, "depth", f"{_p['frame']}.png"))
            _r = cv2.imread(os.path.join(dataset_dir, "rgb", f"{_p['frame']}.jpg"))
            if _r is None:
                continue
            sample_depth, sample_rgb = _d, _r
            break
        except Exception:
            continue
    if sample_depth is None or sample_rgb is None:
        raise RuntimeError(
            "Khong tim duoc khung nao co ca depth va rgb doc duoc de xac "
            "dinh do phan giai. Kiem tra lai du lieu dataset."
        )
    depth_h, depth_w = sample_depth.shape[:2]
    rgb_h, rgb_w = sample_rgb.shape[:2]'''


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print(f"Hay dat file nay cung thu muc voi {TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content:
        print(f"{TARGET_NAME} (sample frame) da duoc va truoc do roi. "
              "Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_sampleframe")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong (sample frame): {target}")
    print("Chay lai pipeline binh thuong.")


if __name__ == "__main__":
    main()