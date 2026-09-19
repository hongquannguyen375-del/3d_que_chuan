#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_detect_lichen_relax_threshold.py
========================================
Va 05_detect_lichen.py: noi nguong "phieu bau" white-lichen tu >=2/15 hang
xom xuong >=1/15. CHAY SAU patch_detect_lichen_diagnose.py.

Ly do: sau khi trunk_pointcloud.ply bi nen xuong voxel 1cm (de tranh treo
may/het RAM o buoc co lap than cay), moi mang dia y nho hon ~1cm chi con
DUNG 1 diem dai dien duy nhat. Vi vay khong bao gio co the co 2 diem trang
gan nhau de dat nguong >=2 - day la gioi han cau truc, da duoc xac nhan
qua histogram chan doan (0% vertex dat >=2 phieu, trong khi ly thuyet
ngau nhien du kien ~9% neu diem phan bo ngau nhien). Noi xuong >=1 de
khong mat tin hieu dia y that.

CACH DUNG (chay SAU patch_detect_lichen_diagnose.py)
-------------------------------------------------------
    python patch_detect_lichen_relax_threshold.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_detect_lichen.py"

OLD_SNIPPET = '''    vert_labels = np.zeros(len(verts), dtype=np.uint8)
    vert_labels[bright_white.sum(axis=1) >= 2] = 2   # >= 2/15 neighbors (nới lỏng từ 3)
    # Green lichen (label=1) khong su dung — chi giu white lichen (label=2)

    return vert_labels'''

NEW_SNIPPET = '''    # Noi long tu >=2 xuong >=1/15: sau downsample 1cm o buoc co lap than
    # cay, mang dia y nho hon 1cm chi con 1 diem dai dien - nguong >=2 la
    # khong the dat duoc ve mat cau truc (da xac nhan qua histogram chan
    # doan: 0% vertex dat >=2 phieu du ly thuyet ngau nhien du kien ~9%).
    vert_labels = np.zeros(len(verts), dtype=np.uint8)
    vert_labels[bright_white.sum(axis=1) >= 1] = 2   # >= 1/15 neighbors (noi tu >=2)
    # Green lichen (label=1) khong su dung — chi giu white lichen (label=2)

    return vert_labels'''


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print(f"Hay dat file nay cung thu muc voi {TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content:
        print(f"{TARGET_NAME} (nguong >=1) da duoc va truoc do roi. Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Kiem tra da chay patch_detect_lichen_diagnose.py truoc chua "
              "(patch nay can chay SAU patch do). Neu roi ma van khong "
              "khop, co the file da bi sua tay - bao lai cho tro ly kiem tra.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_relaxthreshold")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong (nguong >=1): {target}")
    print("Chay lai buoc detect lichen de xem ket qua moi.")


if __name__ == "__main__":
    main()