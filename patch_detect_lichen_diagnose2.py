#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_detect_lichen_diagnose2.py
==================================
Va THEM 05_detect_lichen.py (chay SAU patch_detect_lichen_diagnose.py va
patch_detect_lichen_relax_threshold.py): tach rieng anh huong cua bo loc
"du sang" (bright_enough) khoi viec chi don thuan co nhan trang gan do,
de biet chinh xac buoc nao dang loai bo tin hieu dia y.

In them:
  - So vertex co >=1 lang gieng nhan trang (KHONG can du sang) - "raw"
  - So vertex co >=1 lang gieng nhan trang VA du sang - "sau loc sang"
So sanh 2 con so nay cho biet bo loc do sang (bright_enough) co phai la
nguyen nhan chinh khong.

CACH DUNG (chay SAU 2 patch chan doan/noi nguong truoc)
-----------------------------------------------------------
    python patch_detect_lichen_diagnose2.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_detect_lichen.py"

OLD_SNIPPET = '''    bright_white = (neighbor_labels == 2) & bright_enough

    # --- CHAN DOAN: phan bo so phieu bau white tren moi vertex ---'''

NEW_SNIPPET = '''    bright_white = (neighbor_labels == 2) & bright_enough

    # --- CHAN DOAN 2: tach rieng anh huong cua bo loc do sang ---
    raw_white = (neighbor_labels == 2)  # KHONG ap dung bright_enough
    n_raw_ge1 = int((raw_white.sum(axis=1) >= 1).sum())
    n_filtered_ge1 = int((bright_white.sum(axis=1) >= 1).sum())
    print(f"    [chan doan 2] Vertex co >=1 lang gieng nhan trang: "
          f"raw={n_raw_ge1:,}  sau loc do sang={n_filtered_ge1:,}")
    if n_raw_ge1 > 0 and n_filtered_ge1 == 0:
        print(f"    [chan doan 2] => Bo loc 'du sang' (bright_enough, "
              f">=75% max V lang gieng) dang loai SACH tin hieu dia y.")

    # --- CHAN DOAN: phan bo so phieu bau white tren moi vertex ---'''


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print(f"Hay dat file nay cung thu muc voi {TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content:
        print(f"{TARGET_NAME} (chan doan 2) da duoc va truoc do roi. Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Kiem tra da chay patch_detect_lichen_diagnose.py truoc chua "
              "(patch nay can chay SAU patch do). Neu roi ma van khong "
              "khop, co the file da bi sua tay - bao lai cho tro ly kiem tra.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_diagnose2")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong (chan doan 2): {target}")
    print("Chay lai buoc detect lichen de xem ket qua chan doan moi.")


if __name__ == "__main__":
    main()