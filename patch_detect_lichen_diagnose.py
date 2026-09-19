#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_detect_lichen_diagnose.py
=================================
Va 05_detect_lichen.py CHI DE CHAN DOAN (khong doi ket qua phan loai):
in ra phan bo so "phieu bau" white-lichen tren moi vertex cua mesh, de biet
nguong ">= 2/15 hang xom" dang chat toi muc nao so voi du lieu thuc te.

Van de phat hien: PCD-level detection tim thay ~3% diem trang (lichen), nhung
sau buoc "transfer to mesh" (yeu cau >=2/15 hang xom gan nhat cung la trang
VA du sang), ket qua cuoi cung la 0% - toan bo tin hieu bi loc mat. Patch nay
KHONG tu doi nguong, chi in them thong tin de quyet dinh buoc tiep theo.

CACH DUNG
---------
    python patch_detect_lichen_diagnose.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_detect_lichen.py"

OLD_SNIPPET = '''    bright_white = (neighbor_labels == 2) & bright_enough

    vert_labels = np.zeros(len(verts), dtype=np.uint8)
    vert_labels[bright_white.sum(axis=1) >= 2] = 2   # >= 2/15 neighbors (nới lỏng từ 3)
    # Green lichen (label=1) khong su dung — chi giu white lichen (label=2)

    return vert_labels'''

NEW_SNIPPET = '''    bright_white = (neighbor_labels == 2) & bright_enough

    # --- CHAN DOAN: phan bo so phieu bau white tren moi vertex ---
    # Giup biet nguong >=2/15 dang chat toi muc nao so voi du lieu thuc te,
    # KHONG tu doi ket qua phan loai - chi in them thong tin.
    vote_counts = bright_white.sum(axis=1)
    print(f"    [chan doan] Phan bo so 'phieu bau' white tren {len(verts):,} vertex:")
    for v in range(0, 6):
        n_v = int((vote_counts == v).sum())
        print(f"      {v} phieu: {n_v:,} vertex ({100*n_v/len(verts):.2f}%)")
    n_ge1 = int((vote_counts >= 1).sum())
    n_ge2 = int((vote_counts >= 2).sum())
    n_ge3 = int((vote_counts >= 3).sum())
    print(f"      Tong: >=1 phieu={n_ge1:,}  >=2 phieu={n_ge2:,}  >=3 phieu={n_ge3:,}")

    vert_labels = np.zeros(len(verts), dtype=np.uint8)
    vert_labels[bright_white.sum(axis=1) >= 2] = 2   # >= 2/15 neighbors (nới lỏng từ 3)
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
        print(f"{TARGET_NAME} (chan doan) da duoc va truoc do roi. Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_diagnose")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong (chan doan): {target}")
    print("Chay lai buoc detect lichen de xem histogram.")


if __name__ == "__main__":
    main()