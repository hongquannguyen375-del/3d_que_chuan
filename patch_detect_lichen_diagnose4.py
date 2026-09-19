#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_detect_lichen_diagnose4.py
==================================
Va THEM 05_detect_lichen.py: tach rieng dong gop cua 2 phuong phap phat
hien mau trong ham classify_lichen_local_contrast:
  - local_white: so sanh diem voi 40 hang xom KHONG GIAN gan nhat (cuc bo)
  - abs_white  : so sanh voi nguong sang TUYET DOI tinh chung ca cay (global)

Gia thuyet can kiem chung: neu than cay co chenh lech anh sang ro ret theo
DO CAO (vd tan la thua gan mat dat -> sang hon tu nhien), abs_white (dung
nguong toan cuc, khong biet den chenh lech theo do cao) se bi kich hoat
qua muc rong khap, trong khi local_white (so sanh cuc bo) it bi anh huong
hon. Neu dung, huong sua se la GIAM TRONG SO hoac BO han abs_white, chi
dung local_white.

CACH DUNG (chay SAU cac patch chan doan/nguong/burnout truoc do)
---------------------------------------------------------------------
    python patch_detect_lichen_diagnose4.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_detect_lichen.py"

OLD_SNIPPET = '''    white = (local_white | abs_white) & ~is_burnout'''

NEW_SNIPPET = '''    # --- CHAN DOAN 4: tach rieng dong gop cua local_white vs abs_white ---
    n_local_only = int((local_white & ~abs_white).sum())
    n_abs_only   = int((abs_white & ~local_white).sum())
    n_both       = int((local_white & abs_white).sum())
    n_total_pts  = len(cols)
    print(f"      [chan doan 4] Diem trang theo local_white rieng : "
          f"{n_local_only:,} ({100*n_local_only/n_total_pts:.1f}%)")
    print(f"      [chan doan 4] Diem trang theo abs_white rieng   : "
          f"{n_abs_only:,} ({100*n_abs_only/n_total_pts:.1f}%)")
    print(f"      [chan doan 4] Diem trang ca 2 phuong phap dong y: "
          f"{n_both:,} ({100*n_both/n_total_pts:.1f}%)")
    if n_abs_only > n_local_only * 2 and n_abs_only > 0.05 * n_total_pts:
        print(f"      [chan doan 4] => abs_white (nguong toan cuc) co ve la "
              f"nguyen nhan chinh gay over-detection, khong phai local_white.")

    white = (local_white | abs_white) & ~is_burnout'''


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print(f"Hay dat file nay cung thu muc voi {TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content:
        print(f"{TARGET_NAME} (chan doan 4) da duoc va truoc do roi. Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Kiem tra da chay patch_detect_lichen_burnout_filter.py truoc "
              "chua (patch nay can chay SAU patch do). Neu roi ma van khong "
              "khop, co the file da bi sua tay - bao lai cho tro ly kiem tra.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_diagnose4")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong (chan doan 4): {target}")
    print("Chay lai buoc detect lichen de xem ket qua chan doan moi.")


if __name__ == "__main__":
    main()
