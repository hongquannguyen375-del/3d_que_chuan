#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_detect_lichen_drop_absolute.py
======================================
Va 05_detect_lichen.py: LOAI BO abs_white (nguong sang tuyet doi tinh
chung ca cay), CHI DUNG local_white (so sanh diem voi 40 hang xom khong
gian gan nhat).

CAN CU: da xac nhan bang chan doan thuc te tren cay_0007_1805:
  - local_white rieng : 1.7% tong so diem
  - abs_white rieng   : 9.3% tong so diem (gap hon 5 lan local_white)
abs_white dung nguong TOAN CUC (tinh tu percentile 20 do sang ca cay), nen
KHONG biet duoc neu than cay co chenh lech anh sang that theo do cao (vd
tan la thua gan mat dat -> phan duoi sang hon tu nhien, khong phai dia y).
local_white so sanh CUC BO (40 hang xom gan nhat trong khong gian) nen tu
nhien ben vung hon truoc kieu chenh lech anh sang tren dien rong nay.

Voi ~12% nhieu nen tu abs_white, dù dùng nguong phieu bau bao nhieu
(>=1 hay >=2) cung deu bi khuech dai thanh ket qua sai lech nghiem trong
o muc mesh (da do duoc 76% dia y "ao" tren mot cay nhieu kha nang khong
he co nhieu dia y den vay). Bo abs_white de giai quyet dung goc re, thay
vi tiep tuc dieu chinh nguong phieu bau.

CACH DUNG (chay SAU cac patch chan doan/nguong/burnout truoc do)
---------------------------------------------------------------------
    python patch_detect_lichen_drop_absolute.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_detect_lichen.py"

OLD_SNIPPET = '''    white = (local_white | abs_white) & ~is_burnout'''

NEW_SNIPPET = '''    # Bo abs_white (nguong toan cuc) - da xac nhan qua chan doan thuc te la
    # nguyen nhan chinh gay over-detection khi than cay co chenh lech anh
    # sang that theo do cao (tan la thua/day khac nhau). local_white (so
    # sanh cuc bo voi hang xom khong gian) ben vung hon nhieu truoc kieu
    # chenh lech nay. abs_v/abs_s van duoc tinh o tren de khong pha vo
    # cau truc code, chi khong con dung de xet nhan trang nua.
    white = local_white & ~is_burnout'''


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print(f"Hay dat file nay cung thu muc voi {TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content:
        print(f"{TARGET_NAME} (bo abs_white) da duoc va truoc do roi. Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Kiem tra da chay patch_detect_lichen_burnout_filter.py va "
              "patch_detect_lichen_diagnose4.py truoc chua. Neu roi ma van "
              "khong khop, co the file da bi sua tay - bao lai cho tro ly "
              "kiem tra.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_dropabs")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong (bo abs_white): {target}")
    print("Chay lai buoc detect lichen de xem ket qua moi.")
    print()
    print("KHUYEN NGHI: nen thu ca 2 nguong phieu bau (>=1 va >=2) sau khi")
    print("bo abs_white, vi voi nhieu nen giam manh, nguong >=2 (chat hon,")
    print("it false positive hon) co the phu hop hon >=1.")


if __name__ == "__main__":
    main()
