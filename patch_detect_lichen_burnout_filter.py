#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_detect_lichen_burnout_filter.py
=======================================
Va 05_detect_lichen.py: them bo loc loai tru "vung chay sang" (burnout) -
diem anh bi phan chieu anh sang truc tiep/nang choi qua manh, khien do
sang V gan bao hoa cam bien (V~1.0) va mat het thong tin mau (S~0).

Nguyen ly: dia y trang that, du sang, la mot be mat vat ly co ket cau/mau
sac tu nhien - hiem khi "chay" den muc tuyet doi (V cuc cao VA S cuc thap
DONG THOI) tren dien rong. Nguoc lai, phan chieu anh sang truc tiep (nang
xuyen tan la, be mat uot phan chieu) thuong day ca V va S ve gan cuc tri
tuyet doi do bao hoa cam bien camera.

QUAN TRONG - day la bo loc PHONG NGUA MOT PHAN, khong giai quyet triet de:
  - Khong the phan biet 100% giua "chay sang that" va "dia y rat trang,
    rat sach" chi tu 1 pixel don le - ca 2 co the trung dac trung V/S.
  - Nguong mac dinh (BURNOUT_V=0.95, BURNOUT_S=0.08) la UOC LUONG BAN DAU,
    CAN HIEU CHINH thuc nghiem tren du lieu that (xem phan "CAN THEM
    THONG TIN" o cuoi file nay).

CACH DUNG
---------
    python patch_detect_lichen_burnout_filter.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_detect_lichen.py"

OLD_SNIPPET = '''    is_veg = (h >= 65) & (h <= 150) & (s > 0.25)'''

NEW_SNIPPET = '''    is_veg = (h >= 65) & (h <= 150) & (s > 0.25)

    # --- Loc vung "chay sang" (burnout: phan chieu anh sang truc tiep/nang choi) ---
    # V gan bao hoa cam bien VA S gan 0 DONG THOI - dac trung cua vung anh bi
    # "chay" do phan xa anh sang manh, khac voi dia y trang that (be mat vat
    # ly co ket cau, hiem khi dat ca 2 cuc tri nay cung luc tren dien rong).
    # LUU Y: nguong nay la uoc luong ban dau, can hieu chinh thuc nghiem.
    BURNOUT_V = 0.95   # do sang toi thieu de coi la "chay" (thang [0,1])
    BURNOUT_S = 0.08   # do bao hoa toi da de coi la "chay" (thang [0,1])
    is_burnout = (v >= BURNOUT_V) & (s <= BURNOUT_S)
    n_burnout = int(is_burnout.sum())
    if n_burnout > 0:
        print(f"      [burnout] Loai {n_burnout:,} diem ({100*n_burnout/len(cols):.1f}%) "
              f"nghi la vung chay sang (V>={BURNOUT_V}, S<={BURNOUT_S})", flush=True)'''

OLD_SNIPPET2 = '''    white = local_white | abs_white'''

NEW_SNIPPET2 = '''    white = (local_white | abs_white) & ~is_burnout'''


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print(f"Hay dat file nay cung thu muc voi {TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_SNIPPET in content and NEW_SNIPPET2 in content:
        print(f"{TARGET_NAME} (loc burnout) da duoc va truoc do roi. Khong can lam gi them.")
        return

    if OLD_SNIPPET not in content or OLD_SNIPPET2 not in content:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay trong "
              f"{TARGET_NAME}.")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_burnout")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    new_content = new_content.replace(OLD_SNIPPET2, NEW_SNIPPET2)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong (loc burnout): {target}")
    print("Chay lai buoc detect lichen de xem ket qua moi.")


if __name__ == "__main__":
    main()
