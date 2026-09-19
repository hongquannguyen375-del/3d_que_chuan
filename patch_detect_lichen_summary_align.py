#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_detect_lichen_summary_align.py
====================================
Va 05_detect_lichen.py: bang tom tat cuoi che do --batch gan SAI TEN CAY cho so
lieu, va lang le bo mat cac cay o cuoi danh sach.

LOI
---
    all_stats = []
    for i, scan_dir in enumerate(dirs):
        ...
        if r["ok"]:
            all_stats.append(r["stats"])     # <- CHI khi thanh cong, KHONG kem ten
        else:
            print(f"  FAIL ...")
    ...
    for i, (scan_dir, st) in enumerate(zip(dirs, all_stats)):   # <- 51 ghep voi 46
        name = os.path.basename(scan_dir)

`dirs` co DU moi thu muc scan (51), `all_stats` chi co cac scan CHAY DUOC (46).
zip() ghep theo VI TRI, nen ke tu cay FAIL dau tien moi hang deu mang so lieu
cua cay KE TIEP, va 5 cay cuoi bi zip cat mat khoi bang.

DO DUOC TREN CORPUS THAT (46/51, 3 cay khong co pointcloud.ply):
    cay_0006_1805      11816      20.6       43.0        30.6   <- so cua cay_0007
    cay_0007_1805       9308      15.1       22.6        18.9   <- so cua cay_0008
trong khi cay_0006_1805 con KHONG CO lichen_stats.json (no FAIL), va
lichen_stats.json that cua cay_0007 ghi total_cm2=11816.4, lichen=30.55%.

File lichen_stats.json cua TUNG CAY VAN DUNG (moi file tu mang "scan": <ten>).
Chi bao cao muc corpus la sai. Nhung day la thu duoc doc de so sanh giua cac
cay, nen no du de dan toi ket luan sai ve mot cay cu the.

lichen_summary.json (dong 614) con te hon: no dump thang `all_stats`, tuc mot
DANH SACH TRAN khong co ten cay nao ca -- khong the ghep nguoc ve scan. Da
grep: khong file nao trong repo doc file nay, nen doi cau truc la an toan.

SUA
---
  1. Luu kem ten: all_stats.append((name, r["stats"]))
  2. Duyet thang all_stats, khong zip voi dirs nua
  3. lichen_summary.json ghi list cac {"scan": ..., "stats": ...}
  4. Noi rong cot ten 12 -> 16 (ten that dai 13 ky tu, dang tran cot)

CACH DUNG
---------
    python patch_detect_lichen_summary_align.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_detect_lichen.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- Site 1: luu kem ten ----------------------------------------------------
SNIPPETS.append((
    "append_with_name",
    '''            ok_count += 1
            all_stats.append(r["stats"])''',
    '''            ok_count += 1
            # [Sep 2026] Luu KEM TEN. Truoc day chi append r["stats"], roi bang
            # tom tat zip(dirs, all_stats) -- ma dirs co ca cac scan FAIL nen
            # ten va so lieu lech nhau tu cay FAIL dau tien tro di.
            all_stats.append((name, r["stats"]))''',
))

# --- Site 2: bang tom tat ---------------------------------------------------
SNIPPETS.append((
    "summary_table",
    '''        print(f"{'Scan':<12} {'Total_cm2':>10} "
              f"{'UP_lich%':>9} {'DOWN_lich%':>10} {'TOTAL_lich%':>11}")
        for i, (scan_dir, st) in enumerate(zip(dirs, all_stats)):
            name = os.path.basename(scan_dir)
            print(f"{name:<12} {st['overall']['total_cm2']:>10.0f} "''',
    '''        print(f"{'Scan':<16} {'Total_cm2':>10} "
              f"{'UP_lich%':>9} {'DOWN_lich%':>10} {'TOTAL_lich%':>11}")
        for name, st in all_stats:
            print(f"{name:<16} {st['overall']['total_cm2']:>10.0f} "''',
))

# --- Site 3: lichen_summary.json -------------------------------------------
SNIPPETS.append((
    "summary_json",
    '''    with open(os.path.join(raw_data_dir, "lichen_summary.json"), "w") as f:
        json.dump(all_stats, f, indent=2)''',
    '''    # [Sep 2026] Ghi kem ten scan. Truoc day dump thang mot danh sach tran,
    # khong the ghep nguoc ve tung cay.
    with open(os.path.join(raw_data_dir, "lichen_summary.json"), "w") as f:
        json.dump([{"scan": n, "stats": s} for n, s in all_stats], f, indent=2)''',
))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if all(new in content for _, _, new in SNIPPETS):
        print(f"{TARGET_NAME} (summary align) da duoc va truoc do roi.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_summaryalign")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    applied = []
    for label, old, new in SNIPPETS:
        if new in content:
            continue
        content = content.replace(old, new, 1)
        applied.append(label)

    target.write_text(content, encoding="utf-8")
    print(f"Da va thanh cong {len(applied)} doan (summary align): {applied}")
    print("")
    print("Kiem tra: python -m py_compile 05_detect_lichen.py")
    print("Chay lai: python 05_detect_lichen.py --all --raw-data <DIR>")


if __name__ == "__main__":
    main()
