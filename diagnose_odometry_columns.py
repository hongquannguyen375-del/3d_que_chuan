#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_odometry_columns.py
==============================
Script chan doan DOC LAP, khong dung chung/sua doi bat ky file nao trong
pipeline chinh. Muc dich: tim nguyen nhan tai sao trunk_pointcloud.ply va
mesh TSDF lai nam o 2 he truc khac nhau cho mot so scan.

Kiem tra 2 dieu:
  1. In ra HEADER THAT cua odometry.csv (ten cot + thu tu that su trong
     file) - so sanh voi gia dinh "timestamp,frame,x,y,z,qx,qy,qz,qw" ma
     02_mesh_tsdf.py dang dung khi doc bang np.loadtxt(usecols=range(9)).
  2. Doc CUNG 1 dong du lieu bang 2 CACH:
     a) Theo TEN cot (giong 02_pointcloud_and_mesh.py dung csv.DictReader)
     b) Theo VI TRI cot co dinh 0-8 (giong 02_mesh_tsdf.py dung np.loadtxt)
     Neu 2 ket qua quaternion/position KHAC NHAU -> xac nhan dung nguyen
     nhan: thu tu cot thuc te khong khop voi gia dinh vi tri co dinh.

CACH DUNG
---------
    python diagnose_odometry_columns.py "D:/Backup/Thucdia-18May2026/cay_0031_1805"
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    if len(sys.argv) != 2:
        print('Cach dung: python diagnose_odometry_columns.py "duong/dan/toi/thu_muc_scan"')
        sys.exit(1)

    scan_dir = Path(sys.argv[1]).resolve()
    odo_path = scan_dir / "odometry.csv"
    if not odo_path.exists():
        print(f"Khong tim thay: {odo_path}")
        sys.exit(1)

    print("=" * 70)
    print(f"  Kiem tra: {odo_path}")
    print("=" * 70)

    # ---- 1. In header that ----------------------------------------------
    with open(odo_path, "r", newline="") as f:
        raw_header_line = f.readline().rstrip("\n").rstrip("\r")
    raw_fields = raw_header_line.split(",")
    print(f"\n  Header THAT trong file (theo dung thu tu, {len(raw_fields)} cot):")
    for i, name in enumerate(raw_fields):
        print(f"    Cot {i:2d}: {name!r}")

    assumed = ["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw"]
    print(f"\n  Thu tu 02_mesh_tsdf.py GIA DINH (khong kiem tra lai):")
    for i, name in enumerate(assumed):
        print(f"    Cot {i:2d}: {name!r}")

    # So sanh ten cot (bo khoang trang, khong phan biet hoa/thuong) giua
    # header that va gia dinh, tung vi tri mot.
    print(f"\n  So sanh tung vi tri (header that vs gia dinh):")
    mismatch_found = False
    for i in range(min(len(raw_fields), len(assumed))):
        real_clean = raw_fields[i].strip().lower()
        assumed_clean = assumed[i].strip().lower()
        match = "OK" if real_clean == assumed_clean else "!! KHAC NHAU !!"
        if real_clean != assumed_clean:
            mismatch_found = True
        print(f"    Cot {i:2d}: that={raw_fields[i]!r:15s} gia_dinh={assumed[i]!r:12s}  {match}")

    if mismatch_found:
        print("\n  ==> PHAT HIEN LECH THU TU COT. Day rat co the la nguyen nhan")
        print("      khien 02_mesh_tsdf.py doc nham gia tri quaternion/position,")
        print("      dan den mesh TSDF bi xoay/lech so voi trunk_pointcloud.ply.")
    else:
        print("\n  ==> Thu tu cot khop voi gia dinh. Van de co the o cho khac,")
        print("      xem tiep phan so sanh pose cu the ben duoi.")

    # ---- 2. So sanh pose cu the cho vai dong dau ------------------------
    print(f"\n{'='*70}")
    print("  So sanh POSE cua vai dong dau, doc theo TEN cot vs VI TRI cot")
    print(f"{'='*70}")

    with open(odo_path, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        rows_by_name = []
        for i, row in enumerate(reader):
            if i >= 3:
                break
            try:
                rows_by_name.append({
                    "frame": row.get("frame", row.get(" frame")),
                    "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]),
                    "qx": float(row["qx"]), "qy": float(row["qy"]),
                    "qz": float(row["qz"]), "qw": float(row["qw"]),
                })
            except KeyError as e:
                print(f"  LOI doc theo ten cot: thieu cot {e}. "
                      "Header co the dung ten khac (vd co khoang trang).")
                rows_by_name.append(None)

    raw_data = np.loadtxt(str(odo_path), delimiter=",", skiprows=1,
                          usecols=range(9), max_rows=3)

    for i in range(min(3, len(raw_data))):
        print(f"\n  --- Dong du lieu {i} ---")
        by_name = rows_by_name[i] if i < len(rows_by_name) else None
        if by_name:
            print(f"    [Theo TEN cot]  frame={by_name['frame']}  "
                  f"pos=({by_name['x']:.4f}, {by_name['y']:.4f}, {by_name['z']:.4f})  "
                  f"quat=({by_name['qx']:.4f}, {by_name['qy']:.4f}, "
                  f"{by_name['qz']:.4f}, {by_name['qw']:.4f})")
        row_pos = raw_data[i]
        print(f"    [Theo VI TRI]   frame={int(row_pos[1])}  "
              f"pos=({row_pos[2]:.4f}, {row_pos[3]:.4f}, {row_pos[4]:.4f})  "
              f"quat=({row_pos[5]:.4f}, {row_pos[6]:.4f}, "
              f"{row_pos[7]:.4f}, {row_pos[8]:.4f})")

        if by_name:
            pos_diff = abs(by_name['x'] - row_pos[2]) + abs(by_name['y'] - row_pos[3]) + abs(by_name['z'] - row_pos[4])
            quat_diff = (abs(by_name['qx'] - row_pos[5]) + abs(by_name['qy'] - row_pos[6])
                        + abs(by_name['qz'] - row_pos[7]) + abs(by_name['qw'] - row_pos[8]))
            if pos_diff > 1e-4 or quat_diff > 1e-4:
                print(f"    ==> LECH! pos_diff={pos_diff:.6f}  quat_diff={quat_diff:.6f}")
            else:
                print(f"    ==> Khop nhau (sai so nho).")

    print(f"\n{'='*70}")
    print("  XONG. Gui toan bo output nay cho tro ly de phan tich.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()