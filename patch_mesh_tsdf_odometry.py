#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_mesh_tsdf_odometry.py
=============================
Va file 02_mesh_tsdf.py: sua ham load_odometry() de CHI doc dung 9 cot dau
tien can dung (timestamp, frame, x, y, z, qx, qy, qz, qw) tu odometry.csv,
bo qua hoan toan cac cot thua phia sau (du chung co gia tri hay bi loi).

Loi goc: "could not convert string ' ' to float64 at row 0, column 14"
xay ra vi np.loadtxt() mac dinh co gang doc TAT CA cac cot tren moi dong.
Neu odometry.csv co nhieu hon 9 cot (vi du app quet xuat them cot phu),
va mot trong cac cot thua do bi rong/loi, ca dong se khong doc duoc - du
cac cot 0-8 ma script thuc su can deu hop le.

CACH DUNG
---------
    python patch_mesh_tsdf_odometry.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "02_mesh_tsdf.py"

OLD_SNIPPET = '''    data  = np.loadtxt(os.path.join(scan_dir, "odometry.csv"),
                       delimiter=",", skiprows=1)'''

NEW_SNIPPET = '''    # Chi doc 9 cot dau (timestamp,frame,x,y,z,qx,qy,qz,qw) - bo qua cot
    # thua phia sau du chung co bi rong/loi (vd app quet xuat them cot phu
    # ma script nay khong dung toi).
    data  = np.loadtxt(os.path.join(scan_dir, "odometry.csv"),
                       delimiter=",", skiprows=1, usecols=range(9))'''


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

    backup = target.with_suffix(target.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    new_content = content.replace(OLD_SNIPPET, NEW_SNIPPET)
    target.write_text(new_content, encoding="utf-8")
    print(f"Da va thanh cong: {target}")
    print("Chay lai pipeline binh thuong.")


if __name__ == "__main__":
    main()