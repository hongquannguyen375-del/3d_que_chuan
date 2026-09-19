#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_run_full_pipeline_poisson.py
==================================
Va run_full_pipeline.py cho dong bo voi cac ban va thang 9/2026.

BA VAN DE DUOC SUA
------------------
1) Orchestrator KHONG BAO GIO goi 02_mesh_poisson_trunk.py (buoc 4b).
   Hau qua im lang: chay orchestrator cho mot cay moi se khong sinh ra
   trunk_mesh_poisson.ply, nen 03_recolor_mesh.py (da va boi
   patch_recolor_prefer_poisson.py) tu dong LUI VE trunk_mesh_tsdf.ply --
   tuc hinh hoc van la TSDF, lo thung quay lai, va khong co canh bao nao.

   Do duoc tren cay_0004/0005/0007 (kich thuoc ngang X x Z):
     trunk_pointcloud.ply    0.20x0.16   0.13x0.22   0.51x0.68 m
     trunk_mesh_tsdf.ply     1.92x1.85   1.27x2.18   2.67x2.65 m   <- ca canh
     trunk_mesh_poisson.ply  0.19x0.13   0.11x0.20   0.30x0.52 m
   TSDF dung lai hinh hoc TU ANH THO nen khong ke thua bat ky ban va nao cua
   05_trunk_isolation.py; no mesh ca cay ben canh va mat dat, roi buoc 6 moi
   phai got xuong bang trunk_pointcloud.ply lam khuon. Khuon got duoc phan
   thua nhung KHONG dap lai duoc be mat TSDF chua tung dung (xem CLAUDE.md
   muc 12 item 15).

   Sua: buoc 4 gio la 02_mesh_poisson_trunk.py -> trunk_mesh_poisson.ply.

2) TSDF van ton ~5 phut/cay cho mot file khong con ai dung o luong mac dinh.
   Sua: chi chay khi co co --with-tsdf (van giu nguyen 02_mesh_tsdf.py tren
   dia, va 03_recolor_mesh.py van giu duong lui + co --no-poisson).

3) SCRIPT_MODULE_CHECKS co khoa "05_trunk_isolation.p" -- THIEU CHU 'y'.
   check_script_modules() tra cuu bang script_path.name tuc
   "05_trunk_isolation.py", nen .get() luon tra None va buoc kiem tra module
   tien-chay cua buoc 3 bi bo qua im lang. Vo hai tren thuc te (numpy/open3d
   da duoc kiem o buoc 2) nhung la mot dong chet.
   Sua: dung ten file, va them khoa cho 02_mesh_poisson_trunk.py.

CACH DUNG
---------
    python patch_run_full_pipeline_poisson.py

Sau khi va:
    python run_full_pipeline.py <scan_dir>                  # Poisson (mac dinh)
    python run_full_pipeline.py <scan_dir> --with-tsdf      # chay them TSDF
    python run_full_pipeline.py <scan_dir> --poisson-depth 8
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "run_full_pipeline.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- Site 1: SCRIPT_MODULE_CHECKS -----------------------------------------
SNIPPETS.append((
    "module_checks",
    '''    "05_trunk_isolation.p": ["numpy", "open3d"],
    "02_mesh_tsdf.py": ["numpy", "open3d", "PIL", "scipy"],''',
    '''    "05_trunk_isolation.py": ["numpy", "open3d"],
    "02_mesh_tsdf.py": ["numpy", "open3d", "PIL", "scipy"],
    "02_mesh_poisson_trunk.py": ["numpy", "open3d"],''',
))

# --- Site 2: CLI flags ------------------------------------------------------
SNIPPETS.append((
    "cli_flags",
    '''    parser.add_argument("--with-slope", action="store_true",
                         help="Chạy thêm bước phân tích độ dốc địa hình")''',
    '''    parser.add_argument("--with-slope", action="store_true",
                         help="Chạy thêm bước phân tích độ dốc địa hình")
    parser.add_argument("--with-tsdf", action="store_true",
                         help="Chạy thêm 02_mesh_tsdf.py (bước 4a). Mặc định KHÔNG "
                              "chạy: hình học nay lấy từ Poisson, mesh TSDF không "
                              "còn được dùng ở bước nào (xem CLAUDE.md §12 item 15)")
    parser.add_argument("--poisson-depth", type=int, default=7,
                         help="Độ sâu Poisson cho 02_mesh_poisson_trunk.py "
                              "(mặc định 7 — đã chọn bằng số đo, xem CLAUDE.md §5)")''',
))

# --- Site 3: step 4 ---------------------------------------------------------
SNIPPETS.append((
    "step4_poisson",
    '''        # Bước 4: depth/+rgb/+camera_matrix.csv+odometry.csv -> trunk_mesh_tsdf.ply
        check_script_modules(args.python, scripts_dir / "02_mesh_tsdf.py")
        run_step(
            "4/8", "Dựng mesh chất lượng cao bằng TSDF (02_mesh_tsdf.py)",
            args.python, scripts_dir / "02_mesh_tsdf.py",
            scan_args, scripts_dir,
            expected_outputs=[out_dir / "trunk_mesh_tsdf.ply"],
            resume=args.resume,
        )

        # Bước 5: trunk_mesh_tsdf.ply + trunk_pointcloud.ply -> trunk_mesh_recolored.ply''',
    '''        # Bước 4a (tuỳ chọn): depth/+rgb/+odometry.csv -> trunk_mesh_tsdf.ply
        # Không chạy mặc định. Mesh TSDF dựng lại hình học TỪ ẢNH THÔ nên không
        # kế thừa bất kỳ bản vá nào của 05_trunk_isolation.py — nó mesh cả cảnh
        # (đo được: rộng 1.3–2.7m so với thân cây 0.13–0.68m) và có lỗ thủng trên
        # đúng phần thân đã được quét. Giữ lại chỉ để so sánh/đối chiếu.
        if args.with_tsdf:
            check_script_modules(args.python, scripts_dir / "02_mesh_tsdf.py")
            run_step(
                "4a/8", "Dựng mesh TSDF để đối chiếu (02_mesh_tsdf.py)",
                args.python, scripts_dir / "02_mesh_tsdf.py",
                scan_args, scripts_dir,
                expected_outputs=[out_dir / "trunk_mesh_tsdf.ply"],
                resume=args.resume,
            )

        # Bước 4: trunk_pointcloud.ply -> trunk_mesh_poisson.ply
        # Đây là nguồn HÌNH HỌC hiện tại. Dựng thẳng từ point cloud đã được
        # 05_trunk_isolation.py cô lập, nên hình học và màu cùng một nguồn và
        # bước 6 không còn phải hoà giải hai thứ khác nhau (đo được: bước 6 nay
        # chỉ tỉa 3.2–3.5% số đỉnh).
        check_script_modules(args.python, scripts_dir / "02_mesh_poisson_trunk.py")
        run_step(
            "4/8", "Dựng mesh thân cây bằng Poisson (02_mesh_poisson_trunk.py)",
            args.python, scripts_dir / "02_mesh_poisson_trunk.py",
            scan_args + ["--depth", str(args.poisson_depth)], scripts_dir,
            expected_outputs=[out_dir / "trunk_mesh_poisson.ply"],
            resume=args.resume,
        )

        # Bước 5: trunk_mesh_poisson.ply + trunk_pointcloud.ply -> trunk_mesh_recolored.ply''',
))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if all(new in content for _, _, new in SNIPPETS):
        print(f"{TARGET_NAME} (poisson step) da duoc va truoc do roi.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Dung doan, dung sua tay. Kiem tra lai file goc truoc.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_poissonstep")
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
    print(f"Da va thanh cong {len(applied)} doan: {applied}")
    print("")
    print("Kiem tra:  python -m py_compile run_full_pipeline.py")
    print("Chay    :  python run_full_pipeline.py <scan_dir>")
    print("Doi chieu: python run_full_pipeline.py <scan_dir> --with-tsdf")


if __name__ == "__main__":
    main()
