#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_run_full_pipeline_force.py
================================
Them co --force cho run_full_pipeline.py.

VAN DE
------
scan_args trong main() HARDCODE "--skip-existing" va truyen no cho MOI script
kieu batch (buoc 4, 4a, 5, 6, 7, 8). Khong co cach nao tat.

Hau qua im lang khi chay lai 43 scan da xu ly bang the he TSDF cu:
  buoc 4  trunk_mesh_poisson.ply     chua co  -> CHAY
  buoc 5  trunk_mesh_recolored.ply   DA CO    -> BO QUA
  buoc 6  trunk_mesh_trimmed.ply     DA CO    -> BO QUA
  buoc 7  trunk_mesh_final.ply       DA CO    -> BO QUA
  buoc 8  trunk_mesh_detected.ply    DA CO    -> BO QUA
Tuc orchestrator dung mesh Poisson xong roi bo do khong dung, va giu nguyen
toan bo ket qua cu dan xuat tu TSDF. Khong co canh bao nao, ma exit code van 0.

Day KHONG phai --resume (co san, loc o phia orchestrator theo expected_outputs).
--skip-existing duoc loc o phia SCRIPT CON, nen --resume co tat cung khong anh
huong gi den no.

SUA
---
--force bo "--skip-existing" khoi scan_args, va ep resume=False, de tat ca cac
buoc chay lai tu dau. Mac dinh khong doi hanh vi cu.

CACH DUNG
---------
    python patch_run_full_pipeline_force.py

Sau khi va, de dung lai mot scan da co ket qua cu:
    python run_full_pipeline.py <scan_dir> --force
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "run_full_pipeline.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- Site 1: CLI flag -------------------------------------------------------
SNIPPETS.append((
    "cli_force",
    '''    parser.add_argument("--resume", action="store_true",
                         help="Bỏ qua bước đã có output, chạy tiếp từ chỗ dừng")''',
    '''    parser.add_argument("--resume", action="store_true",
                         help="Bỏ qua bước đã có output, chạy tiếp từ chỗ dừng")
    parser.add_argument("--force", action="store_true",
                         help="Chạy lại MỌI bước kể cả khi output đã tồn tại. "
                              "Cần cờ này khi dựng lại một scan đã xử lý bằng "
                              "thế hệ TSDF cũ — nếu không, các script con sẽ tự "
                              "bỏ qua vì thấy file cũ còn đó")''',
))

# --- Site 2: scan_args ------------------------------------------------------
SNIPPETS.append((
    "scan_args_force",
    '''    scan_args = ["--scan", scan_name, "--raw-data", str(raw_data_dir),
                 "--skip-existing"]''',
    '''    scan_args = ["--scan", scan_name, "--raw-data", str(raw_data_dir)]
    if args.force:
        # --skip-existing được lọc ở phía SCRIPT CON (không phải ở orchestrator
        # như --resume), nên đây là chỗ duy nhất tắt được nó. Không bỏ cờ này
        # thì mọi bước có output cũ sẽ tự bỏ qua và giữ nguyên kết quả TSDF cũ.
        args.resume = False
        log("  --force: chạy lại mọi bước, bỏ qua output cũ")
    else:
        scan_args.append("--skip-existing")''',
))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if all(new in content for _, _, new in SNIPPETS):
        print(f"{TARGET_NAME} (force flag) da duoc va truoc do roi.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_forceflag")
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
    print("Kiem tra: python -m py_compile run_full_pipeline.py")
    print("Chay lai: python run_full_pipeline.py <scan_dir> --force")


if __name__ == "__main__":
    main()
