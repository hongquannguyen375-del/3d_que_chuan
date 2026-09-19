#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_pipeline.py
=====================
Chạy TOÀN BỘ pipeline xử lý 1 scan cây quế, từ dữ liệu thô (rgb.mp4, depth/,
camera_matrix.csv, odometry.csv, imu.csv, location.csv) cho đến file mesh 3D
cuối cùng đã tô màu + phát hiện địa y.

Script này KHÔNG viết lại logic xử lý — nó chỉ gọi lần lượt (subprocess) các
file .py đã có sẵn trong dự án 3D_Que_lichen theo đúng thứ tự, kiểm tra input
đầu vào trước khi chạy, và kiểm tra output sau mỗi bước để phát hiện lỗi sớm
thay vì chạy hết rồi mới biết bị hỏng ở bước nào.

CÁCH DÙNG
---------
1. Copy file này vào đúng thư mục gốc của repo 3D_Que_lichen (cùng cấp với
   các file 01_..., 02_..., 03_... v.v.)
2. Chạy:
       python run_full_pipeline.py "duong/dan/toi/thu_muc_scan"

   Ví dụ (Windows):
       python run_full_pipeline.py "E:\\3D_Que\\Raw_data\\26Q1"

   Thư mục scan phải chứa: rgb.mp4, depth/, camera_matrix.csv, odometry.csv,
   imu.csv, location.csv  (đây là dữ liệu thô xuất ra từ app quét).

3. Kết quả cuối cùng sẽ nằm ở:
       <thu_muc_scan>/output/trunk_mesh_detected.ply   <-- MODEL 3D CUỐI CÙNG
       <thu_muc_scan>/output/lichen_stats.json         <-- thống kê địa y

TÙY CHỌN
--------
  --scripts-dir DIR   Thư mục chứa các script 01_..09_... (mặc định: cùng
                       thư mục với file run_full_pipeline.py này)
  --trunk-radius R    Bán kính (m) quanh trục thân cây khi cô lập point cloud
                       (mặc định 0.30, tăng lên nếu thân cây to hơn)
  --skip-lichen       Dừng lại ở bước làm mịn (trunk_mesh_final.ply), không
                       chạy bước phát hiện địa y
  --with-slope        Chạy thêm bước phân tích độ dốc (06_slope_analysis.py)
  --resume            Bỏ qua các bước mà file output đã tồn tại (chạy tiếp
                       từ chỗ bị dừng thay vì làm lại từ đầu)
  --python PATH       Đường dẫn tới trình thông dịch python muốn dùng để gọi
                       các script con (mặc định: dùng cùng python đang chạy
                       file này)

Nếu 1 bước bị lỗi, script sẽ DỪNG NGAY, in rõ bước nào lỗi, lý do, và toàn bộ
log (stdout+stderr) của bước đó để bạn biết chính xác cần sửa gì, thay vì cố
chạy tiếp với dữ liệu hỏng.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Cấu hình chung
# --------------------------------------------------------------------------- #

REQUIRED_RAW_FILES = [
    "rgb.mp4",
    "camera_matrix.csv",
    "odometry.csv",
    "imu.csv",
    "location.csv",
]
REQUIRED_RAW_DIRS = ["depth"]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def banner(title: str) -> None:
    log()
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)


def die(msg: str) -> None:
    log()
    log("!" * 70)
    log(f"  DỪNG PIPELINE: {msg}")
    log("!" * 70)
    sys.exit(1)


# --------------------------------------------------------------------------- #
#  Kiểm tra dữ liệu đầu vào trước khi chạy bất cứ thứ gì
# --------------------------------------------------------------------------- #

def check_raw_data(scan_dir: Path) -> None:
    banner("BƯỚC 0/8 — Kiểm tra dữ liệu thô đầu vào")

    missing = []
    for f in REQUIRED_RAW_FILES:
        p = scan_dir / f
        if not p.exists():
            missing.append(f)
        else:
            log(f"  [OK] {f}  ({p.stat().st_size:,} bytes)")

    for d in REQUIRED_RAW_DIRS:
        p = scan_dir / d
        if not p.exists() or not p.is_dir():
            missing.append(f"{d}/")
        else:
            n = len(list(p.glob("*.png")))
            log(f"  [OK] {d}/  ({n} file .png)")

    if missing:
        die(
            "Thiếu các file/thư mục bắt buộc trong "
            f"'{scan_dir}':\n    - " + "\n    - ".join(missing) +
            "\n  Kiểm tra lại bạn đã trỏ đúng thư mục scan, và đã giải nén "
            "đầy đủ dữ liệu (không bị thiếu do nén/gửi file)."
        )

    # Cảnh báo sớm nếu số lượng depth lệch nhiều so với odometry — không chặn
    # (01_extract_rgb_frames.py tự xử lý sai lệch nhỏ), chỉ cảnh báo để biết trước.
    try:
        n_depth = len(list((scan_dir / "depth").glob("*.png")))
        with open(scan_dir / "odometry.csv", newline="") as f:
            n_odom = sum(1 for _ in csv.reader(f)) - 1
        gap = abs(n_depth - n_odom)
        if gap > 2:
            log(
                f"  [CẢNH BÁO] Số khung depth ({n_depth}) và số dòng "
                f"odometry.csv ({n_odom}) lệch nhau {gap} khung. "
                "Nếu lệch nhiều, bước trích RGB hoặc TSDF phía sau có thể báo lỗi."
            )
    except Exception:
        pass  # không quan trọng bằng việc chạy tiếp

    log("\n  Dữ liệu đầu vào hợp lệ, bắt đầu chạy pipeline.")


# --------------------------------------------------------------------------- #
#  Hạ tầng chạy từng bước (subprocess) + kiểm tra output
# --------------------------------------------------------------------------- #

class StepFailed(Exception):
    pass


def run_step(
    step_no: str,
    title: str,
    python_exe: str,
    script_path: Path,
    args: list[str],
    cwd: Path,
    expected_outputs: list[Path],
    resume: bool,
) -> None:
    banner(f"BƯỚC {step_no} — {title}")

    if not script_path.exists():
        die(
            f"Không tìm thấy script '{script_path.name}' trong "
            f"'{script_path.parent}'. Kiểm tra lại --scripts-dir."
        )

    if resume and expected_outputs and all(p.exists() for p in expected_outputs):
        log(f"  [BỎ QUA] Output đã tồn tại (--resume): "
            + ", ".join(p.name for p in expected_outputs))
        return

    cmd = [python_exe, str(script_path)] + args
    log("  Lệnh chạy: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    log()

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    elapsed = time.time() - t0
    output_text = proc.stdout or ""
    log(output_text.rstrip())
    log(f"\n  (chạy hết {elapsed:.1f}s, mã thoát = {proc.returncode})")

    if proc.returncode != 0:
        raise StepFailed(
            f"Script '{script_path.name}' trả về mã lỗi {proc.returncode}."
        )

    missing_outputs = [p for p in expected_outputs if not p.exists()]
    if missing_outputs:
        raise StepFailed(
            f"Script '{script_path.name}' chạy xong (mã thoát 0) nhưng KHÔNG "
            "thấy file output mong đợi:\n    - "
            + "\n    - ".join(str(p) for p in missing_outputs)
            + "\n  Xem log phía trên để biết lý do (có thể do dữ liệu đầu vào "
            "không đạt yêu cầu tối thiểu, ví dụ point cloud quá ít điểm)."
        )

    for p in expected_outputs:
        log(f"  [OK] Đã tạo: {p}  ({p.stat().st_size:,} bytes)")


SCRIPT_MODULE_CHECKS = {
    "01_extract_rgb_frames.py": [],
    "02_pointcloud_and_mesh.py": ["numpy", "cv2", "open3d"],
    "05_trunk_isolation.py": ["numpy", "open3d"],
    "02_mesh_tsdf.py": ["numpy", "open3d", "PIL", "scipy"],
    "02_mesh_poisson_trunk.py": ["numpy", "open3d"],
    "03_recolor_mesh.py": ["numpy", "open3d", "scipy"],
    "04_trim_mesh.py": ["numpy", "open3d", "scipy"],
    "04b_finalize_mesh.py": ["numpy", "open3d"],
    "05_detect_lichen.py": ["numpy", "open3d", "scipy"],
    "06_slope_analysis.py": ["numpy", "open3d"],
}


def check_import_module(python_exe: str, module_name: str) -> None:
    """Verify the chosen Python interpreter can import a required module."""
    proc = subprocess.run(
        [python_exe, "-c", f"import {module_name}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        die(
            f"Không thể import module Python '{module_name}' từ '{python_exe}'.\n"
            "Môi trường ảo có thể bị thiếu hoặc bị hỏng.\n"
            "Hãy chạy lại: pip install -r requirements.txt\n\n"
            f"Output:\n{proc.stdout}"
        )


def check_script_modules(python_exe: str, script_path: Path) -> None:
    """Check required modules before running a step script."""
    modules = SCRIPT_MODULE_CHECKS.get(script_path.name)
    if not modules:
        return
    for module in modules:
        check_import_module(python_exe, module)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy toàn bộ pipeline 3D thân cây quế từ dữ liệu thô "
                     "đến model 3D cuối cùng.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scan_dir", help="Đường dẫn tới thư mục scan (chứa rgb.mp4, depth/, ...)")
    parser.add_argument("--scripts-dir", default=None,
                         help="Thư mục chứa các script 01_..09_... "
                              "(mặc định: cùng thư mục với file này)")
    parser.add_argument("--trunk-radius", type=float, default=0.30,
                         help="Bán kính (m) cô lập thân cây (mặc định 0.30)")
    parser.add_argument("--skip-lichen", action="store_true",
                         help="Dừng ở bước làm mịn, không phát hiện địa y")
    parser.add_argument("--with-slope", action="store_true",
                         help="Chạy thêm bước phân tích độ dốc địa hình")
    parser.add_argument("--with-tsdf", action="store_true",
                         help="Chạy thêm 02_mesh_tsdf.py (bước 4a). Mặc định KHÔNG "
                              "chạy: hình học nay lấy từ Poisson, mesh TSDF không "
                              "còn được dùng ở bước nào (xem CLAUDE.md §12 item 15)")
    parser.add_argument("--poisson-depth", type=int, default=7,
                         help="Độ sâu Poisson cho 02_mesh_poisson_trunk.py "
                              "(mặc định 7 — đã chọn bằng số đo, xem CLAUDE.md §5)")
    parser.add_argument("--resume", action="store_true",
                         help="Bỏ qua bước đã có output, chạy tiếp từ chỗ dừng")
    parser.add_argument("--force", action="store_true",
                         help="Chạy lại MỌI bước kể cả khi output đã tồn tại. "
                              "Cần cờ này khi dựng lại một scan đã xử lý bằng "
                              "thế hệ TSDF cũ — nếu không, các script con sẽ tự "
                              "bỏ qua vì thấy file cũ còn đó")
    parser.add_argument("--python", default=sys.executable,
                         help="Đường dẫn python dùng để gọi các script con")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir).resolve()
    if not scan_dir.exists() or not scan_dir.is_dir():
        die(f"Thư mục scan không tồn tại: {scan_dir}")

    scripts_dir = Path(args.scripts_dir).resolve() if args.scripts_dir \
        else Path(__file__).resolve().parent
    if not scripts_dir.exists():
        die(f"Thư mục scripts không tồn tại: {scripts_dir}")

    scan_name = scan_dir.name
    raw_data_dir = scan_dir.parent
    out_dir = scan_dir / "output"

    log(f"Thư mục scan     : {scan_dir}")
    log(f"Tên scan         : {scan_name}")
    log(f"Thư mục Raw_data : {raw_data_dir}")
    log(f"Thư mục scripts  : {scripts_dir}")
    log(f"Python           : {args.python}")

    check_raw_data(scan_dir)

    # -- Tham số dùng chung cho các script kiểu --scan / --raw-data -------- #
    scan_args = ["--scan", scan_name, "--raw-data", str(raw_data_dir)]
    if args.force:
        # --skip-existing được lọc ở phía SCRIPT CON (không phải ở orchestrator
        # như --resume), nên đây là chỗ duy nhất tắt được nó. Không bỏ cờ này
        # thì mọi bước có output cũ sẽ tự bỏ qua và giữ nguyên kết quả TSDF cũ.
        args.resume = False
        log("  --force: chạy lại mọi bước, bỏ qua output cũ")
    else:
        scan_args.append("--skip-existing")

    try:
        # Bước 1: rgb.mp4 + depth/ -> rgb/*.jpg
        check_script_modules(args.python, scripts_dir / "01_extract_rgb_frames.py")
        run_step(
            "1/8", "Trích khung hình RGB từ video (01_extract_rgb_frames.py)",
            args.python, scripts_dir / "01_extract_rgb_frames.py",
            [str(scan_dir)], scripts_dir,
            expected_outputs=[scan_dir / "rgb"],
            resume=args.resume,
        )

        # Bước 2: depth/+rgb/+camera_matrix.csv -> output/pointcloud.ply
        check_script_modules(args.python, scripts_dir / "02_pointcloud_and_mesh.py")
        run_step(
            "2/8", "Dựng point cloud + mesh nhanh (02_pointcloud_and_mesh.py)",
            args.python, scripts_dir / "02_pointcloud_and_mesh.py",
            [str(scan_dir), "--no-visualize"], scripts_dir,
            expected_outputs=[out_dir / "pointcloud.ply"],
            resume=args.resume,
        )

        # Bước 3: output/pointcloud.ply -> output/trunk_pointcloud.ply
        check_script_modules(args.python, scripts_dir / "05_trunk_isolation.py")
        run_step(
            "3/8", "Cô lập thân cây khỏi point cloud (05_trunk_isolation.py)",
            args.python, scripts_dir / "05_trunk_isolation.py",
            [str(scan_dir), "--trunk-radius", str(args.trunk_radius)],
            scripts_dir,
            expected_outputs=[out_dir / "trunk_pointcloud.ply"],
            resume=args.resume,
        )

        # Bước 4a (tuỳ chọn): depth/+rgb/+odometry.csv -> trunk_mesh_tsdf.ply
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

        # Bước 5: trunk_mesh_poisson.ply + trunk_pointcloud.ply -> trunk_mesh_recolored.ply
        check_script_modules(args.python, scripts_dir / "03_recolor_mesh.py")
        run_step(
            "5/8", "Tô màu lại mesh (03_recolor_mesh.py)",
            args.python, scripts_dir / "03_recolor_mesh.py",
            scan_args, scripts_dir,
            expected_outputs=[out_dir / "trunk_mesh_recolored.ply"],
            resume=args.resume,
        )

        # Bước 6: trunk_mesh_recolored.ply + location.csv -> trunk_mesh_trimmed.ply
        check_script_modules(args.python, scripts_dir / "04_trim_mesh.py")
        run_step(
            "6/8", "Cắt/lọc mesh theo phương trọng lực (04_trim_mesh.py)",
            args.python, scripts_dir / "04_trim_mesh.py",
            scan_args, scripts_dir,
            expected_outputs=[out_dir / "trunk_mesh_trimmed.ply"],
            resume=args.resume,
        )

        # Bước 7: trunk_mesh_trimmed.ply + imu.csv -> trunk_mesh_final.ply
        check_script_modules(args.python, scripts_dir / "04b_finalize_mesh.py")
        run_step(
            "7/8", "Làm mịn + bịt 2 đầu mesh (04b_finalize_mesh.py)",
            args.python, scripts_dir / "04b_finalize_mesh.py",
            scan_args, scripts_dir,
            expected_outputs=[out_dir / "trunk_mesh_final.ply"],
            resume=args.resume,
        )

        final_model = out_dir / "trunk_mesh_final.ply"

        if not args.skip_lichen:
            # Bước 8: trunk_mesh_final.ply + trunk_pointcloud.ply -> trunk_mesh_detected.ply
            check_script_modules(args.python, scripts_dir / "05_detect_lichen.py")
            run_step(
                "8/8", "Phát hiện địa y trên mesh (05_detect_lichen.py)",
                args.python, scripts_dir / "05_detect_lichen.py",
                scan_args, scripts_dir,
                expected_outputs=[
                    out_dir / "trunk_mesh_detected.ply",
                    out_dir / "lichen_stats.json",
                ],
                resume=args.resume,
            )
            final_model = out_dir / "trunk_mesh_detected.ply"
        else:
            log("\n  (--skip-lichen: bỏ qua bước phát hiện địa y)")

        if args.with_slope:
            run_step(
                "phụ", "Phân tích độ dốc địa hình (06_slope_analysis.py)",
                args.python, scripts_dir / "06_slope_analysis.py",
                [str(scan_dir)], scripts_dir,
                expected_outputs=[],  # script này in kết quả ra stdout/json tuỳ bản
                resume=False,
            )

    except StepFailed as e:
        die(str(e))

    banner("HOÀN TẤT PIPELINE")
    log(f"  Model 3D cuối cùng : {final_model}")
    if not args.skip_lichen:
        log(f"  Thống kê địa y     : {out_dir / 'lichen_stats.json'}")
    log(f"\n  Xem trực quan bằng: python \"{scripts_dir / '07_preview.py'}\"")
    log("")


if __name__ == "__main__":
    main()