#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_pipeline.py
=====================
Chạy toàn bộ pipeline xử lý MỘT cây quế — hoặc cả thư mục dữ liệu với --all —
từ dữ liệu thô của app quét cho tới mesh 3D đã tô màu và đánh dấu địa y.

Script này KHÔNG chứa logic xử lý. Nó gọi lần lượt các script đã có theo đúng
thứ tự, kiểm tra input trước mỗi bước và output sau mỗi bước, để hỏng ở đâu
biết ngay ở đó thay vì chạy hết rồi mới phát hiện.


QUY TRÌNH — 8 BƯỚC, TỪ VIDEO ĐẾN ĐỊA Y
======================================

  vào:  rgb.mp4 · depth/ · camera_matrix.csv · odometry.csv · imu.csv

  1  01_extract_rgb_frames.py     tách video thành khung ảnh JPEG
                                  -> rgb/*.jpg

  2  02_pointcloud_and_mesh.py    chiếu ngược ảnh độ sâu + màu thành đám mây
                                  điểm 3D của CẢ CẢNH, gộp theo tư thế camera
                                  -> output/pointcloud.ply

  3  05_trunk_isolation.py        tách riêng THÂN CÂY MỤC TIÊU ra khỏi cảnh.
                                  Đây là bước nhiều cơ chế nhất: neo theo tia
                                  nhìn của camera để biết cây nào là mục tiêu,
                                  cắt trụ bám theo trục thật của thân, lọc cây
                                  xanh, tách nhánh, bỏ lát đất.
                                  -> output/trunk_pointcloud.ply

  4a 02_mesh_tsdf.py              (MẶC ĐỊNH BỎ QUA, cần --with-tsdf)
                                  Cách dựng mesh cũ. Đã bị thay vì để lại lỗ
                                  lớn trên vùng đã quét — xem CLAUDE.md §12
                                  item 15.

  4b 02_mesh_poisson_trunk.py     dựng mesh từ đám mây điểm thân cây. Cắt các
                                  lát không khép vòng (chỗ camera không đi hết
                                  vòng quanh) rồi cắt phẳng hai đầu.
                                  -> output/trunk_mesh_poisson.ply

  5  03_recolor_mesh.py           chuyển màu từ đám mây điểm sang đỉnh mesh
                                  -> output/trunk_mesh_recolored.ply

  6  04_trim_mesh.py              bỏ phần mesh không thuộc thân, dùng chính
                                  trunk_pointcloud.ply làm mặt nạ
                                  -> output/trunk_mesh_trimmed.ply

  7  04b_finalize_mesh.py         làm mịn Taubin, bịt hai đầu
                                  -> output/trunk_mesh_final.ply

  8  05_detect_lichen.py          đánh dấu địa y + chia trên/dưới dốc + tính
                                  diện tích
                                  -> output/trunk_mesh_detected.ply
                                  -> output/lichen_stats.json

  phụ 06_slope_analysis.py        độ nghiêng thân, hướng nghiêng (--with-slope)


BƯỚC 8 ĐANG DÙNG GÌ, VÀ CON SỐ NÀO ĐÁNG TIN
===========================================

Từ tháng 9/2026 bước 8 dùng mô hình chỉnh theo NHÃN TAY (lichen_model.json,
huấn luyện trên 10 cây được tô tay, 165.887 điểm có nhãn). Đo bằng cách bỏ
một cây ra mỗi lần:

      F1 trên cây chưa từng thấy     mô hình mới 40,6%     hàm cũ 13,0%

Nhưng phải nói rõ, vì nó quyết định con số nào được phép đưa vào báo cáo:

  DÙNG ĐƯỢC   vị trí địa y trên thân — trunk_mesh_detected.ply

  KHÔNG DÙNG  tỉ lệ phần trăm của TỪNG CÂY (lichen_ratio_pct). Đo trên 10 cây
  ĐƯỢC        có nhãn: tương quan với sự thật chỉ +0,07, và đoán một con số cố
              định cho mọi cây còn chính xác hơn gấp đôi. Nguyên nhân: chênh
              lệch màu giữa địa y và vỏ biến thiên 3 lần giữa các cây và không
              đo được nếu không có nhãn của chính cây đó. Năm cách chữa đã thử
              và thất bại — CLAUDE.md §12 item 22.

  CON SỐ      ước lượng cấp corpus lấy trực tiếp từ nhãn tay:
  BÁO CÁO         23,2%   khoảng tin cậy 95%: 14,4% – 32,0%   (n = 10 cây)
  ĐƯỢC        Con số này kế thừa cách chọn cây khi đi quay (§12 item 20) nên
              mô tả MẪU ĐÃ QUÉT, không phải cả vườn.

--legacy-detector để quay về hàm HSV cũ.


CÁCH DÙNG
---------
Một cây:
    python run_full_pipeline.py "D:\Backup\Thucdia-18May2026\cay_0007_1805"

Cả thư mục dữ liệu:
    python run_full_pipeline.py --all "D:\Backup\Thucdia-18May2026"

Chạy lại từ một bước (ví dụ chỉ dò lại địa y cho cả corpus):
    python run_full_pipeline.py --all "D:\Backup\Thucdia-18May2026" --from-step 8

Kết quả mỗi cây:
    <scan>/output/trunk_mesh_detected.ply   model 3D cuối cùng
    <scan>/output/lichen_stats.json         thống kê (kèm trường WARNING)


TÙY CHỌN
--------
  --all               Chạy cho MỌI thư mục con của đường dẫn đã cho. Mỗi cây
                       chạy trong một tiến trình riêng, cây này hỏng không làm
                       dừng các cây còn lại; cuối cùng in bảng tổng kết.
  --scripts-dir DIR   Thư mục chứa các script 01_..09_...
  --trunk-radius R    Bán kính (m) cô lập thân cây (mặc định 0.30)
  --skip-lichen       Dừng ở bước 7, không chạy bước 8
  --legacy-detector   Bước 8 dùng hàm HSV cũ thay vì mô hình nhãn tay
  --with-slope        Chạy thêm 06_slope_analysis.py
  --with-tsdf         Chạy thêm bước 4a (mesh TSDF, không ai đọc nữa)
  --poisson-depth N   Độ sâu Poisson cho bước 4b
  --from-step ID      Bỏ qua các bước trước mốc này (1,2,3,4,4b,5,6,7,8)
  --resume            Bỏ qua bước nào đã có output
  --force             Chạy lại mọi bước, đè lên output cũ
  --python PATH       Trình thông dịch dùng để gọi script con

Một bước lỗi thì DỪNG NGAY cây đó, in rõ bước nào, lý do, và toàn bộ log của
bước đó.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


# [Sep 2026] File NAY la ngoai le duy nhat cua quy uoc thuan-ASCII (CLAUDE.md
# muc 5): moi chuoi help va log deu co dau tieng Viet. Tren console cp1252 mac
# dinh cua Windows, ngay ca `--help` cung do UnicodeEncodeError va khong in noi
# mot dong nao -- da kiem chung tren ca ban truoc lan ban nay. Quy uoc san co
# cua repo la truyen PYTHONIOENCODING/PYTHONUTF8 cho tien trinh CON; khong co gi
# dat cho chinh tien trinh nay. errors="replace" de truong hop xau nhat la chu
# bi thay bang '?', chu khong phai mot traceback.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


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
    "02_mesh_poisson_trunk.py": ["numpy", "open3d", "scipy"],
    "03_recolor_mesh.py": ["numpy", "open3d", "scipy"],
    "04_trim_mesh.py": ["numpy", "open3d", "scipy"],
    "04b_finalize_mesh.py": ["numpy", "open3d"],
    "05_detect_lichen.py": ["numpy", "open3d", "scipy"],
    "06_slope_analysis.py": ["numpy", "open3d"],
}

# Thứ tự các bước, đánh số THEO CLAUDE.md §3a (bước 4 = TSDF, 4b = Poisson).
# Trước đây banner in "4/8" cho Poisson và "4a/8" cho TSDF, lệch với tài liệu;
# đã sửa cho khớp, nếu không thì gõ `--from-step 4b` lại thấy hiện ra "4/8".
STEP_ORDER = ["1", "2", "3", "4", "4b", "5", "6", "7", "8"]
STEP_ALIAS = {"4a": "4"}      # nhãn cũ của bước TSDF


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

def run_corpus(args) -> None:
    """Chạy pipeline cho mọi thư mục con, mỗi cây một tiến trình riêng.

    Gọi lại chính file này thay vì lặp trong bộ nhớ: một cây hỏng — hết RAM,
    dữ liệu thô thiếu, script con chết — không kéo theo cả lượt, và mỗi cây có
    log riêng đọc được.
    """
    root = Path(args.scan_dir).resolve()
    if not root.is_dir():
        die(f"Không tìm thấy thư mục: {root}")
    scans = sorted(d for d in root.iterdir()
                   if d.is_dir() and (d / "depth").is_dir())
    if not scans:
        die(f"Không có thư mục scan nào trong {root} "
            f"(tìm thư mục con có chứa depth/)")

    banner(f"CHẠY CẢ CORPUS — {len(scans)} cây trong {root}")

    passthrough = []
    for flag, val in (("--scripts-dir", args.scripts_dir),
                      ("--python", args.python),
                      ("--from-step", args.from_step),
                      ("--poisson-depth", args.poisson_depth),
                      ("--trunk-radius", args.trunk_radius)):
        if val is not None:
            passthrough += [flag, str(val)]
    for flag in ("skip_lichen", "legacy_detector", "with_slope", "with_tsdf",
                 "resume", "force", "set_poisson_depth"):
        if getattr(args, flag, False):
            passthrough.append("--" + flag.replace("_", "-"))

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    t0 = time.time()
    ok, fail = [], []
    for i, d in enumerate(scans, 1):
        log(f"\n[{i}/{len(scans)}] {d.name}")
        t1 = time.time()
        rc = subprocess.call([args.python, str(Path(__file__).resolve()),
                              str(d)] + passthrough, env=env)
        dt = time.time() - t1
        if rc == 0:
            ok.append((d.name, dt))
            log(f"  OK  [{dt:.0f}s]")
        else:
            fail.append((d.name, rc))
            log(f"  LỖI (mã {rc}) — chạy riêng cây này để xem log đầy đủ")

    banner("TỔNG KẾT CẢ CORPUS")
    log(f"  Thành công : {len(ok)}/{len(scans)}")
    if fail:
        log(f"  Thất bại   : {len(fail)}")
        for name, rc in fail:
            log(f"      {name}  (mã {rc})")
        log("\n  Chạy lại một cây để xem lỗi:")
        log(f'      python "{Path(__file__).resolve()}" "{root / fail[0][0]}"')
    log(f"\n  Tổng thời gian: {(time.time() - t0) / 60:.0f} phút")
    if not args.skip_lichen:
        log("\n  Nhắc lại: lichen_ratio_pct của TỪNG cây KHÔNG đáng tin")
        log("  (tương quan +0,07 với nhãn tay trên 10 cây). Con số dùng được")
        log("  cho báo cáo là ước lượng cấp corpus 23,2% (14,4–32,0%, n=10).")
        log("  Chi tiết: CLAUDE.md §12 item 22.")
    if fail:
        sys.exit(1)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy toàn bộ pipeline 3D thân cây quế từ dữ liệu thô "
                     "đến model 3D cuối cùng.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scan_dir",
                         help="Thư mục scan. Với --all thì đây là thư mục "
                              "CHA chứa nhiều thư mục scan.")
    parser.add_argument("--all", action="store_true",
                         help="Chạy cho mọi thư mục con. Mỗi cây một tiến "
                              "trình riêng nên cây hỏng không làm dừng cả lượt.")
    parser.add_argument("--scripts-dir", default=None,
                         help="Thư mục chứa các script 01_..09_... "
                              "(mặc định: cùng thư mục với file này)")
    parser.add_argument("--trunk-radius", type=float, default=0.30,
                         help="Bán kính (m) cô lập thân cây (mặc định 0.30)")
    parser.add_argument("--skip-lichen", action="store_true",
                         help="Dừng ở bước làm mịn, không phát hiện địa y")
    parser.add_argument("--legacy-detector", action="store_true",
                         help="Bước 8 dùng hàm HSV cũ thay vì mô hình nhãn "
                              "tay (F1 13,0%% so với 40,6%%, xem đầu file)")
    parser.add_argument("--with-slope", action="store_true",
                         help="Chạy thêm bước phân tích độ dốc địa hình")
    parser.add_argument("--with-tsdf", action="store_true",
                         help="Chạy thêm 02_mesh_tsdf.py (bước 4a). Mặc định KHÔNG "
                              "chạy: hình học nay lấy từ Poisson, mesh TSDF không "
                              "còn được dùng ở bước nào (xem CLAUDE.md §12 item 15)")
    parser.add_argument("--poisson-depth", type=int, default=None,
                         help="Độ sâu Poisson cho 02_mesh_poisson_trunk.py. "
                              "Không gõ thì KHÔNG truyền xuống, để script con "
                              "tự dùng mặc định của nó cộng với lựa chọn riêng "
                              "từng cây trong poisson_overrides.json — một "
                              "nguồn sự thật duy nhất (xem CLAUDE.md §5)")
    parser.add_argument("--set-poisson-depth", action="store_true",
                         help="Ghi nhớ --poisson-depth cho riêng cây này, để "
                              "lần chạy toàn bộ corpus sau vẫn dùng. Chuyển "
                              "thẳng xuống --set-depth của bước 4b")
    parser.add_argument("--from-step", default=None, metavar="ID",
                         choices=STEP_ORDER + list(STEP_ALIAS),
                         help="Bỏ qua mọi bước trước mốc này và chạy từ đây trở "
                              "đi, đè lên output cũ. Dùng khi chỉ muốn dựng lại "
                              "mesh ở depth khác: `--from-step 4b`. "
                              "Giá trị: " + ", ".join(STEP_ORDER))
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

    if args.all:
        run_corpus(args)
        return

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

    # -- Chọn bước bắt đầu -------------------------------------------------- #
    from_step = STEP_ALIAS.get(args.from_step, args.from_step)
    from_idx = STEP_ORDER.index(from_step) if from_step else 0
    if from_step:
        # Cả điểm của --from-step là dựng lại đè lên file cũ, nên nó bao hàm
        # --force: nếu không, script con thấy output cũ còn đó sẽ tự bỏ qua và
        # chẳng có gì được dựng lại.
        args.force = True
        log(f"  --from-step {args.from_step}: chạy từ bước {from_step} trở đi, "
            f"đè lên output cũ")

    def want(step_id: str) -> bool:
        """True nếu bước này nằm từ mốc --from-step trở đi."""
        return STEP_ORDER.index(step_id) >= from_idx

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

    # -- Tham số riêng của bước 4b ------------------------------------------ #
    poisson_args = list(scan_args)
    if args.poisson_depth is not None:
        poisson_args += ["--depth", str(args.poisson_depth)]
    if args.set_poisson_depth:
        if args.poisson_depth is None:
            die("--set-poisson-depth cần biết ghi nhớ gì, hãy kèm --poisson-depth N")
        poisson_args.append("--set-depth")

    try:
        # Bước 1: rgb.mp4 + depth/ -> rgb/*.jpg
        if want("1"):
            check_script_modules(args.python, scripts_dir / "01_extract_rgb_frames.py")
            run_step(
                "1/8", "Trích khung hình RGB từ video (01_extract_rgb_frames.py)",
                args.python, scripts_dir / "01_extract_rgb_frames.py",
                [str(scan_dir)], scripts_dir,
                expected_outputs=[scan_dir / "rgb"],
                resume=args.resume,
            )

        # Bước 2: depth/+rgb/+camera_matrix.csv -> output/pointcloud.ply
        if want("2"):
            check_script_modules(args.python, scripts_dir / "02_pointcloud_and_mesh.py")
            run_step(
                "2/8", "Dựng point cloud + mesh nhanh (02_pointcloud_and_mesh.py)",
                args.python, scripts_dir / "02_pointcloud_and_mesh.py",
                # --no-mesh: bỏ STEP 2 nội bộ (mesh.ply). Không bước nào từ 3
                # đến 8 đọc file đó — hình học lấy từ 02_mesh_poisson_trunk.py.
                # Trên cay_0043 bước đó làm chết cả script bằng MemoryError SAU
                # KHI pointcloud.ply đã ghi xong, tức mất cả lần chạy vì một
                # file không ai dùng.
                [str(scan_dir), "--no-visualize", "--no-mesh"], scripts_dir,
                expected_outputs=[out_dir / "pointcloud.ply"],
                resume=args.resume,
            )

        # Bước 3: output/pointcloud.ply -> output/trunk_pointcloud.ply
        if want("3"):
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
        if args.with_tsdf and want("4"):
            check_script_modules(args.python, scripts_dir / "02_mesh_tsdf.py")
            run_step(
                "4/8", "Dựng mesh TSDF để đối chiếu (02_mesh_tsdf.py)",
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
        if want("4b"):
            check_script_modules(args.python, scripts_dir / "02_mesh_poisson_trunk.py")
            run_step(
                "4b/8", "Dựng mesh thân cây bằng Poisson (02_mesh_poisson_trunk.py)",
                args.python, scripts_dir / "02_mesh_poisson_trunk.py",
                poisson_args, scripts_dir,
                expected_outputs=[out_dir / "trunk_mesh_poisson.ply"],
                resume=args.resume,
            )

        # Bước 5: trunk_mesh_poisson.ply + trunk_pointcloud.ply -> trunk_mesh_recolored.ply
        if want("5"):
            check_script_modules(args.python, scripts_dir / "03_recolor_mesh.py")
            run_step(
                "5/8", "Tô màu lại mesh (03_recolor_mesh.py)",
                args.python, scripts_dir / "03_recolor_mesh.py",
                scan_args, scripts_dir,
                expected_outputs=[out_dir / "trunk_mesh_recolored.ply"],
                resume=args.resume,
            )

        # Bước 6: trunk_mesh_recolored.ply + location.csv -> trunk_mesh_trimmed.ply
        if want("6"):
            check_script_modules(args.python, scripts_dir / "04_trim_mesh.py")
            run_step(
                "6/8", "Cắt/lọc mesh theo phương trọng lực (04_trim_mesh.py)",
                args.python, scripts_dir / "04_trim_mesh.py",
                scan_args, scripts_dir,
                expected_outputs=[out_dir / "trunk_mesh_trimmed.ply"],
                resume=args.resume,
            )

        # Bước 7: trunk_mesh_trimmed.ply + imu.csv -> trunk_mesh_final.ply
        if want("7"):
            check_script_modules(args.python, scripts_dir / "04b_finalize_mesh.py")
            run_step(
                "7/8", "Làm mịn + bịt 2 đầu mesh (04b_finalize_mesh.py)",
                args.python, scripts_dir / "04b_finalize_mesh.py",
                scan_args, scripts_dir,
                expected_outputs=[out_dir / "trunk_mesh_final.ply"],
                resume=args.resume,
            )

        final_model = out_dir / "trunk_mesh_final.ply"

        if not args.skip_lichen and want("8"):
            # Bước 8: trunk_mesh_final.ply + trunk_pointcloud.ply -> trunk_mesh_detected.ply
            check_script_modules(args.python, scripts_dir / "05_detect_lichen.py")
            run_step(
                "8/8", "Phát hiện địa y trên mesh (05_detect_lichen.py)",
                args.python, scripts_dir / "05_detect_lichen.py",
                scan_args + (["--legacy-detector"] if args.legacy_detector else []),
                scripts_dir,
                expected_outputs=[
                    out_dir / "trunk_mesh_detected.ply",
                    out_dir / "lichen_stats.json",
                ],
                resume=args.resume,
            )
            final_model = out_dir / "trunk_mesh_detected.ply"
        elif args.skip_lichen:
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
        log("")
        log("  LƯU Ý: 'lichen_ratio_pct' của một cây KHÔNG đáng tin (tương quan")
        log("  +0,07 với nhãn tay trên 10 cây; đoán một số cố định còn chính xác")
        log("  hơn gấp đôi). File kết quả có sẵn trường WARNING nói điều này.")
        log("  Dùng được: VỊ TRÍ địa y trên thân. Xem CLAUDE.md §12 item 22.")
    log(f"\n  Xem trực quan bằng: python \"{scripts_dir / '07_preview.py'}\"")
    log("")


if __name__ == "__main__":
    main()