#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_diagnostic_trunk_axis_up_axis.py
========================================
Va diagnostic_trunk_axis.py: cong cu nay (chi doc, khong sua du lieu that)
mo phong dung logic goc (sai truc) cua 05_trunk_isolation.py truoc khi
duoc sua - dung cot Z (index 2) lam "chieu cao" va cot Y (index 1) nhu the
la mot toa do ngang. Neu khong sua dong bo, cong cu chan doan nay se tiep
tuc "xac nhan" sai truc sau khi 05_trunk_isolation.py da duoc sua dung
(xem patch_trunk_isolation_up_axis.py va geometry_utils.py), gay nham lan.

Khong anh huong du lieu production (day la script chan doan doc lap,
khong ai goi no trong pipeline chinh).

Ky thuat: thay vi go lai tay toan bo than ham analyze_trunk_axis_detection()
(rui ro sai lech ky tu Unicode - emoji/mui ten - kho phat hien bang mat),
patch nay TRICH XUAT chinh xac doan code hien tai tu file (dam bao khop
100%), roi ap mot loat thay the nho, co muc tieu ro rang, len chinh van
ban da trich xuat do.

CACH DUNG
---------
    python patch_diagnostic_trunk_axis_up_axis.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "diagnostic_trunk_axis.py"
FUNC_START_MARKER = "def analyze_trunk_axis_detection("
FUNC_END_MARKER = "def main():"

# Cac doan text nho, doc lap, ap dung LEN TREN VAN BAN HAM DA TRICH XUAT
# (khong phai toan bo file) - moi doan chi vai dong, de xac minh dung.
FUNC_BODY_REPLACEMENTS: list[tuple[str, str, str]] = [
    (
        "func_docstring",
        "    Phân tích cách trục X, Y được xác định",
        "    Phân tích cách trục X, Z được xác định",
    ),
    (
        "z_raw_source",
        "    z_raw = points_raw[:, 2]",
        "    z_raw = geometry_utils.vertical(points_raw)",
    ),
    (
        "median_xy_40",
        "    cx_40 = np.median(upper_pts_40[:, 0])\n"
        "    cy_40 = np.median(upper_pts_40[:, 1])\n"
        "    print(f\"   Median XY: ({cx_40:.3f}, {cy_40:.3f})\")",
        "    cx_40, cz_40 = geometry_utils.horizontal_center(upper_pts_40)\n"
        "    print(f\"   Median XZ: ({cx_40:.3f}, {cz_40:.3f})\")",
    ),
    (
        "mean_xy_40",
        "    cx_mean_40 = np.mean(upper_pts_40[:, 0])\n"
        "    cy_mean_40 = np.mean(upper_pts_40[:, 1])\n"
        "    print(f\"   Mean XY:   ({cx_mean_40:.3f}, {cy_mean_40:.3f})\")",
        "    xz_mean_40 = geometry_utils.horizontal(upper_pts_40)\n"
        "    cx_mean_40 = np.mean(xz_mean_40[:, 0])\n"
        "    cz_mean_40 = np.mean(xz_mean_40[:, 1])\n"
        "    print(f\"   Mean XZ:   ({cx_mean_40:.3f}, {cz_mean_40:.3f})\")",
    ),
    (
        "loop_median_xy",
        "            cx = np.median(upper_pts[:, 0])\n"
        "            cy = np.median(upper_pts[:, 1])\n"
        "            print(f\"\\n   [Upper {upper_frac*100:.0f}%]\")\n"
        "            print(f\"   Points: {upper_mask.sum():,}\")\n"
        "            print(f\"   Median XY: ({cx:.3f}, {cy:.3f})\")",
        "            cx, cz = geometry_utils.horizontal_center(upper_pts)\n"
        "            print(f\"\\n   [Upper {upper_frac*100:.0f}%]\")\n"
        "            print(f\"   Points: {upper_mask.sum():,}\")\n"
        "            print(f\"   Median XZ: ({cx:.3f}, {cz:.3f})\")",
    ),
    (
        "trunk_xy_vars",
        "    x_trunk = points_trunk[:, 0]\n"
        "    y_trunk = points_trunk[:, 1]",
        "    x_trunk = points_trunk[:, 0]\n"
        "    z_trunk = points_trunk[:, 2]",
    ),
    (
        "trunk_y_range_print",
        '    print(f"   Y range: {y_trunk.min():.3f}m',
        '    print(f"   Z range: {z_trunk.min():.3f}m',
    ),
    (
        "trunk_y_range_print_tail",
        "{y_trunk.max():.3f}m (range={y_trunk.max()-y_trunk.min():.3f}m)\")",
        "{z_trunk.max():.3f}m (range={z_trunk.max()-z_trunk.min():.3f}m)\")",
    ),
    (
        "cx_cy_trunk_median",
        "    cx_trunk = np.median(x_trunk)\n"
        "    cy_trunk = np.median(y_trunk)\n"
        "    print(f\"\\n   Median XY: ({cx_trunk:.3f}, {cy_trunk:.3f})\")",
        "    cx_trunk, cz_trunk = geometry_utils.horizontal_center(points_trunk)\n"
        "    print(f\"\\n   Median XZ: ({cx_trunk:.3f}, {cz_trunk:.3f})\")",
    ),
    (
        "dist_xy_trunk",
        "    dist_xy = np.sqrt((x_trunk - cx_trunk)**2 + (y_trunk - cy_trunk)**2)\n"
        "    print(f\"\\n   Distance from center to trunk points (XY):\")\n"
        "    print(f\"   Max:   {dist_xy.max():.3f}m\")\n"
        "    print(f\"   Mean:  {dist_xy.mean():.3f}m\")\n"
        "    print(f\"   Std:   {dist_xy.std():.3f}m\")\n"
        "    print(f\"   Pct95: {np.percentile(dist_xy, 95):.3f}m\")",
        "    dist_xz = geometry_utils.horizontal_distance(points_trunk, (cx_trunk, cz_trunk))\n"
        "    print(f\"\\n   Distance from center to trunk points (XZ):\")\n"
        "    print(f\"   Max:   {dist_xz.max():.3f}m\")\n"
        "    print(f\"   Mean:  {dist_xz.mean():.3f}m\")\n"
        "    print(f\"   Std:   {dist_xz.std():.3f}m\")\n"
        "    print(f\"   Pct95: {np.percentile(dist_xz, 95):.3f}m\")",
    ),
    (
        "axis_comparison_print",
        '    print(f"   Raw upper 40% median: ({cx_40:.3f}, {cy_40:.3f})")\n'
        '    print(f"   Trunk median:         ({cx_trunk:.3f}, {cy_trunk:.3f})")',
        '    print(f"   Raw upper 40% median: ({cx_40:.3f}, {cz_40:.3f})")\n'
        '    print(f"   Trunk median:         ({cx_trunk:.3f}, {cz_trunk:.3f})")',
    ),
    (
        "dist_axis_calc",
        "    dist_axis = np.sqrt((cx_40 - cx_trunk)**2 + (cy_40 - cy_trunk)**2)",
        "    dist_axis = np.sqrt((cx_40 - cx_trunk)**2 + (cz_40 - cz_trunk)**2)",
    ),
    (
        "warn_xy_text_1",
        "Có thể trục X, Y bị xác định sai!",
        "Có thể trục X, Z bị xác định sai!",
    ),
    (
        "ok_xy_text_1",
        "Trục X, Y xác định hợp lý",
        "Trục X, Z xác định hợp lý",
    ),
    (
        "raw_xy_vars",
        "    x_raw = points_raw[:, 0]\n"
        "    y_raw = points_raw[:, 1]\n"
        "    \n"
        "    # Tính distance từ từng điểm đến tâm predicted (cx_40, cy_40)\n"
        "    dist_from_axis_40 = np.sqrt((x_raw - cx_40)**2 + (y_raw - cy_40)**2)",
        "    # Tính distance từ từng điểm đến tâm predicted (cx_40, cz_40)\n"
        "    dist_from_axis_40 = geometry_utils.horizontal_distance(points_raw, (cx_40, cz_40))",
    ),
    (
        "dist_from_axis_print",
        "    print(f\"   Distance from predicted axis ({cx_40:.3f}, {cy_40:.3f}):\")",
        "    print(f\"   Distance from predicted axis ({cx_40:.3f}, {cz_40:.3f}):\")",
    ),
    (
        "warn_xy_text_2",
        "Trục X, Y có vẻ bị lệch!",
        "Trục X, Z có vẻ bị lệch!",
    ),
    (
        "ok_xy_text_2",
        "Phân bố XY có vẻ hợp lý",
        "Phân bố XZ có vẻ hợp lý",
    ),
]


def patch_function_body(content: str) -> tuple[str, list[str]]:
    """Trich xuat than ham analyze_trunk_axis_detection() tu content, ap
    cac thay the nho, roi tra ve (content_moi, danh_sach_da_ap_dung)."""
    start = content.find(FUNC_START_MARKER)
    if start == -1:
        print(f"KHONG TIM THAY: '{FUNC_START_MARKER}'")
        sys.exit(1)
    end = content.find(FUNC_END_MARKER, start)
    if end == -1:
        print(f"KHONG TIM THAY: '{FUNC_END_MARKER}' sau ham analyze_trunk_axis_detection")
        sys.exit(1)

    func_body = content[start:end]
    original_func_body = func_body
    applied = []
    missing = []
    for label, old, new in FUNC_BODY_REPLACEMENTS:
        if new in func_body:
            continue  # da ap dung roi (idempotent)
        if old not in func_body:
            missing.append(label)
            continue
        func_body = func_body.replace(old, new, 1)
        applied.append(label)

    if missing:
        print(f"KHONG KHOP (trong than ham): {missing}")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi.")
        sys.exit(1)

    new_content = content[:start] + func_body + content[end:]
    return new_content, applied


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    # --- Cac doan nho o ngoai than ham (docstring dau file, import) --------
    TOP_SNIPPETS = [
        (
            "module_docstring",
            "Phân tích xem trục X, Y của thân cây được xác định có chính xác không.",
            "Phân tích xem trục X, Z của thân cây được xác định có chính xác không.",
        ),
        (
            "import_geometry_utils",
            "import os\nimport sys\nimport numpy as np\nimport open3d as o3d",
            "import os\nimport sys\nimport numpy as np\nimport open3d as o3d\n\nimport geometry_utils",
        ),
    ]

    already_all_top = all(new in content for _, _, new in TOP_SNIPPETS)
    already_all_func = FUNC_START_MARKER in content and "geometry_utils.vertical(points_raw)" in content
    if already_all_top and already_all_func:
        print(f"{TARGET_NAME} (up-axis fix) da duoc va truoc do roi. Khong can lam gi them.")
        return

    missing_top = [label for label, old, new in TOP_SNIPPETS
                   if new not in content and old not in content]
    if missing_top:
        print(f"KHONG KHOP (ngoai than ham): {missing_top}")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_upaxis")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    applied_top = []
    for label, old, new in TOP_SNIPPETS:
        if new in content:
            continue
        content = content.replace(old, new, 1)
        applied_top.append(label)

    content, applied_func = patch_function_body(content)

    target.write_text(content, encoding="utf-8")
    print(f"Da va thanh cong (up-axis fix).")
    print(f"  Ngoai than ham: {applied_top}")
    print(f"  Trong than ham: {applied_func}")


if __name__ == "__main__":
    main()
