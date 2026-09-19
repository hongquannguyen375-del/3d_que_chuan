#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_slope_analysis_up_axis.py
=================================
Va 06_slope_analysis.py: sua 2 loi lien quan den viec dung SAI truc lam
"chieu cao"/"mat phang ngang" - cung mot lop loi da sua trong
05_trunk_isolation.py (xem patch_trunk_isolation_up_axis.py va
geometry_utils.py).

Loi 1 (chi anh huong hien thi): estimate_trunk_axis_pca() chuan hoa dau
cua truc PCA bang cach kiem tra trunk_axis[2] (cot Z) thay vi cot Y (truc
that su la "len troi", da xac nhan PCA trong 04_trim_mesh.py). Da kiem tra
lai bang toan hoc: compute_slope_direction() KHONG phu thuoc dau cua
trunk_axis (phep chieu (g.a)a khong doi khi a -> -a), nen loi nay chi lam
sai gia tri trunk_axis ghi trong slope_info.json, khong lam sai slope_down/
slope_up/terrain_slope_deg.

Loi 2 (anh huong that den ket qua): label_trunk_by_slope() va analyze_slope()
tinh "vector tu truc than den tung diem" bang dx=cot X, dy=cot Y - nhung cot
Y la DO CAO, khong phai toa do ngang! Dieu nay lam ro ri chieu cao vao phep
tinh von phai la thuan tuy ngang, co the khien nhan upslope/downslope bi
lech theo do cao doc than cay thay vi theo huong la ban that. Day la mot
trong nhung ket qua dau ra chinh cua du an (upslope/downslope lichen cm2/%).

CACH DUNG
---------
    python patch_slope_analysis_up_axis.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "06_slope_analysis.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- 1. Import geometry_utils ------------------------------------------------
SNIPPETS.append((
    "import_geometry_utils",
    '''import numpy as np
import open3d as o3d''',
    '''import numpy as np
import open3d as o3d

import geometry_utils''',
))

# --- 2. estimate_trunk_axis_pca(): sign check dung dung truc UP_AXIS --------
SNIPPETS.append((
    "estimate_trunk_axis_pca",
    '''def estimate_trunk_axis_pca(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Dùng PCA trên point cloud thân cây để tìm trục chính (hướng thân cây).
    Thường là eigenvector ứng với eigenvalue lớn nhất.
    Trả về unit vector hướng lên (positive z component).
    """
    points = np.asarray(pcd.points)
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Eigenvector của eigenvalue lớn nhất = trục chính thân cây
    trunk_axis = eigvecs[:, -1]
    # Đảm bảo hướng lên
    if trunk_axis[2] < 0:
        trunk_axis = -trunk_axis
    return trunk_axis''',
    '''def estimate_trunk_axis_pca(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Dùng PCA trên point cloud thân cây để tìm trục chính (hướng thân cây).
    Thường là eigenvector ứng với eigenvalue lớn nhất.
    Trả về unit vector hướng lên (positive component theo trục thẳng đứng).
    """
    points = np.asarray(pcd.points)
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Eigenvector của eigenvalue lớn nhất = trục chính thân cây
    trunk_axis = eigvecs[:, -1]
    # Đảm bảo hướng lên
    trunk_axis = geometry_utils.ensure_points_up(trunk_axis)
    return trunk_axis''',
))

# --- 3. label_trunk_by_slope(): cy->cz, dx/dy/radial -> dx/dz qua X,Z -------
SNIPPETS.append((
    "label_trunk_by_slope",
    '''def label_trunk_by_slope(
    pcd: o3d.geometry.PointCloud,
    cx: float,
    cy: float,
    slope_down: np.ndarray,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Gán nhãn từng điểm là upslope (0) hoặc downslope (1).

    Nguyên tắc:
    - Project vị trí điểm từ trục thân ra hướng ngang (XY plane)
    - Dot product với slope_down:
        > 0  → downslope (phía dốc xuống)
        <= 0 → upslope   (phía dốc lên)

    Màu hóa:
        upslope   → đỏ  (1,0,0)
        downslope → xanh dương (0,0,1)
    """
    points = np.asarray(pcd.points)

    # Vector từ trục thân đến từng điểm (XY plane)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    radial = np.stack([dx, dy, np.zeros(len(dx))], axis=1)  # (N, 3)

    # Dot product với slope_down (chỉ dùng XY của slope_down)
    slope_xy = slope_down.copy()
    slope_xy[2] = 0.0
    slope_xy_mag = np.linalg.norm(slope_xy)
    if slope_xy_mag > 1e-6:
        slope_xy /= slope_xy_mag

    dot = radial @ slope_xy  # (N,) — dương = phía downslope''',
    '''def label_trunk_by_slope(
    pcd: o3d.geometry.PointCloud,
    cx: float,
    cz: float,
    slope_down: np.ndarray,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Gán nhãn từng điểm là upslope (0) hoặc downslope (1).

    Nguyên tắc:
    - Project vị trí điểm từ trục thân ra mặt phẳng ngang (X, Z)
    - Dot product với slope_down:
        > 0  → downslope (phía dốc xuống)
        <= 0 → upslope   (phía dốc lên)

    Màu hóa:
        upslope   → đỏ  (1,0,0)
        downslope → xanh dương (0,0,1)
    """
    points = np.asarray(pcd.points)

    # Vector từ trục thân đến từng điểm, trên mặt phẳng ngang (X, Z)
    dx = points[:, 0] - cx
    dz = points[:, 2] - cz
    radial = geometry_utils.embed_horizontal(np.stack([dx, dz], axis=1))  # (N, 3)

    # Dot product với thành phần ngang của slope_down
    slope_horiz = geometry_utils.project_to_horizontal(slope_down)
    slope_horiz_mag = np.linalg.norm(slope_horiz)
    if slope_horiz_mag > 1e-6:
        slope_horiz /= slope_horiz_mag

    dot = radial @ slope_horiz  # (N,) — dương = phía downslope''',
))

# --- 4. analyze_slope(): tam truc + call site -------------------------------
SNIPPETS.append((
    "analyze_slope_center",
    '''    # Tâm trục (để project radial)
    pts = np.asarray(pcd.points)
    cx = float(np.median(pts[:, 0]))
    cy = float(np.median(pts[:, 1]))

    # Gán nhãn upslope/downslope
    labeled_pcd, labels = label_trunk_by_slope(
        pcd, cx, cy, slope_info["slope_down"]
    )''',
    '''    # Tâm trục (để project radial)
    pts = np.asarray(pcd.points)
    cx, cz = geometry_utils.horizontal_center(pts)

    # Gán nhãn upslope/downslope
    labeled_pcd, labels = label_trunk_by_slope(
        pcd, cx, cz, slope_info["slope_down"]
    )''',
))

# --- 5. analyze_slope(): JSON output field ----------------------------------
SNIPPETS.append((
    "analyze_slope_json_field",
    '        "trunk_center_xy": [cx, cy],',
    '        "trunk_center_xz": [cx, cz],   # doi ten tu trunk_center_xy: gia tri gio la (X,Z)',
))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        print(f"Hay dat file nay cung thu muc voi {TARGET_NAME} roi chay lai.")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    already_all = all(new in content for _, _, new in SNIPPETS)
    if already_all:
        print(f"{TARGET_NAME} (up-axis fix) da duoc va truoc do roi. Khong can lam gi them.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "de tranh lam hong file. Bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_upaxis")
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
    print(f"Da va thanh cong {len(applied)} doan (up-axis fix): {applied}")
    print("Chay lai 06_slope_analysis.py de kiem tra ket qua.")


if __name__ == "__main__":
    main()
