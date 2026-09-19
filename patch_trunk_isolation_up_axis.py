#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trunk_isolation_up_axis.py
==================================
Va 05_trunk_isolation.py: sua loi dung SAI truc lam "chieu cao" trong
remove_ground_plane(), estimate_trunk_axis(), va crop_to_trunk_cylinder().

Da xac nhan (PCA tren mesh sach, xem 04_trim_mesh.py's estimate_gravity(),
KHONG sua file do): the gioi toa do cua bo du lieu nay dung truc Y (cot
index 1) lam phuong THANG DUNG, con X (0) va Z (2) la mat phang NGANG.

05_trunk_isolation.py truoc patch nay lai dung cot 2 (Z) lam "chieu cao"
o ca 3 ham tren - vua khien remove_ground_plane() khong bao gio nhan dien
dung mat dat that (vi no so sanh vector phap tuyen voi truc Z thay vi Y),
vua khien estimate_trunk_axis()/crop_to_trunk_cylinder() tron lan mot truc
ngang that (Y) vao phep tinh "tam/ban kinh ngang" cung X - day chinh la
nguyen nhan tao ra cac khoi diem hinh "L" sai (goc cay + dat dinh lien
thanh mot khoi) da quan sat duoc tren du lieu that.

Patch nay dung module geometry_utils.py (moi, dung chung voi
06_slope_analysis.py va diagnostic_trunk_axis.py) de dinh nghia mot lan
duy nhat truc nao la "len troi", tranh lap lai loi nay o file khac.

KHONG dung chung voi remove_green_vegetation() - ham do dung colors[:,0/1/2]
la kenh mau RGB, khong phai toa do khong gian, nen KHONG bi dung.

CACH DUNG (doc lap voi patch_trunk_isolation_kdtree_anchor.py - ca hai
dung vao 2 vung khac nhau cua file, thu tu chay truoc/sau deu duoc)
------------------------------------------------------------------------
    python patch_trunk_isolation_up_axis.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_trunk_isolation.py"

# Moi snippet la doc lap: (nhan, OLD, NEW). Ap dung tung cai, bao loi ro
# rang neu khong khop thay vi bo qua am tham.
SNIPPETS: list[tuple[str, str, str]] = []

# --- 1. Import geometry_utils ------------------------------------------------
SNIPPETS.append((
    "import_geometry_utils",
    '''import numpy as np

try:
    import open3d as o3d''',
    '''import numpy as np

import geometry_utils

try:
    import open3d as o3d''',
))

# --- 2. Module docstring: "3. Crop theo ban kinh XY..." ----------------------
SNIPPETS.append((
    "module_docstring_xy",
    "  3. Crop theo bán kính XY xung quanh trục thân",
    "  3. Crop theo bán kính X,Z (ngang) xung quanh trục thân",
))

# --- 3. remove_ground_plane(): Z (cot 2) -> Y (cot 1, qua geometry_utils) ----
SNIPPETS.append((
    "remove_ground_plane",
    '''    if len(pcd.points) < 10:
        return pcd

    points = np.asarray(pcd.points)
    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_cut = z_min + (z_max - z_min) * bottom_fraction

    low_idx = np.where(z_vals <= z_cut)[0]
    if len(low_idx) < 50:
        print("  Ground removal: too few low points, skipping")
        return pcd

    pcd_low = pcd.select_by_index(low_idx)
    plane_model, inliers_local = pcd_low.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    if abs(normal[2]) >= min_ground_normal_z:
        # Map inliers từ pcd_low → pcd gốc
        ground_idx_global = low_idx[inliers_local]
        # Thêm buffer: loại thêm điểm ngay trên mặt đất (+5cm)
        ground_z = np.mean(points[ground_idx_global, 2])
        above_ground = z_vals > ground_z + 0.05
        kept_idx = np.where(above_ground)[0]
        result = pcd.select_by_index(kept_idx)
        print(f"  Ground removed: {len(pcd.points)} -> {len(result.points)} pts "
              f"(normal_z={normal[2]:.3f}, ground_z={ground_z:.3f})")
        return result
    else:
        print(f"  Ground plane not detected (normal_z={normal[2]:.3f}), "
              f"using z_min offset fallback")
        # Fallback: loại bỏ điểm quá gần sàn (z < z_min + 10cm)
        kept = pcd.select_by_index(np.where(z_vals > z_min + 0.10)[0])
        print(f"  Fallback z-floor cut: {len(pcd.points)} -> {len(kept.points)} pts")
        return kept''',
    '''    if len(pcd.points) < 10:
        return pcd

    points = np.asarray(pcd.points)
    z_vals = geometry_utils.vertical(points)
    z_min, z_max = z_vals.min(), z_vals.max()
    z_cut = z_min + (z_max - z_min) * bottom_fraction

    low_idx = np.where(z_vals <= z_cut)[0]
    if len(low_idx) < 50:
        print("  Ground removal: too few low points, skipping")
        return pcd

    pcd_low = pcd.select_by_index(low_idx)
    plane_model, inliers_local = pcd_low.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    if abs(normal[geometry_utils.UP_AXIS]) >= min_ground_normal_z:
        # Map inliers từ pcd_low → pcd gốc
        ground_idx_global = low_idx[inliers_local]
        # Thêm buffer: loại thêm điểm ngay trên mặt đất (+5cm)
        ground_z = np.mean(points[ground_idx_global, geometry_utils.UP_AXIS])
        above_ground = z_vals > ground_z + 0.05
        kept_idx = np.where(above_ground)[0]
        result = pcd.select_by_index(kept_idx)
        print(f"  Ground removed: {len(pcd.points)} -> {len(result.points)} pts "
              f"(normal_vert={normal[geometry_utils.UP_AXIS]:.3f}, ground_z={ground_z:.3f})")
        return result
    else:
        print(f"  Ground plane not detected (normal_vert={normal[geometry_utils.UP_AXIS]:.3f}), "
              f"using z_min offset fallback")
        # Fallback: loại bỏ điểm quá gần sàn (z < z_min + 10cm)
        kept = pcd.select_by_index(np.where(z_vals > z_min + 0.10)[0])
        print(f"  Fallback z-floor cut: {len(pcd.points)} -> {len(kept.points)} pts")
        return kept''',
))

# --- 4. estimate_trunk_axis(): Z->Y cho "cao", XY->XZ cho tam ngang ----------
SNIPPETS.append((
    "estimate_trunk_axis",
    '''    """    python diagnostic_trunk_axis.py D:/Backup/Thucdia-18May2026/cay_0007_1805
    Ước tính tâm trục thân cây (cx, cy) bằng cách:
    - Lấy 40% điểm cao nhất (ít bị nhiễu mặt đất nhất)
    - Tính trung vị XY của vùng đó

    Trả về (cx, cy) — tâm trục thân cây trên mặt phẳng XY.
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return 0.0, 0.0

    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_threshold = z_max - (z_max - z_min) * upper_fraction

    upper_mask = z_vals >= z_threshold
    upper_pts = points[upper_mask]

    if len(upper_pts) < 5:
        # Fallback: dùng toàn bộ điểm
        upper_pts = points

    cx = float(np.median(upper_pts[:, 0]))
    cy = float(np.median(upper_pts[:, 1]))
    print(f"  Trunk axis estimated at XY=({cx:.3f}, {cy:.3f}) "
          f"from top {upper_fraction*100:.0f}% ({upper_mask.sum()} pts)")
    return cx, cy''',
    '''    """
    Ước tính tâm trục thân cây (cx, cz) bằng cách:
    - Lấy 40% điểm cao nhất (ít bị nhiễu mặt đất nhất)
    - Tính trung vị mặt phẳng ngang (X, Z) của vùng đó

    Trả về (cx, cz) — tâm trục thân cây trên mặt phẳng ngang (X, Z).
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return 0.0, 0.0

    z_vals = geometry_utils.vertical(points)
    z_min, z_max = z_vals.min(), z_vals.max()
    z_threshold = z_max - (z_max - z_min) * upper_fraction

    upper_mask = z_vals >= z_threshold
    upper_pts = points[upper_mask]

    if len(upper_pts) < 5:
        # Fallback: dùng toàn bộ điểm
        upper_pts = points

    cx, cz = geometry_utils.horizontal_center(upper_pts)
    print(f"  Trunk axis estimated at XZ=({cx:.3f}, {cz:.3f}) "
          f"from top {upper_fraction*100:.0f}% ({upper_mask.sum()} pts)")
    return cx, cz''',
))

# --- 5. Section banner comment before crop_to_trunk_cylinder -----------------
SNIPPETS.append((
    "cylinder_banner_comment",
    "# 3. Cylinder Crop (XY radius around trunk axis)",
    "# 3. Cylinder Crop (horizontal X,Z radius around trunk axis)",
))

# --- 6. crop_to_trunk_cylinder(): cy->cz, XY dist -> XZ dist -----------------
SNIPPETS.append((
    "crop_to_trunk_cylinder",
    '''def crop_to_trunk_cylinder(
    pcd: o3d.geometry.PointCloud,
    cx: float,
    cy: float,
    radius: float = 0.30,
    z_min_offset: float = 0.05,
) -> o3d.geometry.PointCloud:
    """
    Giữ lại điểm trong bán kính `radius` tính theo XY (trụ đứng).
    Loại bỏ thêm những điểm quá gần mặt đất (z < z_min + z_min_offset).
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    # XY distance từ trục thân
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    xy_dist = np.sqrt(dx**2 + dy**2)

    # Bỏ điểm ngay sát đất (rễ cây, lá rụng)
    z_vals = points[:, 2]
    z_floor = z_vals.min() + z_min_offset

    mask = (xy_dist <= radius) & (z_vals >= z_floor)
    cropped = pcd.select_by_index(np.where(mask)[0])
    print(f"  Cylinder crop r={radius}m: {len(points)} -> {len(cropped.points)} pts")
    return cropped''',
    '''def crop_to_trunk_cylinder(
    pcd: o3d.geometry.PointCloud,
    cx: float,
    cz: float,
    radius: float = 0.30,
    z_min_offset: float = 0.05,
) -> o3d.geometry.PointCloud:
    """
    Giữ lại điểm trong bán kính `radius` tính theo mặt phẳng ngang X,Z (trụ đứng).
    Loại bỏ thêm những điểm quá gần mặt đất (chiều cao < min + z_min_offset).
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    # Khoang cach tren mat phang ngang (X, Z) tu truc than
    xz_dist = geometry_utils.horizontal_distance(points, (cx, cz))

    # Bỏ điểm ngay sát đất (rễ cây, lá rụng)
    z_vals = geometry_utils.vertical(points)
    z_floor = z_vals.min() + z_min_offset

    mask = (xz_dist <= radius) & (z_vals >= z_floor)
    cropped = pcd.select_by_index(np.where(mask)[0])
    print(f"  Cylinder crop r={radius}m: {len(points)} -> {len(cropped.points)} pts")
    return cropped''',
))

# --- 7. Call sites inside isolate_trunk() ------------------------------------
SNIPPETS.append((
    "call_site_estimate",
    "    cx, cy = estimate_trunk_axis(pcd, upper_fraction=0.6)",
    "    cx, cz = estimate_trunk_axis(pcd, upper_fraction=0.6)",
))
SNIPPETS.append((
    "call_site_crop",
    "    pcd = crop_to_trunk_cylinder(pcd, cx, cy, radius=trunk_radius)",
    "    pcd = crop_to_trunk_cylinder(pcd, cx, cz, radius=trunk_radius)",
))

# --- 8. CLI help text ---------------------------------------------------------
SNIPPETS.append((
    "cli_help",
    '        help="XY cylinder radius in meters to keep around trunk axis (default: 0.30)"',
    '        help="Horizontal (X,Z) cylinder radius in meters to keep around trunk axis (default: 0.30)"',
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
            continue  # da co san (idempotent), bo qua rieng doan nay
        content = content.replace(old, new, 1)
        applied.append(label)

    target.write_text(content, encoding="utf-8")
    print(f"Da va thanh cong {len(applied)} doan (up-axis fix): {applied}")
    print("Chay lai buoc co lap than cay (buoc 3/8) de kiem tra ket qua.")


if __name__ == "__main__":
    main()
