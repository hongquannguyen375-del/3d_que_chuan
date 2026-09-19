#!/usr/bin/env python3
"""
Step 5: Trunk Isolation — Tách thân cây quế khỏi nhiễu nền.

Vấn đề với pointcloud raw:
  - Có nhiều thực vật mặt đất (cỏ, rêu xanh)
  - Cây khác ở phía sau
  - Điểm lơ lửng (floating noise)
  - Mặt đất (ground plane)

Pipeline:
  1. RANSAC ground plane removal — loại bỏ mặt phẳng đất
  2. Ước tính trục thân cây từ vùng trên cao (ít bị nhiễu nhất)
  3. Crop theo bán kính X,Z (ngang) xung quanh trục thân
  4. Statistical Outlier Removal — loại điểm lẻ
  5. Lọc theo màu (tuỳ chọn) — loại lá xanh còn sót
  6. Lưu trunk_pointcloud.ply

Usage:
    python 05_trunk_isolation.py E:/3D_Que/Raw_data/26Q1
    python 05_trunk_isolation.py E:/3D_Que/Raw_data/26Q1 --trunk-radius 0.3
    python 05_trunk_isolation.py E:/3D_Que/Raw_data  --batch   (xử lý tất cả 92 mẫu)
"""

import argparse
import os
import sys

import numpy as np

import geometry_utils

try:
    import open3d as o3d
except Exception as exc:
    print("ERROR: Failed to import open3d. This is usually an environment/dependency issue, not bad scan data.")
    print("  Suggested fix:")
    print("    .\\venv\\Scripts\\python.exe -m pip install --force-reinstall open3d==0.19.0 dash==4.4.1 pydantic==2.13.4")
    print("    or recreate the venv and install requirements.txt")
    raise


# ---------------------------------------------------------------------------
# 1. Ground Plane Removal (RANSAC)
# ---------------------------------------------------------------------------

def remove_ground_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.15,      # ← 0.08 → 0.15
    ransac_n: int = 3,
    num_iterations: int = 2000,
    min_ground_normal_z: float = 0.60,     # ← 0.70 → 0.60
    bottom_fraction: float = 0.15,         # ← 0.25 → 0.15
) -> o3d.geometry.PointCloud:
    """
    Fit RANSAC plane trên vùng z thấp, loại bỏ nếu là mặt ngang.

    Chiến lược:
    - Chỉ chạy RANSAC trên bottom_fraction điểm thấp nhất
      (tránh RANSAC tìm tường/thân cây thay vì đất)
    - Chấp nhận mặt phẳng nếu normal_z >= min_ground_normal_z
    - distance_threshold cao hơn để bắt đất dốc
    """
    if len(pcd.points) < 10:
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
        return kept


# ---------------------------------------------------------------------------
# 2. Trunk Axis Estimation
# ---------------------------------------------------------------------------

def estimate_target_axis_from_views(
    dataset_dir: str,
    min_poses: int = 20,
    max_ray_residual: float = 0.25,
    min_conditioning: float = 0.15,
    min_front_fraction: float = 0.60,
    verbose: bool = True,
) -> "tuple[float, float] | None":
    """
    Xac dinh CAY DICH bang giao diem cac TIA NHIN cua camera.

    Nguoi quay chia camera vao cay dich suot buoi (CLAUDE.md muc 4), nen giao
    diem binh phuong toi thieu cua moi tia nhin ngang chinh la cay do. Day la
    co che DUY NHAT trong pipeline thuc su tra loi "cay nao la cay dich":
    estimate_trunk_axis() chi lay mot trung vi (X,Z), nen khi trong khung co
    tu hai cay tro len no roi vao cho tuy y -- do duoc lech toi 97cm tren 8
    scan, ngay ca khi cay that co nhieu diem gap 4 lan cay bi chon.

    Huong nhin la R @ (0,0,+1) voi R = quaternion cua odometry.csv (R la
    camera->world; xem pose_to_extrinsic() trong 02_pointcloud_and_mesh.py va
    patch_slope_analysis_gravity_frame.py).

    Tra ve (cx, cz), hoac None neu bat ky chot an toan nao khong dat -- khi do
    goi noi nen dung lai estimate_trunk_axis(). Nguong lay tu so do that tren
    49 scan, moi nguong de bien it nhat 4 lan (xem docstring cua
    patch_trunk_isolation_view_anchor.py).
    """
    import csv as _csv

    odo_path = os.path.join(dataset_dir, "odometry.csv")
    if not os.path.exists(odo_path):
        if verbose:
            print("  Neo huong nhin: BO QUA (khong co odometry.csv)")
        return None

    pos, quat = [], []
    try:
        with open(odo_path, "r", newline="") as f:
            for row in _csv.DictReader(f, skipinitialspace=True):
                try:
                    pos.append([float(row["x"]), float(row["z"])])
                    quat.append([float(row["qx"]), float(row["qy"]),
                                 float(row["qz"]), float(row["qw"])])
                except (KeyError, ValueError):
                    continue
    except Exception as e:
        if verbose:
            print(f"  Neo huong nhin: BO QUA (khong doc duoc odometry.csv: {e})")
        return None

    if len(pos) < min_poses:
        if verbose:
            print(f"  Neo huong nhin: BO QUA (chi co {len(pos)} pose)")
        return None

    o = np.asarray(pos, dtype=float)
    q = np.asarray(quat, dtype=float)

    # Huong nhin +Z cua tung pose, chieu xuong mat phang ngang (X, Z).
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    dx = 2.0 * (qx * qz + qy * qw)
    dz = 1.0 - 2.0 * (qx * qx + qy * qy)
    d = np.stack([dx, dz], axis=1)
    nrm = np.linalg.norm(d, axis=1)
    keep = np.isfinite(nrm) & (nrm > 1e-6)
    if keep.sum() < min_poses:
        if verbose:
            print("  Neo huong nhin: BO QUA (huong nhin gan nhu thang dung)")
        return None
    o, d = o[keep], d[keep] / nrm[keep, None]

    # Binh phuong toi thieu: diem gan nhat toi moi duong thang (o_i, d_i).
    # A = sum (I - d d^T),  b = sum (I - d d^T) o
    outer = np.einsum("ni,nj->nij", d, d)
    M = np.eye(2)[None, :, :] - outer
    A = M.sum(axis=0)
    b = np.einsum("nij,nj->i", M, o)

    ev = np.linalg.eigvalsh(A)
    cond = float(ev.min() / max(ev.max(), 1e-12))
    if not np.isfinite(cond) or cond < min_conditioning:
        if verbose:
            print(f"  Neo huong nhin: BO QUA (cac tia gan nhu song song, "
                  f"dieu kien {cond:.3f} < {min_conditioning})")
        return None

    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        if verbose:
            print("  Neo huong nhin: BO QUA (he phuong trinh suy bien)")
        return None

    perp = np.linalg.norm(np.einsum("nij,nj->ni", M, p[None, :] - o), axis=1)
    resid = float(np.median(perp))
    front = float(np.mean(np.einsum("ni,ni->n", p[None, :] - o, d) > 0))

    if resid > max_ray_residual:
        if verbose:
            print(f"  Neo huong nhin: BO QUA (cac tia khong hoi tu, "
                  f"residual {resid:.3f}m > {max_ray_residual}m)")
        return None
    if front < min_front_fraction:
        if verbose:
            print(f"  Neo huong nhin: BO QUA (moc nam phia sau camera, "
                  f"chi {front:.0%} < {min_front_fraction:.0%} nhin ve phia no)")
        return None

    if verbose:
        print(f"  Neo huong nhin camera: XZ=({p[0]:.3f}, {p[1]:.3f}) tu "
              f"{int(keep.sum())} pose [residual={resid*100:.1f}cm, "
              f"dieu kien={cond:.2f}, phia truoc={front:.0%}]")
    return float(p[0]), float(p[1])


def estimate_trunk_axis(
    pcd: o3d.geometry.PointCloud,
    upper_fraction: float = 0.4,
) -> tuple[float, float]:
    """
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
    return cx, cz


# ---------------------------------------------------------------------------
# 3. Cylinder Crop (horizontal X,Z radius around trunk axis)
# ---------------------------------------------------------------------------

def crop_to_trunk_cylinder(
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
    return cropped


# ---------------------------------------------------------------------------
# 3b. Adaptive per-slice cylinder crop (tam rieng cho tung lat chieu cao)
# ---------------------------------------------------------------------------

def _slice_blobs_2d(x_h, z_h, idx, eps: float, min_points: int) -> list:
    """
    Phan cum 2D tren mat phang ngang (X,Z) trong MOT lat chieu cao.
    Tra ve list cac blob dang ((cx, cz), n_pts, r_med).
    """
    n = len(idx)
    if n < min_points:
        return []
    xyz = np.zeros((n, 3))
    xyz[:, geometry_utils.HORIZONTAL_AXES[0]] = x_h[idx]
    xyz[:, geometry_utils.HORIZONTAL_AXES[1]] = z_h[idx]
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz)
    labels = np.array(p.cluster_dbscan(eps=eps, min_points=min_points,
                                       print_progress=False))
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return []
    uniq, counts = np.unique(valid, return_counts=True)
    out = []
    for u, c in zip(uniq, counts):
        if c < max(min_points, 0.02 * n):
            continue
        bm = labels == u
        bx, bz = x_h[idx][bm], z_h[idx][bm]
        ccx, ccz = float(bx.mean()), float(bz.mean())
        d = np.sqrt((bx - ccx) ** 2 + (bz - ccz) ** 2)
        out.append(((ccx, ccz), int(c), float(np.median(d))))
    return out


def crop_to_trunk_cylinder_adaptive(
    pcd: o3d.geometry.PointCloud,
    cx: float,
    cz: float,
    radius: float = 0.30,
    wide_radius: float = 0.80,
    n_slices: int = 40,
    eps: float = 0.03,
    min_points: int = 10,
    max_jump: float = 0.10,
    seed_max_offset: float = 0.40,
    max_coast_ratio: float = 0.50,
    max_radius_ratio: float = 3.0,
    z_min_offset: float = 0.05,
) -> o3d.geometry.PointCloud:
    """
    Crop hinh tru THICH UNG: moi lat chieu cao co tam rieng, bam theo quy
    dao that cua than cay, thay vi dung MOT tam co dinh cho ca scan.

    Ly do (xac nhan tren du lieu that cay_0004_1805, xem CLAUDE.md muc 12):
    estimate_trunk_axis() tinh mot trung vi (X,Z) duy nhat tu 60% diem cao
    nhat. Neu scan co khoi to khong phai than cay (re/goc phinh, dat, rac)
    thi trung vi do bi keo lech -- do duoc 22.6-29.5cm khoi tam than that
    tren cay_0004, gan bang dung ban kinh crop (30cm), khien than cay nam
    sat mep vung crop va bi "cat doc" mat mot nua. Tang ban kinh KHONG sua
    duoc (chi keo them rac vao); phai sua bang cach bam dung tam that.

    Thuat toan:
      1. Crop tho rong (wide_radius) quanh (cx, cz) -- chi de gioi han vung
         tim kiem.
      2. Cat thanh n_slices lat; moi lat phan cum 2D (X,Z) de lay blob.
      3. Seed: trong 60% tren chieu cao, lay ban kinh blob lon nhat cua moi
         lat, tinh trung vi; seed = lat co ban kinh gan trung vi nhat.
      4. Lan toa len/xuong tu seed: moi lat chon blob co tam GAN NHAT voi
         tam lat lien truoc (KHONG phai blob nhieu diem nhat -- khoi goc to
         thuong nhieu diem hon than that, chon theo so diem se khoa nham
         vao no). Neu khong blob nao trong max_jump -> giu nguyen tam lat
         truoc (coast), khong doan bua.
      5. Cat cuoi: giu diem neu nam trong `radius` quanh tam CUA LAT no.

    Tu dong fallback ve crop_to_trunk_cylinder() co dinh neu qua it diem
    hoac khong tim duoc lat seed.
    """
    pcd_wide = crop_to_trunk_cylinder(pcd, cx, cz, radius=wide_radius,
                                      z_min_offset=z_min_offset)
    pcd_wide = denoise(pcd_wide, nb_neighbors=20, std_ratio=2.5)
    pts = np.asarray(pcd_wide.points)
    if len(pts) < min_points * 2:
        print("  Adaptive crop: qua it diem -> fallback crop co dinh")
        return crop_to_trunk_cylinder(pcd, cx, cz, radius=radius,
                                      z_min_offset=z_min_offset)

    y = geometry_utils.vertical(pts)
    x_h = pts[:, geometry_utils.HORIZONTAL_AXES[0]]
    z_h = pts[:, geometry_utils.HORIZONTAL_AXES[1]]
    y_min, y_max = float(y.min()), float(y.max())
    if y_max - y_min < 1e-6:
        return crop_to_trunk_cylinder(pcd, cx, cz, radius=radius,
                                      z_min_offset=z_min_offset)

    edges = np.linspace(y_min, y_max, n_slices + 1)
    slice_idx = []
    slice_blobs = []
    for i in range(n_slices):
        lo, hi = edges[i], edges[i + 1]
        if i < n_slices - 1:
            m = (y >= lo) & (y < hi)
        else:
            m = (y >= lo) & (y <= hi)
        idx = np.where(m)[0]
        slice_idx.append(idx)
        slice_blobs.append(_slice_blobs_2d(x_h, z_h, idx, eps, min_points))

    # CHOT CHAN SEED: ung vien phai nam gan truc tho (cx, cz). Neu khong co
    # rang buoc nay, scan co CAY KHAC dung sat se khien seed khoa nham vao
    # cay hang xom -- da xay ra that tren cay_0013_1805 (cay hang xom cach
    # ~55-60cm nhung o dai do cao 1.22-1.61m lai nhieu diem hon cay dich),
    # keo theo ca chuoi bam sai va xuat ra dam may vo dinh hinh. Nguong
    # 0.40m do bang so that: than dich cua cay_0004 lech toi da 29.5cm khoi
    # truc tho (phai giu), cay hang xom cua cay_0013 lech toi thieu 55.5cm
    # (phai loai). Xem CLAUDE.md muc 12.
    upper_threshold = y_max - 0.6 * (y_max - y_min)
    cands = []
    for i in range(n_slices):
        if edges[i] < upper_threshold or not slice_blobs[i]:
            continue
        near = [b for b in slice_blobs[i]
                if (b[0][0] - cx) ** 2 + (b[0][1] - cz) ** 2 <= seed_max_offset ** 2]
        if not near:
            continue
        cands.append((i, max(near, key=lambda b: b[1])))
    if not cands:
        print(f"  Adaptive crop: khong co lat seed nao trong "
              f"{seed_max_offset*100:.0f}cm quanh truc tho -> fallback crop co dinh")
        return crop_to_trunk_cylinder(pcd, cx, cz, radius=radius,
                                      z_min_offset=z_min_offset)

    baseline_r = float(np.median([c[1][2] for c in cands]))
    seed_i, seed_blob = min(cands, key=lambda c: abs(c[1][2] - baseline_r))

    centers = [None] * n_slices
    centers[seed_i] = seed_blob[0]
    n_tracked, n_coast = 1, 0
    for rng in (range(seed_i + 1, n_slices), range(seed_i - 1, -1, -1)):
        prev = seed_blob[0]
        for i in rng:
            blobs = slice_blobs[i]
            if not blobs:
                centers[i] = prev
                n_coast += 1
                continue
            best = min(blobs, key=lambda b: (b[0][0] - prev[0]) ** 2
                                          + (b[0][1] - prev[1]) ** 2)
            dist = ((best[0][0] - prev[0]) ** 2
                    + (best[0][1] - prev[1]) ** 2) ** 0.5
            if dist <= max_jump:
                centers[i] = best[0]
                prev = best[0]
                n_tracked += 1
            else:
                centers[i] = prev
                n_coast += 1

    # LUOI AN TOAN: neu qua nua so lat co du lieu bi mat dau thi viec bam
    # tam ro rang da hong (seed sai, hoac scan qua lon xon) -- quay ve crop
    # co dinh cu con hon xuat ra rac. Do tren 6 scan that: ty le coast binh
    # thuong chi 5-25%, nen nguong 50% khong bat nham scan tot.
    n_data = sum(1 for idx in slice_idx if len(idx) >= min_points)
    coast_ratio = n_coast / max(n_data, 1)
    if coast_ratio > max_coast_ratio:
        print(f"  Adaptive crop: mat dau {n_coast}/{n_data} lat "
              f"({coast_ratio:.0%} > {max_coast_ratio:.0%}) -> bam hong, "
              f"fallback crop co dinh")
        return crop_to_trunk_cylinder(pcd, cx, cz, radius=radius,
                                      z_min_offset=z_min_offset)

    # Cat theo tam rieng tung lat, dong thoi LOAI LAT CO BAN KINH BAT
    # THUONG. Xem CLAUDE.md muc 12: tren cay_0029_1805, 6 lat day nam trong
    # khoi dat/bui ram day 0.53m ma remove_ground_plane() khong cat het;
    # chung co r_med 16-25cm (nen chi 3.5cm) va gop 58% tong so diem, khien
    # keep_largest_cluster() phia sau -- von chi dem diem -- giu khoi dat va
    # VUT DI than cay. Than cay that giu ban kinh trong vai lan ban kinh
    # nen; da do tren 6 scan: k=2.0 qua chat (duc lo giua than cay_0007),
    # k=4.0 qua long (cay_0010 lot), k=3.0 tach sach ca 6.
    slice_r_med = np.zeros(n_slices)
    slice_d = [None] * n_slices
    for i in range(n_slices):
        idx = slice_idx[i]
        if len(idx) == 0 or centers[i] is None:
            continue
        ccx, ccz = centers[i]
        d = np.sqrt((x_h[idx] - ccx) ** 2 + (z_h[idx] - ccz) ** 2)
        slice_d[i] = d
        inr = d <= radius
        if inr.any():
            slice_r_med[i] = float(np.median(d[inr]))

    n_drop_slices, n_drop_pts = 0, 0
    keep = np.zeros(len(pts), dtype=bool)
    for i in range(n_slices):
        idx = slice_idx[i]
        if slice_d[i] is None:
            continue
        inr = slice_d[i] <= radius
        if max_radius_ratio > 0 and baseline_r > 1e-9 and \
                slice_r_med[i] > max_radius_ratio * baseline_r:
            n_drop_slices += 1
            n_drop_pts += int(inr.sum())
            continue
        keep[idx[inr]] = True

    out = pcd_wide.select_by_index(np.where(keep)[0])
    print(f"  Adaptive crop (r={radius}m, {n_slices} lat, max_jump={max_jump*100:.0f}cm, "
          f"seed=lat {seed_i} r_med={baseline_r*100:.1f}cm): "
          f"{len(pts)} -> {len(out.points)} pts "
          f"[tracked={n_tracked}, coast={n_coast}/{n_data} = {coast_ratio:.0%}, "
          f"loai {n_drop_slices} lat ban kinh >{max_radius_ratio:.1f}x nen "
          f"({n_drop_pts:,} pts)]")
    return out


# ---------------------------------------------------------------------------
# 4. Statistical Outlier Removal
# ---------------------------------------------------------------------------

def denoise(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    if len(pcd.points) < nb_neighbors + 1:
        return pcd
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    cleaned = pcd.select_by_index(ind)
    print(f"  SOR denoised: {len(pcd.points)} -> {len(cleaned.points)} pts")
    return cleaned


# ---------------------------------------------------------------------------
# 5. Green Vegetation Filter (loại lá xanh còn sót)
# ---------------------------------------------------------------------------

def remove_green_vegetation(
    pcd: o3d.geometry.PointCloud,
    green_threshold: float = 0.15,
    use_hue_filter: bool = True,
    veg_hue_min: float = 65.0,
    veg_hue_max: float = 150.0,
    veg_sat_min: float = 0.25,
) -> o3d.geometry.PointCloud:
    """
    Loai diem co mau xanh la cay (la, reu, dia y nhat mau, co).
    Ket hop 2 tieu chi bang OR:
      1. RGB-diff cu (green_threshold): g > r+thr VA g > b+thr -- bat mau
         xanh manh, ro net.
      2. [Sep 2026] Hue+Saturation moi (use_hue_filter): H trong
         [veg_hue_min, veg_hue_max] do va S > veg_sat_min -- CONG THUC Y
         HET is_vegetation trong 05_detect_lichen.py's
         classify_lichen_local_contrast(). Xem CLAUDE.md muc 12: xac nhan
         tren du lieu that (cay_0007_1805) rang reu/dia y nhat mau (do anh
         sang lom dom qua tan la) co kenh g-r chi ~0.06 -- duoi xa nguong
         0.15 -- khien tieu chi (1) gan nhu khong bao gio kich hoat (chi
         151/90328 diem, 0.2%), du Hue (79 do) va Saturation (0.53) cua
         chung van tach biet ro voi vo cay (47 do, 0.24). Ha thap
         green_threshold khong giai duoc goc re (quet nguong tren du lieu
         that: 0.06 chi bat ~49% voi 82% chinh xac, thap hon nua bat nham
         vo cay ngay cang nhieu) -- vi RGB-diff khong phai phep do phu hop
         cho loai mau nhat nay.
    """
    if not pcd.has_colors():
        return pcd

    colors = np.asarray(pcd.colors)  # [0,1] RGB
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]

    # Tieu chi 1: RGB-diff cu (mau xanh manh, ro net)
    is_green_rgb = (g > r + green_threshold) & (g > b + green_threshold)

    is_green_hue = np.zeros(len(r), dtype=bool)
    if use_hue_filter:
        # Tieu chi 2: Hue+Saturation (bat reu/dia y nhat mau hon)
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        delta = maxc - minc
        sat = np.where(maxc > 1e-9, delta / np.maximum(maxc, 1e-9), 0.0)
        hue = np.zeros_like(r)
        dnz = delta > 1e-9
        mask_r = (maxc == r) & dnz
        mask_g = (maxc == g) & dnz
        mask_b = (maxc == b) & dnz
        hue[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
        hue[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
        hue[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
        is_green_hue = (hue >= veg_hue_min) & (hue <= veg_hue_max) & (sat > veg_sat_min)

    is_green = is_green_rgb | is_green_hue
    not_green = ~is_green
    kept = pcd.select_by_index(np.where(not_green)[0])
    removed = len(pcd.points) - len(kept.points)
    n_rgb = int(is_green_rgb.sum())
    n_hue_only = int((is_green_hue & ~is_green_rgb).sum())
    print(f"  Green filter: removed {removed} pts ({100*removed/max(len(pcd.points),1):.1f}%) "
          f"[rgb-diff={n_rgb}, hue-sat-only={n_hue_only}]")
    return kept


# ---------------------------------------------------------------------------
# 6. Keep Largest Connected Cluster (DBSCAN)
# ---------------------------------------------------------------------------

def keep_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.05,
    min_points: int = 50,
) -> o3d.geometry.PointCloud:
    """
    Dùng DBSCAN để tìm cluster lớn nhất — thường là thân cây chính.
    """
    if len(pcd.points) < min_points:
        return pcd

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() < 0:
        print("  DBSCAN: no clusters found, keeping all")
        return pcd

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_label = unique[np.argmax(counts)]

    mask = labels == largest_label
    kept = pcd.select_by_index(np.where(mask)[0])
    print(f"  DBSCAN largest cluster: {counts.max()} / {len(pcd.points)} pts "
          f"({len(unique)} clusters total)")
    return kept


# ---------------------------------------------------------------------------
# 6b. Per-slice declutter (loai canh cay / vat the phu dinh gan truc)
# ---------------------------------------------------------------------------

def remove_off_axis_clusters_per_slice(
    pcd: o3d.geometry.PointCloud,
    n_slices: int = 40,
    eps: float = 0.03,
    min_points: int = 10,
    verbose: bool = True,
) -> o3d.geometry.PointCloud:
    """
    Cat than cay thanh n_slices lat mong theo chieu cao (geometry_utils.
    vertical), trong MOI lat chay DBSCAN 2D tren mat phang ngang (X,Z) va
    chi giu cum LON NHAT. Xem CLAUDE.md muc 12: xac nhan tren du lieu that
    (cay_0007_1805) rang mot canh cay dinh vao giua than (~0.8m trong ~3m,
    lech ~25-30cm khoi truc) bi DBSCAN 3D (keep_largest_cluster(), chay
    xuyen suot ca khoi) nhap chung vao cum than vi no "dinh" lien tuc theo
    chieu cao -- nhung khi xu ly TUNG LAT DOC LAP thi tach ro rang thanh 2
    cum rieng (cum than ~1200-3700 diem, cum canh chi ~80-880 diem). Ham
    nay khai thac dung diem khac biet do.

    Lat co qua it diem (< min_points) de phan cum tin cay thi GIU NGUYEN,
    khong xu ly -- tranh cat nham khi khong chac chan (cung triet ly voi
    tieu chi sparse-only cua 04b_finalize_mesh.py's trim_ragged_top()).
    """
    points = np.asarray(pcd.points)
    n_pts = len(points)
    if n_pts < min_points * 2:
        return pcd

    y = geometry_utils.vertical(points)
    xz = geometry_utils.horizontal(points)
    y_min, y_max = float(y.min()), float(y.max())
    if y_max - y_min < 1e-6:
        return pcd

    edges = np.linspace(y_min, y_max, n_slices + 1)
    keep_mask = np.zeros(n_pts, dtype=bool)

    for i in range(n_slices):
        lo, hi = edges[i], edges[i + 1]
        if i == n_slices - 1:
            in_slice = (y >= lo) & (y <= hi)
        else:
            in_slice = (y >= lo) & (y < hi)
        idx = np.where(in_slice)[0]
        n_slice = len(idx)
        if n_slice < min_points:
            keep_mask[idx] = True
            continue

        slice_xyz = np.zeros((n_slice, 3))
        slice_xyz[:, 0] = xz[idx, 0]
        slice_xyz[:, 2] = xz[idx, 1]
        slice_pcd = o3d.geometry.PointCloud()
        slice_pcd.points = o3d.utility.Vector3dVector(slice_xyz)
        labels = np.array(slice_pcd.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=False))
        valid = labels[labels >= 0]
        if len(valid) == 0:
            keep_mask[idx] = True
            continue

        uniq, counts = np.unique(valid, return_counts=True)
        largest_label = uniq[np.argmax(counts)]
        keep_mask[idx[labels == largest_label]] = True

    n_removed = n_pts - int(keep_mask.sum())
    if verbose:
        print(f"  Per-slice declutter ({n_slices} lat, eps={eps}): "
              f"remove {n_removed:,} / {n_pts:,} pts (canh/vat the phu dinh gan truc)")

    if n_removed == 0:
        return pcd
    return pcd.select_by_index(np.where(keep_mask)[0])


# ---------------------------------------------------------------------------
# Main isolation pipeline
# ---------------------------------------------------------------------------

def isolate_trunk(
    dataset_dir: str,
    trunk_radius: float = 0.30,
    remove_green: bool = True,
    use_dbscan: bool = True,
    declutter_slices: bool = True,
    hue_green_filter: bool = True,
    adaptive_crop: bool = True,
    max_jump: float = 0.10,
    seed_max_offset: float = 0.40,
    max_radius_ratio: float = 3.0,
    view_anchor: bool = True,
) -> bool:
    """
    Đọc pointcloud.ply từ output/, chạy pipeline làm sạch, lưu trunk_pointcloud.ply.
    Trả về True nếu thành công.
    """
    pcd_path = os.path.join(dataset_dir, "output", "pointcloud.ply")
    if not os.path.exists(pcd_path):
        print(f"  SKIP: {pcd_path} not found")
        return False

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(dataset_dir)}")
    print(f"{'='*60}")
    print(f"  Loading {pcd_path} ...")

    pcd = o3d.io.read_point_cloud(pcd_path)
    n_orig = len(pcd.points)
    print(f"  Loaded {n_orig} points")

    if n_orig == 0:
        print("  SKIP: empty point cloud")
        return False

    # --- Loc phong ve: bo diem co toa do bat thuong (NaN/Inf/qua xa) ---
    # File .ply doi khi bi loi ghi/doc, tao ra vai diem co toa do "rac"
    # (vi du ~1e260 met) khien cac ham KDTree/RANSAC cua Open3D bi crash
    # (STATUS_STACK_BUFFER_OVERRUN). Loc som o day de tranh hoan toan.
    raw_points = np.asarray(pcd.points)
    SANE_LIMIT = 50.0  # met - khong scan Stray Scanner nao rong hon muc nay
    sane_mask = (
        np.isfinite(raw_points).all(axis=1)
        & (np.abs(raw_points) <= SANE_LIMIT).all(axis=1)
    )
    n_bad = n_orig - int(sane_mask.sum())
    if n_bad > 0:
        print(f"  CANH BAO: loai {n_bad} diem co toa do bat thuong "
              f"(NaN/Inf/qua {SANE_LIMIT}m) truoc khi xu ly.")
        pcd = pcd.select_by_index(np.where(sane_mask)[0])
        n_orig = len(pcd.points)
        if n_orig == 0:
            print("  SKIP: point cloud rong sau khi loc diem bat thuong")
            return False

    # --- Neo theo quy dao camera bang cKDTree 3D dang huong ---
    # Tinh khoang cach Euclid 3D that su tu moi diem den camera pose GAN
    # NHAT (khong phai mot tam trung binh don, khong tach rieng "ngang" va
    # "cao"), dua tren toan bo quy dao camera. Vi khong tach truc nao ca,
    # cach nay KHONG PHU THUOC vao truc nao la chieu cao trong du lieu -
    # tranh hoan toan lop loi da gap voi cach cu (xem patch_trunk_isolation
    # _up_axis.py). Giong het filter_by_camera_distance() trong
    # 01_clean_pointcloud.py. Da xac nhan qua 2 ca loi thuc te:
    # cay_0031_1805 (chon nham cum theo chieu ngang) va cay_0007_1805
    # (chon nham cum cung XY nhung khac do cao) - ca hai deu duoc xu ly.
    from scipy.spatial import cKDTree as _cKDTree
    import csv as _csv
    CAMERA_ANCHOR_RADIUS = 1.0  # met - toi camera pose gan nhat
    odo_path = os.path.join(dataset_dir, "odometry.csv")
    if os.path.exists(odo_path):
        cam_xyz = []
        try:
            with open(odo_path, "r", newline="") as _f:
                _reader = _csv.DictReader(_f, skipinitialspace=True)
                for _row in _reader:
                    try:
                        cam_xyz.append([float(_row["x"]), float(_row["y"]), float(_row["z"])])
                    except (KeyError, ValueError):
                        continue
        except Exception as e:
            cam_xyz = []
            print(f"  CANH BAO: khong doc duoc odometry.csv de neo camera: {e}")
        if cam_xyz:
            cam_xyz = np.array(cam_xyz, dtype=np.float32)
            all_pts = np.asarray(pcd.points, dtype=np.float32)
            tree = _cKDTree(cam_xyz)
            dist_to_cam, _ = tree.query(all_pts, k=1, workers=-1)
            anchor_mask = dist_to_cam <= CAMERA_ANCHOR_RADIUS
            n_before_anchor = len(pcd.points)
            n_kept_anchor = int(anchor_mask.sum())
            if n_kept_anchor > 0:
                pcd = pcd.select_by_index(np.where(anchor_mask)[0])
                n_orig = len(pcd.points)
                print(f"  Neo theo quy dao camera (cKDTree 3D, r={CAMERA_ANCHOR_RADIUS}m "
                      f"toi camera pose gan nhat): {n_before_anchor:,} -> {n_orig:,} pts")
            else:
                print(f"  CANH BAO: khong con diem nao trong ban kinh "
                      f"{CAMERA_ANCHOR_RADIUS}m toi bat ky camera pose nao - "
                      f"giu nguyen point cloud, bo qua buoc neo camera.")
    else:
        print(f"  CANH BAO: khong tim thay {odo_path}, bo qua buoc neo "
              f"theo quy dao camera.")

    # --- Downsample som de giam RAM/thoi gian cho RANSAC/KDTree/DBSCAN ---
    # Voi may RAM han che, point cloud qua nhieu diem (hang trieu) lam cac
    # buoc phia sau cham hoac treo do swap bo nho. Voxel 1cm van du chi
    # tiet de xac dinh hinh dang than cay (chi tiet dia y lay tu TSDF mesh
    # o buoc rieng, khong phu thuoc vao day).
    if n_orig > 1_000_000:
        before = n_orig
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        n_orig = len(pcd.points)
        print(f"  Downsample (voxel 1cm): {before:,} -> {n_orig:,} pts")

    # Step 1: Loại bỏ ground plane
    pcd = remove_ground_plane(pcd)
    
    # Sau ground removal, TRƯỚC estimate_trunk_axis:
    pcd = denoise(pcd, nb_neighbors=20, std_ratio=3.0)  # Nới lỏng hơn
    
    # Step 2: Xac dinh CAY DICH.
    # [Sep 2026] estimate_trunk_axis() chi lay MOT trung vi (X,Z) tren 60%
    # diem cao nhat, nen khi trong khung co tu hai cay tro len no roi vao cho
    # tuy y -- do duoc lech toi 97cm tren 8 scan, va tren cay_0051 cay that co
    # nhieu diem GAP 4 LAN cay bi chon ma van thua (nen khong luat dem diem
    # nao cuu duoc). Moc dang tin cay la giao diem cac TIA NHIN cua camera:
    # sai 0.00-0.08m tren 37 scan chay tot, va chi cach truc trung vi cu
    # 0.8-1.8cm o cac scan von da dung. Xem docstring
    # estimate_target_axis_from_views() + CLAUDE.md muc 14.
    cx, cz = estimate_trunk_axis(pcd, upper_fraction=0.6)
    if view_anchor:
        _anchor = estimate_target_axis_from_views(dataset_dir)
        if _anchor is not None:
            _ax, _az = _anchor
            _d = float(np.hypot(_ax - cx, _az - cz))
            if _d > 0.30:
                print(f"  CANH BAO: truc trung vi ({cx:.3f}, {cz:.3f}) cach moc "
                      f"huong nhin {_d*100:.0f}cm -- gan nhu chac chan no dang "
                      f"chi sang CAY KHAC. Dung moc huong nhin.")
            else:
                print(f"  (truc trung vi cach moc huong nhin {_d*100:.1f}cm)")
            cx, cz = _ax, _az

    # Step 3: Crop hình trụ quanh trục thân
    # [Sep 2026] Mac dinh dung crop THICH UNG theo tung lat -- truc co dinh
    # cu bi khoi goc/re phinh keo lech tam, cat mat mot nua than cay theo
    # chieu doc (xem docstring crop_to_trunk_cylinder_adaptive + CLAUDE.md
    # muc 12). --no-adaptive-crop de quay ve cach cu.
    if adaptive_crop:
        pcd = crop_to_trunk_cylinder_adaptive(pcd, cx, cz,
                                              radius=trunk_radius,
                                              max_jump=max_jump,
                                              seed_max_offset=seed_max_offset,
                                              max_radius_ratio=max_radius_ratio)
    else:
        pcd = crop_to_trunk_cylinder(pcd, cx, cz, radius=trunk_radius)

    # Step 4: Denoising lần 1 (rộng)
    pcd = denoise(pcd, nb_neighbors=20, std_ratio=2.5)

    # Step 5: Loại thực vật xanh
    if remove_green:
        pcd = remove_green_vegetation(pcd, use_hue_filter=hue_green_filter)

    # Step 5b: Per-slice declutter -- loai canh cay/vat the phu dinh vao
    # giua than (xem CLAUDE.md muc 12). Chay TRUOC DBSCAN 3D vi canh dinh
    # lien tuc theo chieu cao se bi keep_largest_cluster() (chay xuyen suot
    # ca khoi 3D) nhap chung vao cum than -- xu ly TUNG LAT truoc se tach no
    # ra som, giup DBSCAN 3D phia sau lam viec tren du lieu da sach hon.
    if declutter_slices and len(pcd.points) > 100:
        pcd = remove_off_axis_clusters_per_slice(pcd, n_slices=40, eps=0.03, min_points=10)

    # Step 6: DBSCAN — giữ cluster lớn nhất (thân cây)
    if use_dbscan and len(pcd.points) > 100:
        pcd = keep_largest_cluster(pcd, eps=0.04, min_points=30)

    # Step 7: Denoising lần 2 (tinh)
    pcd = denoise(pcd, nb_neighbors=30, std_ratio=2.0)

    n_final = len(pcd.points)
    print(f"\n  Result: {n_orig} -> {n_final} pts "
          f"({100*n_final/n_orig:.1f}% retained)")

    if n_final < 100:
        print("  WARNING: very few points remain, check trunk_radius parameter")

    # Lưu
    out_path = os.path.join(dataset_dir, "output", "trunk_pointcloud.ply")
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"  Saved: {out_path}")
    return True


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch(raw_data_dir: str, trunk_radius: float = 0.30) -> None:
    """Xử lý tất cả mẫu trong thư mục Raw_data/."""
    scan_dirs = sorted([
        os.path.join(raw_data_dir, d)
        for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
        and os.path.exists(os.path.join(raw_data_dir, d, "output", "pointcloud.ply"))
    ])

    print(f"Found {len(scan_dirs)} scans with pointcloud.ply")

    success, skip = 0, 0
    for scan_dir in scan_dirs:
        ok = isolate_trunk(scan_dir, trunk_radius=trunk_radius)
        if ok:
            success += 1
        else:
            skip += 1

    print(f"\n{'='*60}")
    print(f"Batch done: {success} processed, {skip} skipped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Isolate tree trunk from raw point cloud"
    )
    parser.add_argument(
        "dataset",
        help="Path to single scan folder (e.g., Raw_data/26Q1) "
             "or to Raw_data/ folder with --batch"
    )
    parser.add_argument(
        "--trunk-radius", type=float, default=0.30,
        help="Horizontal (X,Z) cylinder radius in meters to keep around trunk axis (default: 0.30)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Process all scans in the given folder"
    )
    parser.add_argument(
        "--no-green-filter", action="store_true",
        help="Disable green vegetation color filter"
    )
    parser.add_argument(
        "--no-dbscan", action="store_true",
        help="Disable DBSCAN largest-cluster selection"
    )
    parser.add_argument(
        "--no-slice-declutter", action="store_true",
        help="Disable per-slice off-axis cluster removal (branch/neighbor-object filter)"
    )
    parser.add_argument(
        "--no-hue-green-filter", action="store_true",
        help="Disable Hue+Saturation vegetation check in green filter (RGB-diff rule only)"
    )
    parser.add_argument(
        "--no-adaptive-crop", action="store_true",
        help="Disable adaptive per-slice cylinder crop (use the old single fixed axis)"
    )
    parser.add_argument(
        "--max-jump", type=float, default=0.10,
        help="Max horizontal shift (m) of the trunk center between adjacent slices "
             "in adaptive crop (default: 0.10)"
    )
    parser.add_argument(
        "--seed-max-offset", type=float, default=0.40,
        help="Max distance (m) from the rough trunk axis for an adaptive-crop seed "
             "candidate -- guards against seeding on a neighbouring tree (default: 0.40)"
    )
    parser.add_argument(
        "--no-view-anchor", action="store_true",
        help="Do not locate the target tree from the camera viewing rays; fall back "
             "to the old median-of-upper-points trunk axis (see CLAUDE.md 12/14)"
    )
    parser.add_argument(
        "--max-radius-ratio", type=float, default=3.0,
        help="Drop a height slice whose radius exceeds this multiple of the scan's "
             "own baseline trunk radius -- removes leftover ground/undergrowth mass "
             "that would otherwise outweigh the trunk. 0 disables (default: 3.0)"
    )
    args = parser.parse_args()

    if args.batch:
        run_batch(args.dataset, trunk_radius=args.trunk_radius)
    else:
        ok = isolate_trunk(
            args.dataset,
            trunk_radius=args.trunk_radius,
            remove_green=not args.no_green_filter,
            use_dbscan=not args.no_dbscan,
            declutter_slices=not args.no_slice_declutter,
            hue_green_filter=not args.no_hue_green_filter,
            adaptive_crop=not args.no_adaptive_crop,
            max_jump=args.max_jump,
            seed_max_offset=args.seed_max_offset,
            max_radius_ratio=args.max_radius_ratio,
            view_anchor=not args.no_view_anchor,
        )
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
