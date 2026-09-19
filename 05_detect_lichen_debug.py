#!/usr/bin/env python3
"""
Debug script: Test tunable parameters để tìm best settings
Thay đổi các giá trị tại đầu file và chạy lại để so sánh

Usage:
    python 05_detect_lichen_debug.py <path_to_trunk_pointcloud.ply> <path_to_mesh.ply>
"""

import os, sys, json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TUNABLE PARAMETERS — Thay đổi những giá trị này để test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FIX 1: bright_enough threshold — đây là nguyên nhân CHÍNH gây under-detection
# Mục đích: Cho phép lichen "bình thường" được chấp nhận
# 0.75 = quá chặt (chỉ lấy lichen sáng nhất)
# 0.60 = nới lỏng hợp lý (cho phép ~40% neighbors là lichen)
# 0.50 = quá nới lỏng (nhiều false positive)
BRIGHT_ENOUGH_RATIO = 0.60

# FIX 2: Burnout filter — loại bỏ vùng bị phản xạ ánh nắng
# BURNOUT_V: ngưỡng độ sáng (tối thiểu)
#   0.95 = quá chặt (loại hết lichen sáng)
#   0.90 = hợp lý
#   0.85 = quá nới lỏng
BURNOUT_V = 0.90

# BURNOUT_S: ngưỡng độ bão hoà (tối đa)
#   0.08 = quá chặt
#   0.10 = hợp lý
#   0.15 = quá nới lỏng
BURNOUT_S = 0.10

# FIX 3: Local contrast — có sử dụng hay không
# True = dùng local contrast (so sánh với 40 hàng xóm)
# False = test không dùng (chỉ dùng absolute thresholds)
USE_LOCAL_CONTRAST = True

# FIX 4: Req saturation drop — lichen phải nhạt hơn lân cận bao nhiêu
# Giá trị cũ: 0.08 (đối với cây trung bình)
# Có thể thử: 0.05, 0.03 (nới lỏng hơn)
REQ_S_DROP_FACTOR = 0.8  # nhân 0.8 với req_s_drop cũ (giảm yêu cầu)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """rgb: (N,3) float [0,1] -> hsv: (N,3) H[0,360] S[0,1] V[0,1]"""
    r, g, b = rgb[:,0], rgb[:,1], rgb[:,2]
    cmax = np.maximum(np.maximum(r,g),b)
    cmin = np.minimum(np.minimum(r,g),b)
    delta_c = cmax - cmin + 1e-9
    
    h = np.zeros(len(rgb), dtype=np.float32)
    mr=cmax==r; mg=cmax==g; mb=cmax==b
    h[mr] = 60*(((g[mr]-b[mr])/delta_c[mr])%6)
    h[mg] = 60*(((b[mg]-r[mg])/delta_c[mg])+2)
    h[mb] = 60*(((r[mb]-g[mb])/delta_c[mb])+4)
    
    s = np.where(cmax>0, delta_c/(cmax+1e-9), 0).astype(np.float32)
    v = cmax.astype(np.float32)
    
    return np.stack([h, s, v], axis=1)


def classify_lichen_local_contrast_debug(pts, cols, k_local=40):
    """
    Debug version với tunable parameters
    """
    print("\n" + "="*70)
    print("STEP 1: HSV ANALYSIS & LOCAL CONTRAST")
    print("="*70)
    
    r, g, b = cols[:,0], cols[:,1], cols[:,2]
    cmax = np.maximum(np.maximum(r,g),b)
    cmin = np.minimum(np.minimum(r,g),b)
    delta_c = cmax - cmin + 1e-9
    
    h = np.zeros(len(cols), dtype=np.float32)
    mr=cmax==r; mg=cmax==g; mb=cmax==b
    h[mr] = 60*(((g[mr]-b[mr])/delta_c[mr])%6)
    h[mg] = 60*(((b[mg]-r[mg])/delta_c[mg])+2)
    h[mb] = 60*(((r[mb]-g[mb])/delta_c[mb])+4)
    
    s = np.where(cmax>0, delta_c/(cmax+1e-9), 0).astype(np.float32)
    v = cmax.astype(np.float32)

    # Global stats
    global_med_V = float(np.percentile(v, 20))
    global_std_V = float(np.std(v))
    print(f"\n[Bark estimation]")
    print(f"  percentile(V, 20) = {global_med_V:.3f}  (estimated bark brightness)")
    print(f"  std(V) = {global_std_V:.3f}")
    
    # Determine bark type
    if global_med_V < 0.32:
        bark_type = "Dark (V < 0.32)"
        abs_v, abs_s = float(max(0.55, global_med_V + 1.8 * global_std_V)), 0.28
        v_sigma_k, v_sigma_base, req_s_drop_orig = 0.8, 0.03, 0.03
    elif global_med_V < 0.45:
        bark_type = "Medium (0.32 <= V < 0.45)"
        abs_v, abs_s = float(max(0.63, global_med_V + 2.1 * global_std_V)), 0.24
        v_sigma_k, v_sigma_base, req_s_drop_orig = 1.0, 0.04, 0.08
    else:
        bark_type = "Light (V >= 0.45)"
        abs_v, abs_s = float(max(0.70, global_med_V + 2.5 * global_std_V)), 0.20
        v_sigma_k, v_sigma_base, req_s_drop_orig = 1.2, 0.05, 0.08

    print(f"  Bark type: {bark_type}")
    print(f"  → abs_v threshold: {abs_v:.3f}  (tuyệt đối)")
    print(f"  → abs_s threshold: {abs_s:.3f}  (tuyệt đối)")
    print(f"  → req_s_drop (original): {req_s_drop_orig:.3f}")
    
    # Apply FIX 4: reduce req_s_drop
    req_s_drop = req_s_drop_orig * REQ_S_DROP_FACTOR
    print(f"  → req_s_drop (adjusted by {REQ_S_DROP_FACTOR}): {req_s_drop:.3f}")

    # KDTree for local neighbors
    print(f"\n[Building KDTree for k={k_local} neighbors]")
    tree = cKDTree(pts)
    _, idx_nbr = tree.query(pts, k=k_local+1, workers=-1)
    idx_nbr = idx_nbr[:, 1:]  # remove self

    local_mean_V = v[idx_nbr].mean(axis=1)
    local_std_V  = v[idx_nbr].std(axis=1)
    local_mean_S = s[idx_nbr].mean(axis=1)

    v_contrast = v - local_mean_V
    s_drop = local_mean_S - s

    is_veg = (h >= 65) & (h <= 150) & (s > 0.25)
    n_veg = int(is_veg.sum())
    print(f"  Vegetation filtered: {n_veg:,} pts ({100*n_veg/len(cols):.1f}%)")

    # FIX 2: Burnout filter
    print(f"\n[Burnout filter: V >= {BURNOUT_V}, S <= {BURNOUT_S}]")
    is_burnout = (v >= BURNOUT_V) & (s <= BURNOUT_S)
    n_burnout = int(is_burnout.sum())
    print(f"  Filtered (burnout): {n_burnout:,} pts ({100*n_burnout/len(cols):.1f}%)")

    # Local contrast detection
    if USE_LOCAL_CONTRAST:
        print(f"\n[Local contrast detection (ENABLED)]")
        v_thr = np.maximum(v_sigma_base, v_sigma_k * local_std_V)
        local_white = (
            (v_contrast > v_thr) &
            (s < local_mean_S - req_s_drop) &
            (s < 0.42) &
            ~is_veg
        )
        print(f"  Points brighter than neighbors: {(v_contrast > v_thr).sum():,}")
        print(f"  Points less saturated than neighbors: {(s < local_mean_S - req_s_drop).sum():,}")
        print(f"  → Local white candidates: {int(local_white.sum()):,} ({100*local_white.sum()/len(cols):.1f}%)")
    else:
        print(f"\n[Local contrast detection (DISABLED)]")
        local_white = np.zeros(len(cols), dtype=bool)

    # Absolute thresholds
    abs_white = (v >= abs_v) & (s < abs_s) & ~is_veg
    print(f"  Absolute white candidates: {int(abs_white.sum()):,} ({100*abs_white.sum()/len(cols):.1f}%)")

    # Breakdown
    n_local_only = int((local_white & ~abs_white).sum())
    n_abs_only   = int((abs_white & ~local_white).sum())
    n_both       = int((local_white & abs_white).sum())
    print(f"\n[Overlap analysis]")
    print(f"  Local only: {n_local_only:,}  ({100*n_local_only/len(cols):.1f}%)")
    print(f"  Absolute only: {n_abs_only:,}  ({100*n_abs_only/len(cols):.1f}%)")
    print(f"  Both methods agree: {n_both:,}  ({100*n_both/len(cols):.1f}%)")

    # Final labels
    white = local_white & ~is_burnout
    labels = np.zeros(len(cols), dtype=np.uint8)
    labels[white] = 2
    
    print(f"\n[Final result after burnout filter]")
    print(f"  White lichen points: {(labels==2).sum():,} ({100*(labels==2).sum()/len(cols):.1f}%)")

    return labels


def detect_on_pcd_transfer_to_mesh_debug(pcd_path, verts, k=15):
    """
    Debug version với FIX 1: bright_enough threshold
    """
    print("\n" + "="*70)
    print("STEP 2: TRANSFER TO MESH VERTICES")
    print("="*70)
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32)

    MAX_PTS = 500_000
    if len(pts) > MAX_PTS:
        idx_sub = np.random.choice(len(pts), MAX_PTS, replace=False)
        idx_sub.sort()
        pts = pts[idx_sub]
        cols = cols[idx_sub]
        print(f"\nPCD subsampled: {len(idx_sub):,} / {len(np.asarray(o3d.io.read_point_cloud(pcd_path).points)):,} pts")

    pcd_v = np.max(cols, axis=1)

    print(f"\nRunning local contrast analysis on PCD ({len(pts):,} points)...")
    pcd_labels = classify_lichen_local_contrast_debug(pts, cols)

    # Transfer to mesh
    print("\n" + "="*70)
    print("STEP 3: KNN TRANSFER TO MESH + bright_enough FILTER")
    print("="*70)
    
    tree = cKDTree(pts)
    _, idx = tree.query(verts.astype(np.float32), k=k, workers=-1)

    neighbor_labels = pcd_labels[idx]
    neighbor_V = pcd_v[idx]

    max_V_per_vert = neighbor_V.max(axis=1, keepdims=True)
    
    # FIX 1: bright_enough với tunable param
    print(f"\n[FIX 1: bright_enough = {BRIGHT_ENOUGH_RATIO}]")
    print(f"  (dùng để lọc neighbors thấp sáng)")
    
    bright_enough = neighbor_V >= (max_V_per_vert * BRIGHT_ENOUGH_RATIO)
    bright_white = (neighbor_labels == 2) & bright_enough

    # Diagnostics
    raw_white = (neighbor_labels == 2)
    n_raw_ge1 = int((raw_white.sum(axis=1) >= 1).sum())
    n_filtered_ge1 = int((bright_white.sum(axis=1) >= 1).sum())
    n_lost = n_raw_ge1 - n_filtered_ge1
    
    print(f"\n[Impact of bright_enough filter]")
    print(f"  Vertices with >=1 white neighbor (raw): {n_raw_ge1:,}")
    print(f"  Vertices with >=1 white neighbor (after filter): {n_filtered_ge1:,}")
    print(f"  → Lost: {n_lost:,} vertices ({100*n_lost/max(n_raw_ge1,1):.1f}%)")
    
    if n_lost > 0:
        print(f"\n  ⚠️  Bạn mất {100*n_lost/max(n_raw_ge1,1):.1f}% địa y do bright_enough filter!")
        if BRIGHT_ENOUGH_RATIO > 0.50:
            print(f"  💡 Thử giảm BRIGHT_ENOUGH_RATIO xuống {BRIGHT_ENOUGH_RATIO - 0.05}")

    # Vote distribution
    vote_counts = bright_white.sum(axis=1)
    print(f"\n[Vote distribution (bright_white)]")
    for v in range(0, min(6, int(vote_counts.max())+1)):
        n_v = int((vote_counts == v).sum())
        if n_v > 0:
            pct = 100*n_v/len(verts)
            bar = "█" * int(pct / 2)
            print(f"  {v} votes: {n_v:6,} vertex ({pct:6.2f}%) {bar}")

    n_ge1 = int((vote_counts >= 1).sum())
    n_ge2 = int((vote_counts >= 2).sum())
    print(f"\n  Vertices with >=1 vote: {n_ge1:,}  ({100*n_ge1/len(verts):.2f}%)")
    print(f"  Vertices with >=2 votes: {n_ge2:,}  ({100*n_ge2/len(verts):.2f}%)")

    vert_labels = np.zeros(len(verts), dtype=np.uint8)
    vert_labels[bright_white.sum(axis=1) >= 1] = 2

    return vert_labels


def run_debug(pcd_path, mesh_path, scan_name):
    """Main debug runner"""
    
    print(f"\n{'#'*70}")
    print(f"# DEBUG LICHEN DETECTION: {scan_name}")
    print(f"{'#'*70}")
    
    print(f"\n[Configuration]")
    print(f"  BRIGHT_ENOUGH_RATIO = {BRIGHT_ENOUGH_RATIO}")
    print(f"  BURNOUT_V = {BURNOUT_V}")
    print(f"  BURNOUT_S = {BURNOUT_S}")
    print(f"  USE_LOCAL_CONTRAST = {USE_LOCAL_CONTRAST}")
    print(f"  REQ_S_DROP_FACTOR = {REQ_S_DROP_FACTOR}")
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    
    print(f"\n[Mesh info]")
    print(f"  Vertices: {len(verts):,}")
    print(f"  Faces: {len(tris):,}")
    
    # Run detection
    vert_labels = detect_on_pcd_transfer_to_mesh_debug(pcd_path, verts)
    
    # Per-face labels
    face_label_votes = np.stack([vert_labels[tris[:,0]],
                                  vert_labels[tris[:,1]],
                                  vert_labels[tris[:,2]]], axis=1)
    lichen_labels = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=3).argmax(), 1,
        face_label_votes).astype(np.uint8)
    
    # Calculate area
    def triangle_areas(v, t):
        v0 = v[t[:, 0]]
        v1 = v[t[:, 1]]
        v2 = v[t[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    areas = triangle_areas(verts, tris)
    lichen_area_cm2 = float((areas[lichen_labels > 0]).sum()) * 10000
    total_area_cm2 = float(areas.sum()) * 10000
    lichen_pct = 100 * lichen_area_cm2 / total_area_cm2 if total_area_cm2 > 0 else 0
    
    print(f"\n" + "="*70)
    print(f"FINAL RESULT")
    print(f"="*70)
    print(f"  Total bark area: {total_area_cm2:,.0f} cm²")
    print(f"  Lichen area: {lichen_area_cm2:,.0f} cm²")
    print(f"  Lichen %: {lichen_pct:.1f}%")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <pcd_path> <mesh_path>")
        print(f"\nExample:")
        print(f"  python {sys.argv[0]} Raw_data/cay_0007_1805/output/trunk_pointcloud.ply Raw_data/cay_0007_1805/output/trunk_mesh_final.ply")
        sys.exit(1)
    
    pcd_path = sys.argv[1]
    mesh_path = sys.argv[2]
    
    if not os.path.exists(pcd_path):
        print(f"Error: PCD file not found: {pcd_path}")
        sys.exit(1)
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found: {mesh_path}")
        sys.exit(1)
    
    scan_name = os.path.basename(os.path.dirname(pcd_path))
    run_debug(pcd_path, mesh_path, scan_name)