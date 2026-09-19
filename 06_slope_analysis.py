#!/usr/bin/env python3
"""
Step 6: Slope Analysis — Xác định mặt Upslope / Downslope của thân cây.

Nguyên lý:
  - IMU accelerometer (a_x, a_y, a_z) khi thiết bị gần tĩnh ≈ gravity vector
    trong device frame (~9.8 m/s²).
  - Odometry cho camera pose (position + quaternion) ở mỗi frame.
  - Gravity trong world frame = R_cam2world @ g_device (trung bình nhiều frame)
  - World-frame gravity gồm:
      • g_vertical  = thành phần thẳng đứng (dọc theo thân cây)
      • g_slope     = thành phần nằm ngang (hướng dốc xuống)
  - Phía DOWNSLOPE = hướng g_slope (đất dốc xuống về phía đó)
  - Phía UPSLOPE   = ngược lại (-g_slope)

Output:
  - slope_info.json  : gravity vector, slope direction, góc nghiêng, trunk axis
  - trunk_labeled.ply: thêm scalar field "side" (0=upslope, 1=downslope) vào vertex colors
                        upslope  → màu đỏ
                        downslope → màu xanh dương

Usage:
    python 06_slope_analysis.py E:/3D_Que/Raw_data/26Q1
    python 06_slope_analysis.py E:/3D_Que/Raw_data  --batch
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import open3d as o3d

import geometry_utils


# ---------------------------------------------------------------------------
# Load sensor data
# ---------------------------------------------------------------------------

def load_imu(dataset_dir: str) -> np.ndarray:
    """
    Load accelerometer data. Returns (N, 4) array [timestamp, a_x, a_y, a_z].

    Timestamp is kept because estimate_gravity_world() now rotates EACH sample
    by the camera pose in effect at that instant; without it there is no way
    to pair an IMU reading with a pose.

    Units are whatever the device wrote and are NOT assumed here: this
    dataset's files are in g (|a| ~ 1.0), not m/s^2. estimate_gravity_world()
    detects the scale.
    """
    path = os.path.join(dataset_dir, "imu.csv")
    data = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            try:
                data.append([float(row["timestamp"]), float(row["a_x"]),
                             float(row["a_y"]), float(row["a_z"])])
            except (KeyError, ValueError):
                continue
    return np.array(data)


def load_odometry(dataset_dir: str) -> list[dict]:
    """Load camera poses from odometry.csv."""
    path = os.path.join(dataset_dir, "odometry.csv")
    poses = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            poses.append({
                "timestamp": float(row["timestamp"]),
                "frame": row["frame"].strip(),
                "position": np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                "quaternion": np.array([
                    float(row["qx"]), float(row["qy"]),
                    float(row["qz"]), float(row["qw"])
                ]),
            })
    return poses


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),    2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ])


# ---------------------------------------------------------------------------
# Gravity estimation in world frame
# ---------------------------------------------------------------------------

# Phep xoay HANG SO tu khung thiet bi (CoreMotion, imu.csv) sang khung camera
# (quaternion trong odometry.csv). Code cu gia dinh hai khung nay TRUNG NHAU
# ("Stray Scanner: camera frame ~ device frame") -- do chinh la ly do gravity
# world tinh ra nam ngang. Ma tran duoi day KHONG phai gia dinh: no duoc giai
# ra tu du lieu that bang Kabsch/Wahba tren 49 scan va moi scan doc lap cho ra
# cung mot ma tran (lech nhau 0.1-1.3 do). Y nghia:
#     device +x -> camera -y      device +y -> camera -x
#     device +z -> camera -z      (camera nhin nguoc huong man hinh)
# Xem docstring cua patch_slope_analysis_gravity_frame.py va CLAUDE.md muc 13.
DEVICE_TO_CAMERA = np.array([
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])


def estimate_gravity_world(
    imu_data: np.ndarray,
    poses: list[dict],
    low_motion_fraction: float = 0.05,
    min_consistency: float = 0.80,
    max_tilt_deg: float = 25.0,
) -> np.ndarray:
    """
    Gravity vector trong world frame, tinh theo TUNG MAU IMU.

    Voi moi mau IMU: tim pose gan nhat theo thoi gian, doi sang khung camera
    bang DEVICE_TO_CAMERA, roi xoay sang world bang rotation cua pose do.
    Trung binh CAC KET QUA -- khong phai trung binh cac quaternion: nguoi quay
    orbit quanh than cay nen cung xoay rong (trung vi 80-97 do, max ~180 do),
    trung binh cong quaternion tren cung do la vo nghia.

    Hai chot an toan, nguong lay tu so do that tren 49 scan (consistency
    0.996, lech 0.0-0.7 do khoi phuong thang dung):
      - consistency = do dai vector trung binh cua cac ket qua don vi. Gravity
        la hang so trong world nen gia tri nay phai gan 1. Thap = sai quy uoc
        khung, hoac pose/IMU khong khop thoi gian.
      - do lech khoi phuong thang dung. ARKit dung worldAlignment .gravity nen
        world Y da thang truc trong luc san; lech lon = co gi do sai.
    Vi pham chot nao cung CANH BAO RO va tra ve world down, thay vi im lang
    ghi mot vector sai vao slope_info.json nhu truoc.
    """
    world_down = np.zeros(3)
    world_down[geometry_utils.UP_AXIS] = -1.0

    if imu_data is None or len(imu_data) == 0 or not poses:
        print("  WARNING: no IMU or pose data -- using world down")
        return world_down
    if np.ndim(imu_data) != 2 or np.shape(imu_data)[1] < 4:
        print("  WARNING: imu_data has no timestamp column -- using world down")
        return world_down

    t_imu = imu_data[:, 0]
    acc = imu_data[:, 1:4]
    a_mag = np.linalg.norm(acc, axis=1)
    usable = np.isfinite(a_mag) & (a_mag > 1e-9)
    if usable.sum() < 10:
        print("  WARNING: no usable IMU samples -- using world down")
        return world_down

    # Don vi: bo du lieu nay ghi theo g (|a| ~ 1.0), khong phai m/s^2. Nguong
    # cu la |(|a| - 9.80665)| < 0.5 nen khong bao gio khop, va bo loc chuyen
    # dong bi vo hieu hoan toan.
    med = float(np.median(a_mag[usable]))
    unit = "g" if med < 3.0 else "m/s^2"
    g_nominal = 1.0 if med < 3.0 else 9.80665
    print(f"  IMU units: {unit} (|a| median {med:.3f})")

    static = usable & (np.abs(a_mag - g_nominal) < low_motion_fraction * g_nominal)
    if static.sum() < 10:
        print(f"  Warning: only {int(static.sum())} static IMU samples, using all")
        static = usable
    print(f"  Static IMU samples: {int(static.sum())} / {len(a_mag)}")

    g_device = acc[static].mean(axis=0)
    g_device = g_device / np.linalg.norm(g_device)
    print(f"  Gravity in device frame: {g_device.round(4)}")

    # Ghep tung mau IMU voi pose gan nhat theo timestamp.
    t_pose = np.array([float(p["timestamp"]) for p in poses])
    order = np.argsort(t_pose)
    t_sorted = t_pose[order]
    ts = t_imu[static]
    j = np.clip(np.searchsorted(t_sorted, ts), 1, len(t_sorted) - 1)
    take_left = np.abs(ts - t_sorted[j - 1]) < np.abs(ts - t_sorted[j])
    j[take_left] -= 1
    sel = order[j]

    # Xoay TUNG mau roi moi trung binh.
    R_all = np.array([quaternion_to_rotation_matrix(p["quaternion"]) for p in poses])
    a_unit = acc[static] / a_mag[static][:, None]
    g_samples = np.einsum("nij,jk,nk->ni", R_all[sel], DEVICE_TO_CAMERA, a_unit)

    g_mean = g_samples.mean(axis=0)
    consistency = float(np.linalg.norm(g_mean))
    g_world_norm = g_mean / max(consistency, 1e-12)
    tilt = float(np.degrees(np.arccos(
        np.clip(float(np.dot(g_world_norm, world_down)), -1.0, 1.0))))
    print(f"  Gravity in world frame: {g_world_norm.round(4)} "
          f"(consistency {consistency:.3f}, {tilt:.1f} deg from world down)")

    if consistency < min_consistency or tilt > max_tilt_deg:
        print(f"  WARNING: gravity estimate REJECTED (consistency "
              f"{consistency:.3f} < {min_consistency:.2f}, or tilt {tilt:.1f} "
              f"deg > {max_tilt_deg:.0f} deg). Using world down instead -- "
              f"check the device/camera frame convention for this scan.")
        return world_down

    return g_world_norm


# ---------------------------------------------------------------------------
# Trunk axis estimation
# ---------------------------------------------------------------------------

def estimate_trunk_axis_pca(pcd: o3d.geometry.PointCloud) -> np.ndarray:
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
    return trunk_axis


# ---------------------------------------------------------------------------
# Slope direction
# ---------------------------------------------------------------------------

def compute_slope_direction(
    g_world: np.ndarray,
    trunk_axis: np.ndarray,
) -> dict:
    """
    Tính hướng dốc (slope direction) từ gravity và trục thân cây.

    slope_down = thành phần của gravity vuông góc với trục thân
               = hướng dốc XUỐNG (downslope direction)
    slope_up   = -slope_down

    Trả về dict với các thông tin slope.
    """
    # Bỏ thành phần dọc theo trục thân cây
    g_proj_trunk = np.dot(g_world, trunk_axis) * trunk_axis
    slope_down = g_world - g_proj_trunk  # thành phần nằm ngang

    slope_magnitude = np.linalg.norm(slope_down)
    if slope_magnitude < 1e-6:
        print("  WARNING: slope direction nearly zero (flat terrain?)")
        # Chọn hướng tùy ý
        slope_down = np.array([1.0, 0.0, 0.0])
        slope_magnitude = 1.0

    slope_down_norm = slope_down / slope_magnitude
    slope_up_norm = -slope_down_norm

    # Do nghieng cua THAN CAY so voi phuong thang dung.
    # Cong thuc cu la  90 - goc(gravity, truc than), nen mot than cay THANG
    # DUNG tren dat BANG PHANG bao ra 90 do -- nguoc hoan toan. Voi gravity da
    # sua (= world -Y), than cay that nghieng 1.3-6.0 do, tuc cong thuc cu se
    # in 84-89 do cho MOI cay.
    # LUU Y VE MO HINH (xem CLAUDE.md muc 13): dai luong nay do DO NGHIENG CUA
    # THAN CAY, khong phai do doc dia hinh. Hai cai chi bang nhau neu cay moc
    # vuong goc voi mat dat; cay thuong moc thang dung bat ke doc. Trong du
    # lieu tho khong co moc do doc dia hinh nao de doi chieu (truong
    # slope_degrees trong location.csv la do nghieng cua DIEN THOAI).
    cos_angle = abs(np.dot(g_world, trunk_axis))
    trunk_tilt_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    terrain_slope_deg = trunk_tilt_deg

    print(f"  Slope down direction (world): {slope_down_norm.round(4)}")
    print(f"  Trunk tilt from vertical: {trunk_tilt_deg:.1f} deg "
          f"(reported as terrain_slope_deg)")

    return {
        "slope_down": slope_down_norm,
        "slope_up": slope_up_norm,
        "terrain_slope_deg": float(terrain_slope_deg),
        "trunk_tilt_deg": float(trunk_tilt_deg),
        "slope_horizontal_magnitude": float(slope_magnitude),
    }


# ---------------------------------------------------------------------------
# Label trunk vertices: upslope / downslope
# ---------------------------------------------------------------------------

def label_trunk_by_slope(
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

    dot = radial @ slope_horiz  # (N,) — dương = phía downslope

    # Nhãn
    labels = np.where(dot > 0, 1, 0)  # 1=downslope, 0=upslope

    # Màu hóa
    colors = np.zeros((len(points), 3))
    colors[labels == 0] = [1.0, 0.0, 0.0]   # upslope = đỏ
    colors[labels == 1] = [0.0, 0.0, 1.0]   # downslope = xanh dương

    labeled_pcd = o3d.geometry.PointCloud()
    labeled_pcd.points = pcd.points
    labeled_pcd.colors = o3d.utility.Vector3dVector(colors)

    n_up = (labels == 0).sum()
    n_down = (labels == 1).sum()
    print(f"  Upslope: {n_up} pts ({100*n_up/len(points):.1f}%)")
    print(f"  Downslope: {n_down} pts ({100*n_down/len(points):.1f}%)")

    return labeled_pcd, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_slope(dataset_dir: str) -> dict | None:
    """
    Chạy toàn bộ slope analysis cho một mẫu.
    Trả về dict thông tin slope, lưu trunk_labeled.ply và slope_info.json.
    """
    trunk_pcd_path = os.path.join(dataset_dir, "output", "trunk_pointcloud.ply")
    if not os.path.exists(trunk_pcd_path):
        print(f"  SKIP: {trunk_pcd_path} not found. Run 05_trunk_isolation.py first.")
        return None

    print(f"\n{'='*60}")
    print(f"Slope analysis: {os.path.basename(dataset_dir)}")
    print(f"{'='*60}")

    # Load dữ liệu
    imu_data = load_imu(dataset_dir)
    poses = load_odometry(dataset_dir)
    pcd = o3d.io.read_point_cloud(trunk_pcd_path)

    print(f"  IMU samples: {len(imu_data)}")
    print(f"  Poses: {len(poses)}")
    print(f"  Trunk points: {len(pcd.points)}")

    # Gravity trong world frame
    g_world = estimate_gravity_world(imu_data, poses)

    # Trục thân cây
    trunk_axis = estimate_trunk_axis_pca(pcd)
    print(f"  Trunk axis (PCA): {trunk_axis.round(4)}")

    # Hướng dốc
    slope_info = compute_slope_direction(g_world, trunk_axis)

    # Tâm trục (để project radial)
    pts = np.asarray(pcd.points)
    cx, cz = geometry_utils.horizontal_center(pts)

    # Gán nhãn upslope/downslope
    labeled_pcd, labels = label_trunk_by_slope(
        pcd, cx, cz, slope_info["slope_down"]
    )

    # Lưu labeled point cloud
    out_labeled = os.path.join(dataset_dir, "output", "trunk_labeled.ply")
    o3d.io.write_point_cloud(out_labeled, labeled_pcd)
    print(f"  Saved labeled PCD: {out_labeled}")

    # Lưu slope info JSON
    result = {
        "sample": os.path.basename(dataset_dir),
        "gravity_world": g_world.tolist(),
        "trunk_axis": trunk_axis.tolist(),
        "slope_down_direction": slope_info["slope_down"].tolist(),
        "slope_up_direction": slope_info["slope_up"].tolist(),
        "terrain_slope_deg": slope_info["terrain_slope_deg"],
        "trunk_tilt_deg": slope_info["trunk_tilt_deg"],
        "trunk_center_xz": [cx, cz],   # doi ten tu trunk_center_xy: gia tri gio la (X,Z)
        "n_upslope_pts": int((labels == 0).sum()),
        "n_downslope_pts": int((labels == 1).sum()),
    }
    json_path = os.path.join(dataset_dir, "output", "slope_info.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved slope info: {json_path}")

    return result


def run_batch(raw_data_dir: str) -> None:
    scan_dirs = sorted([
        os.path.join(raw_data_dir, d)
        for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
        and os.path.exists(os.path.join(raw_data_dir, d, "output", "trunk_pointcloud.ply"))
    ])
    print(f"Found {len(scan_dirs)} scans with trunk_pointcloud.ply")

    results = []
    for scan_dir in scan_dirs:
        r = analyze_slope(scan_dir)
        if r:
            results.append(r)

    # Tổng hợp
    summary_path = os.path.join(raw_data_dir, "slope_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBatch done. Summary saved: {summary_path}")

    # In thống kê
    if results:
        angles = [r["terrain_slope_deg"] for r in results]
        print(f"Terrain slope: mean={np.mean(angles):.1f}°, "
              f"min={np.min(angles):.1f}°, max={np.max(angles):.1f}°")


def main():
    parser = argparse.ArgumentParser(
        description="Slope analysis: determine upslope/downslope sides of tree trunk"
    )
    parser.add_argument(
        "dataset",
        help="Path to single scan folder or Raw_data/ folder with --batch"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Process all scans with trunk_pointcloud.ply"
    )
    args = parser.parse_args()

    if args.batch:
        run_batch(args.dataset)
    else:
        result = analyze_slope(args.dataset)
        if result is None:
            sys.exit(1)
        print(f"\nSlope analysis complete:")
        print(f"  Terrain slope: {result['terrain_slope_deg']:.1f}°")
        print(f"  Slope down direction: {np.array(result['slope_down_direction']).round(3)}")


if __name__ == "__main__":
    main()
