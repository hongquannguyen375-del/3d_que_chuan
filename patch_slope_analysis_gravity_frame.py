#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_slope_analysis_gravity_frame.py
=====================================
Va 06_slope_analysis.py: sua estimate_gravity_world(), von tra ve mot vector
gravity NAM NGANG tren moi scan da do (lech 88.7-93.5 do khoi phuong thang
dung), trong khi IMU tho chi lech 1.1-7.6 do. CLAUDE.md muc 6 goi ham nay la
"nguon do doc dang tin nhat", va toan bo ~90 file slope_info.json trong kho
deu lay terrain_slope_deg + nhan upslope/downslope tu no.

BA LOI DOC LAP, deu do bang du lieu that truoc khi va:

1. SAI DON VI -> bo loc chuyen dong bi vo hieu hoan toan.
   imu.csv cua bo du lieu nay ghi theo don vi g (|a| trung vi 0.996-1.001),
   khong phai m/s^2. Bo loc cu la  |(|a| - 9.80665)| < 0.5  nen KHONG khop
   noi mot mau nao tren ca 49 scan: ham luon roi vao nhanh canh bao va dung
   TOAN BO mau, ke ca luc dang lac manh.

2. TRUNG BINH QUATERNION THAY VI TRUNG BINH KET QUA -> vo nghia khi quay vong.
   Docstring huong dan "Cach 1 (preferred): voi moi IMU sample, rotate bang
   camera rotation gan nhat, roi trung binh" -- nhung code KHONG lam the. No
   lam Cach 2: trung binh IMU truoc, roi xoay bang MOT quaternion trung binh
   cong theo tung thanh phan. Nguoi quay ORBIT quanh than cay (CLAUDE.md muc
   4): do duoc goc xoay camera so voi pose dau tien co trung vi 80-97 do va
   max 179.6-179.9 do. Trung binh cong quaternion tren mot cung rong nhu vay
   khong con la phep trung binh xoay; chuan hoa cua no la nhieu. Gravity thi
   HANG SO trong world frame, nen cach dung la xoay TUNG mau roi moi trung
   binh ket qua.

3. THIEU PHEP XOAY HANG SO GIUA KHUNG IMU VA KHUNG CAMERA.
   Day moi la nguyen nhan chinh lam ket qua nam ngang. Code cu gia dinh
   "camera frame ~ device frame" (chinh comment trong file noi vay). That ra
   accelerometer o khung thiet bi cua CoreMotion, con quaternion o khung
   camera cua ARKit; hai khung lech nhau mot phep xoay HANG SO.

CACH DO RA HANG SO DO (khong doan):
  - Kiem tra nhat quan: gravity la hang so trong world, nen quy uoc dung phai
    cho cac ket qua tung mau TRUNG NHAU. R_i @ a_i tan mac 65-81 do
    (consistency 0.11-0.29); R_i.T @ a_i bo chat 10-16 do (0.91-0.96).
  - Kiem tra huong nhin (moc doc lap, khong dinh den gravity): camera phai
    CHI VAO than cay, ma vi tri than cay da biet tu trunk_pointcloud.ply. Ket
    qua: R @ +Z lech 3-6 do so voi huong ve than cay -> R DUNG la
    camera->world (khop pose_to_extrinsic() trong 02_pointcloud_and_mesh.py).
  - Giai bang Kabsch/Wahba cho hang so C trong  g = R_i @ C @ a_i : residual
    2.9-4.3 do, va C giong het nhau tren moi scan. Doi thu  g = R_i.T @ C @ a
    cho residual 61-78 do -> loai.
  - Giai lai CA C LAN g tu khoi tao NGAU NHIEN, khong dung mot moc nao: tren
    49 scan, 47 hoi tu ve g = world -Y trong 0.0-0.7 do (trung vi 0.1) voi
    dung C do (lech 0.1-1.3 do). Hai scan con lai (cay_0009, cay_0033) roi
    vao cuc tri sai cua bo giai (residual cua chung cao hon: 6.8-7.2 do so
    voi 2.9-4.6 do); da kiem tra rieng: gravity khung thiet bi cua chung van
    la (0,-1,0) binh thuong (2.9 va 6.5 do khoi -Y) va than cay van thon dan
    len tren, tuc +Y van la huong len. Voi C co dinh, CA 49/49 scan cho
    g = world -Y trong 0.0-0.7 do, consistency 0.996.

LOI THU TU (cung file, lo ra ngay khi gravity duoc sua): terrain_slope_deg
tinh nguoc. Cong thuc cu  terrain_slope_deg = 90 - goc(gravity, truc than)
nen mot than cay THANG DUNG tren dat BANG bao ra 90 do. Voi gravity da sua,
than cay that nghieng 1.3-6.0 do, tuc cong thuc cu se in 84-89 do cho MOI
cay. Sua thanh chinh goc nghieng do.

LUU Y VE MO HINH (KHONG phai loi code, can nguoi dung quyet): sau khi sua,
gravity = world -Y, nen compute_slope_direction() rut gon thanh "huong than
cay nghieng ve", chu khong phai huong doc cua dia hinh. Hai dai luong chi
bang nhau neu cay moc vuong goc voi mat dat, ma cay thuong moc thang dung bat
ke doc. Trong du lieu tho KHONG co moc do doc dia hinh nao de doi chieu:
truong slope_degrees trong location.csv la DO NGHIENG CUA DIEN THOAI (trung
vi 78-84 do, khop voi 86 do do duoc tu IMU), khong phai do doc dat. Da do do
on dinh cua huong nghieng bang bootstrap: rat on dinh (p90 chi 0.5-1.9 do,
<=0.7% diem doi phia), nen day KHONG phai van de nhieu lay mau.

CACH DUNG
---------
    python patch_slope_analysis_gravity_frame.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "06_slope_analysis.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- Site 1: load_imu keeps timestamps -------------------------------------
SNIPPETS.append((
    "load_imu_timestamps",
    '''def load_imu(dataset_dir: str) -> np.ndarray:
    """Load accelerometer data. Returns (N, 3) array [a_x, a_y, a_z] m/s²."""
    path = os.path.join(dataset_dir, "imu.csv")
    data = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            data.append([float(row["a_x"]), float(row["a_y"]), float(row["a_z"])])
    return np.array(data)''',
    '''def load_imu(dataset_dir: str) -> np.ndarray:
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
    return np.array(data)''',
))

# --- Site 2: load_odometry keeps timestamps --------------------------------
SNIPPETS.append((
    "load_odometry_timestamps",
    '''            poses.append({
                "frame": row["frame"].strip(),''',
    '''            poses.append({
                "timestamp": float(row["timestamp"]),
                "frame": row["frame"].strip(),''',
))

# --- Site 3: the device->camera constant + rewritten gravity estimate ------
SNIPPETS.append((
    "estimate_gravity_world_per_sample",
    '''def estimate_gravity_world(
    imu_data: np.ndarray,
    poses: list[dict],
    low_motion_threshold: float = 0.5,
) -> np.ndarray:
    """
    Ước tính gravity vector trong world frame.

    Cách 1 (preferred): Với mỗi IMU sample, rotate từ device frame sang world frame
    bằng camera rotation gần nhất. Trung bình tất cả → gravity world.

    Cách 2 (fallback): Dùng trực tiếp trung bình IMU trong device frame
    rồi rotate bằng rotation trung bình.

    Lọc các frame có chuyển động mạnh (|a| xa 9.81) trước khi trung bình.
    """
    g_nominal = 9.80665

    # Lọc IMU: chỉ lấy sample gần tĩnh (|a| ≈ g)
    a_mag = np.linalg.norm(imu_data, axis=1)
    static_mask = np.abs(a_mag - g_nominal) < low_motion_threshold
    static_imu = imu_data[static_mask]

    if len(static_imu) < 10:
        print(f"  Warning: only {len(static_imu)} static IMU samples, using all")
        static_imu = imu_data

    # Trung bình gravity trong device frame
    g_device = static_imu.mean(axis=0)
    g_device_norm = g_device / np.linalg.norm(g_device)
    print(f"  Gravity in device frame: {g_device_norm.round(4)}")

    # Rotate sang world frame dùng rotation trung bình của camera
    # (Stray Scanner: camera frame ≈ device frame)
    rotations = [quaternion_to_rotation_matrix(p["quaternion"]) for p in poses]

    # Trung bình rotation: dùng trung bình quaternion đơn giản
    quats = np.array([p["quaternion"] for p in poses])
    # Chuẩn hóa dấu quaternion (flip nếu cần để tránh averaging cancel)
    ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], ref) < 0:
            quats[i] = -quats[i]
    q_mean = quats.mean(axis=0)
    q_mean /= np.linalg.norm(q_mean)
    R_mean = quaternion_to_rotation_matrix(q_mean)

    # Gravity trong world frame
    g_world = R_mean @ g_device_norm
    g_world_norm = g_world / np.linalg.norm(g_world)
    print(f"  Gravity in world frame: {g_world_norm.round(4)}")

    return g_world_norm''',
    '''# Phep xoay HANG SO tu khung thiet bi (CoreMotion, imu.csv) sang khung camera
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

    return g_world_norm''',
))

# --- Site 4: terrain_slope_deg was inverted --------------------------------
SNIPPETS.append((
    "terrain_slope_deg_inverted",
    '''    # Góc nghiêng địa hình (angle between gravity and trunk axis)
    cos_angle = abs(np.dot(g_world, trunk_axis))
    slope_angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    terrain_slope_deg = 90.0 - slope_angle_deg  # góc nghiêng so với nằm ngang

    print(f"  Slope down direction (world): {slope_down_norm.round(4)}")
    print(f"  Terrain slope angle: {terrain_slope_deg:.1f}°")''',
    '''    # Do nghieng cua THAN CAY so voi phuong thang dung.
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
          f"(reported as terrain_slope_deg)")''',
))

SNIPPETS.append((
    "slope_dict_trunk_tilt",
    '''        "terrain_slope_deg": float(terrain_slope_deg),
        "slope_horizontal_magnitude": float(slope_magnitude),''',
    '''        "terrain_slope_deg": float(terrain_slope_deg),
        "trunk_tilt_deg": float(trunk_tilt_deg),
        "slope_horizontal_magnitude": float(slope_magnitude),''',
))

SNIPPETS.append((
    "json_trunk_tilt",
    '''        "terrain_slope_deg": slope_info["terrain_slope_deg"],''',
    '''        "terrain_slope_deg": slope_info["terrain_slope_deg"],
        "trunk_tilt_deg": slope_info["trunk_tilt_deg"],''',
))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if all(new in content for _, _, new in SNIPPETS):
        print(f"{TARGET_NAME} (gravity frame fix) da duoc va truoc do roi.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Luu y: patch nay yeu cau patch_slope_analysis_up_axis.py da duoc "
              "ap dung truoc.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_gravityframe")
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
    print(f"Da va thanh cong {len(applied)} doan (gravity frame fix): {applied}")
    print("Chay lai buoc slope analysis, vd:")
    print(f'  python {TARGET_NAME} <scan_dir>')
    print("Hoac ca kho:  python 06_slope_analysis.py --batch <raw_data_dir>")


if __name__ == "__main__":
    main()
