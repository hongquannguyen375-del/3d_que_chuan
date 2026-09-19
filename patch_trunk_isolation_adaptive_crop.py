#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trunk_isolation_adaptive_crop.py
============================================
Va 05_trunk_isolation.py: thay buoc crop hinh tru bang MOT TAM CO DINH
(crop_to_trunk_cylinder) bang crop THICH UNG theo tung lat chieu cao
(crop_to_trunk_cylinder_adaptive) -- moi lat co tam rieng, bam theo quy dao
that cua than cay.

BOI CANH (do tren du lieu that cay_0004_1805, xem CLAUDE.md muc 12):
Nguoi dung bao than cay trong trunk_pointcloud.ply bi "cat doc" -- chi con
mot mieng mong det thay vi mot cot tron day du, trong khi khoi to gan goc
van nguyen ven. Gia thuyet ban dau cua nguoi dung ("ban kinh crop qua nho")
da bi DU LIEU THAT BAC BO:
  - Vung goc (lat 0-10): ban kinh that toi 41cm -- day la khoi re/goc phinh
    hoac dat/rac sot lai, KHONG phai than cay.
  - Vung than that (lat 11-22): ban kinh that chi 4.1-4.8cm (dung kich thuoc
    than que binh thuong), tam di chuyen muot theo mot quy dao rieng.
  - Nhung tam that do LECH 22.6-29.5cm khoi truc co dinh (cx=-0.319,
    cz=-0.586) -- gan bang dung ban kinh crop (30cm). Vi vay than cay nam
    sat MEP vung crop: nua gan truc duoc giu, nua xa bi cat -> "cat doc".
  - Nguyen nhan goc: estimate_trunk_axis() lay MOT trung vi (X,Z) duy nhat
    cho toan bo scan, va trung vi do bi khoi goc to o day keo lech khoi vi
    tri than that o giua/tren. Tang --trunk-radius chi keo them khoi goc +
    rac vao chung, KHONG sua duoc viec lech tam.

CO CHE FIX (da kiem chung bang diagnostic tren 5 scan truoc khi va):
  1. Crop tho rong (wide_radius, mac dinh 0.8m) quanh truc uoc tinh cu --
     chi de gioi han vung tim kiem, khong dung de cat chinh xac.
  2. Cat thanh n_slices lat theo chieu cao; moi lat chay DBSCAN 2D (X,Z)
     nho (eps=0.03) de lay cac blob ung vien (tam, so diem, ban kinh).
  3. Chon LAT SEED: trong 60% tren cua chieu cao (cung quy uoc voi
     estimate_trunk_axis upper_fraction=0.6, de khong bao gio seed vao rac
     vung goc), lay ban kinh blob lon nhat cua tung lat, tinh TRUNG VI cua
     chung; seed = lat co ban kinh gan trung vi nhat (tranh seed trung vao
     blob co kich thuoc di thuong).
  4. LAN TOA len va xuong tu seed: moi lat ke tiep chon blob co tam GAN
     NHAT voi tam lat lien truoc (tinh lien tuc, KHONG phai nhieu diem
     nhat -- vi o cay_0004 khoi goc to co nhieu diem HON than that, chon
     theo so diem se lai khoa nham vao no y nhu loi cu). Chi nhan neu
     khoang cach <= max_jump; neu khong, GIU NGUYEN tam lat truoc (coast),
     khong doan bua.
  5. Cat cuoi: moi diem chi giu neu nam trong trunk_radius quanh tam CUA
     CHINH LAT no, thay vi quanh mot tam chung.

KET QUA KIEM CHUNG (diagnostic, 5 scan, max_jump=10cm):
  - cay_0004 (ca loi): lat 5-39 ban kinh on dinh 4.1-6.5cm, thon dan len
    ngon (2.1cm), buoc nhay giua cac lat 0.1-1.3cm (rat muot). Vung goc
    (lat 0-4) dung nhu thiet ke: TU CHOI nhay toi khoi to (nearest 19-33cm
    > max_jump) thay vi bi hut vao nhu truc cu. Tong 50,496 pts so voi
    47,985 cu -- lay lai duoc phan than bi cat oan.
  - cay_0007/0010/0005/0009 (doi chung): theo doi muot (buoc nhay da so
    <2cm), tong diem thay doi nhe (-0.4% den +12%), khong scan nao bi vo.
    Con vai "dai lat mat dau" giua than (3-5 lat coast lien tiep, nearest
    11-20cm > max_jump) o vung gan canh cay/thieu du lieu -- an toan (khong
    nhay lung tung) nhung do chinh xac cuc bo kem hon; co the chinh
    --max-jump neu can.

Fallback: neu khong tim duoc lat seed, hoac qua it diem, tu dong quay ve
crop co dinh cu. Tat hoan toan bang --no-adaptive-crop.

CACH DUNG
---------
    python patch_trunk_isolation_adaptive_crop.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_trunk_isolation.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- Site 1: new functions, inserted after crop_to_trunk_cylinder() --------
SNIPPETS.append((
    "new_functions_adaptive_crop",
    '''    mask = (xz_dist <= radius) & (z_vals >= z_floor)
    cropped = pcd.select_by_index(np.where(mask)[0])
    print(f"  Cylinder crop r={radius}m: {len(points)} -> {len(cropped.points)} pts")
    return cropped


# ---------------------------------------------------------------------------
# 4. Statistical Outlier Removal
# ---------------------------------------------------------------------------''',
    '''    mask = (xz_dist <= radius) & (z_vals >= z_floor)
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

    upper_threshold = y_max - 0.6 * (y_max - y_min)
    cands = []
    for i in range(n_slices):
        if edges[i] < upper_threshold or not slice_blobs[i]:
            continue
        cands.append((i, max(slice_blobs[i], key=lambda b: b[1])))
    if not cands:
        print("  Adaptive crop: khong tim duoc lat seed -> fallback crop co dinh")
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

    keep = np.zeros(len(pts), dtype=bool)
    for i in range(n_slices):
        idx = slice_idx[i]
        if len(idx) == 0 or centers[i] is None:
            continue
        ccx, ccz = centers[i]
        d = np.sqrt((x_h[idx] - ccx) ** 2 + (z_h[idx] - ccz) ** 2)
        keep[idx[d <= radius]] = True

    out = pcd_wide.select_by_index(np.where(keep)[0])
    print(f"  Adaptive crop (r={radius}m, {n_slices} lat, max_jump={max_jump*100:.0f}cm, "
          f"seed=lat {seed_i} r_med={baseline_r*100:.1f}cm): "
          f"{len(pts)} -> {len(out.points)} pts [tracked={n_tracked}, coast={n_coast}]")
    return out


# ---------------------------------------------------------------------------
# 4. Statistical Outlier Removal
# ---------------------------------------------------------------------------''',
))

# --- Site 2: isolate_trunk() signature -------------------------------------
SNIPPETS.append((
    "isolate_trunk_signature_adaptive",
    '''def isolate_trunk(
    dataset_dir: str,
    trunk_radius: float = 0.30,
    remove_green: bool = True,
    use_dbscan: bool = True,
    declutter_slices: bool = True,
    hue_green_filter: bool = True,
) -> bool:''',
    '''def isolate_trunk(
    dataset_dir: str,
    trunk_radius: float = 0.30,
    remove_green: bool = True,
    use_dbscan: bool = True,
    declutter_slices: bool = True,
    hue_green_filter: bool = True,
    adaptive_crop: bool = True,
    max_jump: float = 0.10,
) -> bool:''',
))

# --- Site 3: Step 3 call site ----------------------------------------------
SNIPPETS.append((
    "call_adaptive_crop",
    '''    # Step 3: Crop hình trụ quanh trục thân
    pcd = crop_to_trunk_cylinder(pcd, cx, cz, radius=trunk_radius)''',
    '''    # Step 3: Crop hình trụ quanh trục thân
    # [Sep 2026] Mac dinh dung crop THICH UNG theo tung lat -- truc co dinh
    # cu bi khoi goc/re phinh keo lech tam, cat mat mot nua than cay theo
    # chieu doc (xem docstring crop_to_trunk_cylinder_adaptive + CLAUDE.md
    # muc 12). --no-adaptive-crop de quay ve cach cu.
    if adaptive_crop:
        pcd = crop_to_trunk_cylinder_adaptive(pcd, cx, cz,
                                              radius=trunk_radius,
                                              max_jump=max_jump)
    else:
        pcd = crop_to_trunk_cylinder(pcd, cx, cz, radius=trunk_radius)''',
))

# --- Site 4: CLI flags + threading -----------------------------------------
SNIPPETS.append((
    "cli_flags_adaptive_crop",
    '''    parser.add_argument(
        "--no-hue-green-filter", action="store_true",
        help="Disable Hue+Saturation vegetation check in green filter (RGB-diff rule only)"
    )
    args = parser.parse_args()''',
    '''    parser.add_argument(
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
    args = parser.parse_args()''',
))

SNIPPETS.append((
    "cli_call_adaptive_crop",
    '''            declutter_slices=not args.no_slice_declutter,
            hue_green_filter=not args.no_hue_green_filter,
        )''',
    '''            declutter_slices=not args.no_slice_declutter,
            hue_green_filter=not args.no_hue_green_filter,
            adaptive_crop=not args.no_adaptive_crop,
            max_jump=args.max_jump,
        )''',
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
        print(f"{TARGET_NAME} (adaptive per-slice crop) da duoc va truoc do roi. "
              f"Khong can lam gi them.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "de tranh lam hong file. Bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_adaptivecrop")
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
    print(f"Da va thanh cong {len(applied)} doan (adaptive per-slice crop): {applied}")
    print("Chay lai buoc trunk isolation (buoc 3) de kiem tra ket qua, vd:")
    print(f'  python {TARGET_NAME} <scan_dir>')
    print("De tat buoc moi (so sanh/rollback): them --no-adaptive-crop")
    print("De chinh nguong nhay giua cac lat: --max-jump 0.15")


if __name__ == "__main__":
    main()
