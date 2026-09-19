#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trunk_isolation_slice_radius_ratio.py
============================================
Va 05_trunk_isolation.py: them bo loc TY LE BAN KINH THEO LAT vao
crop_to_trunk_cylinder_adaptive() -- loai bo lat nao co ban kinh lon bat
thuong so voi ban kinh nen cua chinh scan do (mac dinh > 3.0 lan).

BOI CANH (cay_0029_1805, xem CLAUDE.md muc 12):
trunk_pointcloud.ply cua cay_0029 ra mot khoi bet nam cheo 20,320 diem --
PCA lech 60.9 do khoi phuong thang dung, ty le L1/L2 chi 1.4 (cay that:
45-574). Nguoi dung nghi sai truc. DA DO VA LOAI TRU: IMU tho cua scan nay
chi lech 4.3 do khoi world +Y, ngang voi cac scan chay tot (2.8-8.1 do) --
truc Y VAN DUNG. Y tuong "lay tam quy dao camera lam moc" cung da do va
loai tru: sai so 8.7-48.6cm tren cac scan da biet dung (cay_0005 lech
48.6cm, con vuot ca chot chan 40cm) vi nguoi quay khong di vong tron deu
nhu CLAUDE.md muc 4 mo ta.

NGUYEN NHAN THAT (tu bang per-slice):
  - Adaptive crop bam DUNG than cay: seed lat 18 tam (0.099,-0.279)
    r_med=3.5cm, 34/40 lat bam duoc voi buoc nhay 0.2-2.7cm, ban kinh deu
    2.0-4.1cm tu lat 6 den 39.
  - NHUNG 6 lat day (0-5) nam trong mot khoi dat/bui ram day 0.53m ma
    remove_ground_plane() khong cat het (no chi cat toi ground_z=-1.074).
    Cac lat do co r_med 16-25cm va gop ~21,900 / 37,950 diem (58%), trong
    khi moi lat than cay that chi ~500 diem.
  - keep_largest_cluster() sau do lam dung chuc nang cua no -- giu cum lien
    thong LON NHAT THEO SO DIEM -- va khoi dat nang ky hon than cay, nen
    THAN CAY BI VUT DI, khoi dat duoc giu lai.

Tuc la trong tai cuoi cung cua pipeline la DEM DIEM, mot cuoc binh chon
theo so dong, khong he co khai niem "hinh dang than cay" -- du rang den
buoc do adaptive crop DA xac dinh chinh xac than cay o dau. cay_0005 la
cung co che nhung may hon: dia dat cua no thua cuoc binh chon thay vi
thang.

FIX: sau khi bam xong tam tung lat, tinh r_med cua tung lat va loai lat nao
co r_med > max_radius_ratio * baseline_r. Than cay that giu ban kinh trong
vai lan ban kinh nen; khoi dat/bui ram rong gap nhieu lan.

KIEM CHUNG NGUONG (6 scan that, truoc khi va):
  k=2.0 QUA CHAT: tren cay_0007 no loai ca lat 7, 11, 27 -- nam GIUA than
       cay, tuc la duc lo vao than that.
  k=4.0 QUA LONG: cay_0010 lot luoi (lat day ty le 2.97).
  k=3.0 TACH SACH tren ca 6 scan:
    - cay_0029: loai lat 0-5, 22,884 pts (60%) -> con 2.28m, 15,043 pts
    - cay_0005: loai lat 0-2, 12,002 pts (41%) -> con 2.29m, 17,448 pts
    - cay_0004: loai lat 0-4, 23,317 pts (46%) -> con 2.43m, 27,178 pts
    - cay_0007: loai lat 0-3, 24,846 pts (25%) -> con 2.81m, 72,416 pts
                (chi cac lat day, KHONG dung vao giua than)
    - cay_0010: khong loai lat nao
    - cay_0013: khong loai lat nao
Luu y: cac lat bi loai deu la vung goc lan dat -- pipeline se ngan hon o
day (vd cay_0007 mat 0.31m duoi cung). Do la vung du lieu lan dat, khong
dung de do dia y, nhung day VAN LA mot thay doi hanh vi tren cac scan
truoc do da duoc duyet bang mat.

Dat max_radius_ratio <= 0 (hoac --max-radius-ratio 0) de tat bo loc nay.

CACH DUNG
---------
    python patch_trunk_isolation_slice_radius_ratio.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_trunk_isolation.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- Site 1: signature ------------------------------------------------------
SNIPPETS.append((
    "adaptive_signature_radius_ratio",
    '''    seed_max_offset: float = 0.40,
    max_coast_ratio: float = 0.50,
    z_min_offset: float = 0.05,
) -> o3d.geometry.PointCloud:''',
    '''    seed_max_offset: float = 0.40,
    max_coast_ratio: float = 0.50,
    max_radius_ratio: float = 3.0,
    z_min_offset: float = 0.05,
) -> o3d.geometry.PointCloud:''',
))

# --- Site 2: per-slice radius-ratio filter in the final crop ---------------
SNIPPETS.append((
    "slice_radius_ratio_filter",
    '''    keep = np.zeros(len(pts), dtype=bool)
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
          f"{len(pts)} -> {len(out.points)} pts "
          f"[tracked={n_tracked}, coast={n_coast}/{n_data} = {coast_ratio:.0%}]")
    return out''',
    '''    # Cat theo tam rieng tung lat, dong thoi LOAI LAT CO BAN KINH BAT
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
        if max_radius_ratio > 0 and baseline_r > 1e-9 and \\
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
    return out''',
))

# --- Site 3: isolate_trunk signature + call --------------------------------
SNIPPETS.append((
    "isolate_trunk_signature_radius_ratio",
    '''    max_jump: float = 0.10,
    seed_max_offset: float = 0.40,
) -> bool:''',
    '''    max_jump: float = 0.10,
    seed_max_offset: float = 0.40,
    max_radius_ratio: float = 3.0,
) -> bool:''',
))

SNIPPETS.append((
    "isolate_trunk_call_radius_ratio",
    '''                                              max_jump=max_jump,
                                              seed_max_offset=seed_max_offset)''',
    '''                                              max_jump=max_jump,
                                              seed_max_offset=seed_max_offset,
                                              max_radius_ratio=max_radius_ratio)''',
))

# --- Site 4: CLI -----------------------------------------------------------
SNIPPETS.append((
    "cli_max_radius_ratio",
    '''    parser.add_argument(
        "--seed-max-offset", type=float, default=0.40,
        help="Max distance (m) from the rough trunk axis for an adaptive-crop seed "
             "candidate -- guards against seeding on a neighbouring tree (default: 0.40)"
    )''',
    '''    parser.add_argument(
        "--seed-max-offset", type=float, default=0.40,
        help="Max distance (m) from the rough trunk axis for an adaptive-crop seed "
             "candidate -- guards against seeding on a neighbouring tree (default: 0.40)"
    )
    parser.add_argument(
        "--max-radius-ratio", type=float, default=3.0,
        help="Drop a height slice whose radius exceeds this multiple of the scan's "
             "own baseline trunk radius -- removes leftover ground/undergrowth mass "
             "that would otherwise outweigh the trunk. 0 disables (default: 3.0)"
    )''',
))

SNIPPETS.append((
    "cli_call_max_radius_ratio",
    '''            max_jump=args.max_jump,
            seed_max_offset=args.seed_max_offset,
        )''',
    '''            max_jump=args.max_jump,
            seed_max_offset=args.seed_max_offset,
            max_radius_ratio=args.max_radius_ratio,
        )''',
))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if all(new in content for _, _, new in SNIPPETS):
        print(f"{TARGET_NAME} (slice radius-ratio filter) da duoc va truoc do roi.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Luu y: patch nay yeu cau patch_trunk_isolation_adaptive_crop.py va "
              "patch_trunk_isolation_adaptive_seed_gate.py da duoc ap dung truoc.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_radiusratio")
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
    print(f"Da va thanh cong {len(applied)} doan (slice radius-ratio filter): {applied}")
    print("Chay lai buoc trunk isolation (buoc 3), vd:")
    print(f'  python {TARGET_NAME} <scan_dir>')
    print("Tat bo loc nay: --max-radius-ratio 0")


if __name__ == "__main__":
    main()
