#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trunk_isolation_adaptive_seed_gate.py
============================================
Va 05_trunk_isolation.py: them 2 lop bao ve cho
crop_to_trunk_cylinder_adaptive() (da them boi
patch_trunk_isolation_adaptive_crop.py):
  1. CHOT CHAN SEED (seed_max_offset, mac dinh 0.40m)
  2. LUOI AN TOAN COAST (max_coast_ratio, mac dinh 0.50)

BOI CANH (cay_0013_1805, xem CLAUDE.md muc 12):
Sau khi adaptive crop chay tot tren 5 scan, cay_0013_1805 hong nang:
trunk_pointcloud.ply ra mot dam may vo dinh hinh 15,784 diem. Nguoi dung
nghi "than cay nay to hon nhieu cac cay khac" -- DU LIEU THAT BAC BO gia
thuyet do:
  - Than that nam o (0.34, 0.11), ban kinh chi 12.1cm -- hoan toan binh
    thuong (cay_0007 con to hon: 16.9cm, van chay tot).
  - Nhung o dai do cao 1.22-1.61m co MOT CAY KHAC dung cach ~55-60cm, va o
    dai do no NHIEU DIEM HON cay dich.
  - Quy tac chon seed cu ("lat trong 60% tren co ban kinh gan trung vi
    nhat") KHONG he biet ung vien nao thuoc cay dich, nen no khoa vao cay
    hang xom (seed lat 28, tam (-0.033,-0.539)), roi bam theo cay do,
    coast 21/40 lat, quet bua moi thu trong 30cm quanh mot tam dong cung
    sai -> ra dam may vo dinh hinh.
  - Dang chu y: truc tho cu (0.245,-0.012) lai GAN cay dich (lech 15cm) va
    XA cay hang xom (lech 55cm). Tuc la truc tho van chi dung huong -- chi
    la thuat toan cu khong dung no sau buoc crop rong.

FIX 1 -- CHOT CHAN SEED: blob ung vien lam seed phai nam trong
seed_max_offset quanh truc tho (cx, cz). Nguong 0.40m duoc chon bang SO DO
THAT, khong phai uoc chung:
  - cay_0004: than dich lech toi da 29.5cm khoi truc tho (bi khoi goc/re
    phinh keo lech) -> PHAI GIU.
  - cay_0013: cay hang xom lech toi thieu 55.5cm -> PHAI LOAI.
  0.40m nam giua, bien an toan rong ca hai phia.

FIX 2 -- LUOI AN TOAN COAST: neu qua max_coast_ratio so lat co du lieu bi
"mat dau" (khong blob nao trong max_jump), coi nhu bam hong -> tu dong
quay ve crop co dinh cu kem canh bao, thay vi xuat ra rac. cay_0013 ban cu
coast 21/40 se kich hoat ngay lop nay.

KET QUA KIEM CHUNG (prototype tren 6 scan, truoc khi va):
  - cay_0013: seed chuyen tu (-0.033,-0.539) [cay hang xom] sang
    (0.359,0.152) r_med=11.4cm [cay dich, khop voi so do thuc te
    (0.34,0.11) r~12cm]. Coast 21/40 -> 7/40. Tong 29,969 -> 102,866 pts.
    54 blob bi chot chan loai bo.
  - cay_0004/0007/0010/0005/0009: seed KHONG DOI, tong diem khong doi
    (50,496->50,499; 97,262->97,268; 84,769->84,780; 29,450->29,450;
    96,895->96,895) -- khong hoi quy.
  - Coast ratio ca 6 scan: 5-25%, xa nguong 50% -> luoi an toan khong bat
    nham scan tot nao.

CACH DUNG
---------
    python patch_trunk_isolation_adaptive_seed_gate.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_trunk_isolation.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- Site 1: signature -- them seed_max_offset + max_coast_ratio ----------
SNIPPETS.append((
    "adaptive_signature_guards",
    '''    max_jump: float = 0.10,
    z_min_offset: float = 0.05,
) -> o3d.geometry.PointCloud:''',
    '''    max_jump: float = 0.10,
    seed_max_offset: float = 0.40,
    max_coast_ratio: float = 0.50,
    z_min_offset: float = 0.05,
) -> o3d.geometry.PointCloud:''',
))

# --- Site 2: chot chan seed ------------------------------------------------
SNIPPETS.append((
    "seed_gate",
    '''    upper_threshold = y_max - 0.6 * (y_max - y_min)
    cands = []
    for i in range(n_slices):
        if edges[i] < upper_threshold or not slice_blobs[i]:
            continue
        cands.append((i, max(slice_blobs[i], key=lambda b: b[1])))
    if not cands:
        print("  Adaptive crop: khong tim duoc lat seed -> fallback crop co dinh")
        return crop_to_trunk_cylinder(pcd, cx, cz, radius=radius,
                                      z_min_offset=z_min_offset)''',
    '''    # CHOT CHAN SEED: ung vien phai nam gan truc tho (cx, cz). Neu khong co
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
                                      z_min_offset=z_min_offset)''',
))

# --- Site 3: luoi an toan coast + cat cuoi ---------------------------------
SNIPPETS.append((
    "coast_sanity_check",
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
          f"{len(pts)} -> {len(out.points)} pts [tracked={n_tracked}, coast={n_coast}]")
    return out''',
    '''    # LUOI AN TOAN: neu qua nua so lat co du lieu bi mat dau thi viec bam
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
          f"{len(pts)} -> {len(out.points)} pts "
          f"[tracked={n_tracked}, coast={n_coast}/{n_data} = {coast_ratio:.0%}]")
    return out''',
))

# --- Site 4: isolate_trunk() truyen tham so xuong --------------------------
SNIPPETS.append((
    "isolate_trunk_pass_guards",
    '''        pcd = crop_to_trunk_cylinder_adaptive(pcd, cx, cz,
                                              radius=trunk_radius,
                                              max_jump=max_jump)''',
    '''        pcd = crop_to_trunk_cylinder_adaptive(pcd, cx, cz,
                                              radius=trunk_radius,
                                              max_jump=max_jump,
                                              seed_max_offset=seed_max_offset)''',
))

SNIPPETS.append((
    "isolate_trunk_signature_seed_gate",
    '''    adaptive_crop: bool = True,
    max_jump: float = 0.10,
) -> bool:''',
    '''    adaptive_crop: bool = True,
    max_jump: float = 0.10,
    seed_max_offset: float = 0.40,
) -> bool:''',
))

# --- Site 5: CLI -----------------------------------------------------------
SNIPPETS.append((
    "cli_seed_max_offset",
    '''    parser.add_argument(
        "--max-jump", type=float, default=0.10,
        help="Max horizontal shift (m) of the trunk center between adjacent slices "
             "in adaptive crop (default: 0.10)"
    )''',
    '''    parser.add_argument(
        "--max-jump", type=float, default=0.10,
        help="Max horizontal shift (m) of the trunk center between adjacent slices "
             "in adaptive crop (default: 0.10)"
    )
    parser.add_argument(
        "--seed-max-offset", type=float, default=0.40,
        help="Max distance (m) from the rough trunk axis for an adaptive-crop seed "
             "candidate -- guards against seeding on a neighbouring tree (default: 0.40)"
    )''',
))

SNIPPETS.append((
    "cli_call_seed_max_offset",
    '''            adaptive_crop=not args.no_adaptive_crop,
            max_jump=args.max_jump,
        )''',
    '''            adaptive_crop=not args.no_adaptive_crop,
            max_jump=args.max_jump,
            seed_max_offset=args.seed_max_offset,
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
        print(f"{TARGET_NAME} (seed gate + coast sanity check) da duoc va truoc do roi. "
              f"Khong can lam gi them.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Luu y: patch nay yeu cau patch_trunk_isolation_adaptive_crop.py da "
              "duoc ap dung truoc.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_seedgate")
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
    print(f"Da va thanh cong {len(applied)} doan (seed gate + coast check): {applied}")
    print("Chay lai buoc trunk isolation (buoc 3), vd:")
    print(f'  python {TARGET_NAME} <scan_dir>')
    print("Chinh chot chan seed neu can: --seed-max-offset 0.50")


if __name__ == "__main__":
    main()
