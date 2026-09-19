#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trunk_isolation_slice_declutter.py
============================================
Va 05_trunk_isolation.py: them buoc "per-slice declutter" -- cat than cay
thanh nhieu lat mong theo chieu cao, trong MOI lat chi giu cum 2D (X,Z)
LON NHAT, loai bo cum phu (canh cay/vat the khac dinh gan truc chinh).

BOI CANH (xac nhan tren du lieu that, cay_0007_1805, xem CLAUDE.md muc 12):
Nguoi dung bao trunk_pointcloud.ply (va moi buoc sau no) dinh kem mot "cuc
la/canh" vao than cay. Da loai tru lan luot 3 gia thuyet bang du lieu that
truoc khi tim ra dung nguyen nhan:
  1. "Canh phinh ra theo ban kinh" -- SAI: r_max o moi lat chi cham tran
     ban kinh crop (0.3m) rai rac khap noi, khong tap trung thanh 1 dai
     lien tuc; mo phong cat o factor=2.0*baseline giu nguyen 100% diem.
  2. "DBSCAN eps qua rong, chinh eps se tach duoc" -- SAI: cum #1 (lon
     nhat) giu nguyen chieu cao ~3.08m o MOI eps tu 0.02 den 0.04, khong
     co cum #2 kich thuoc dang ke nao tach ra rieng.
  3. "Do 2 than cay rieng biet qua sat nhau" -- SAI (voi crop rong 0.6m):
     cum chinh troi muot theo do nghieng ~11 do da biet (cz -0.48 -> -0.78
     khi len cao), con cac cum phu thi nhay vi tri hon loan giua cac lat
     lien tiep (khong theo quy dao nao) -- giong rac/la vun hon la 1 than
     cay thu hai lien mach.
DUNG: chinh tren trunk_pointcloud.ply that (crop 0.3m + DBSCAN nhu pipeline
that dang chay), phan cum 2D (X,Z) TUNG LAT rieng le phat hien mot cum phu
ON DINH tu lat 11 den 18 (h ~ -0.44m den 0.38m, tuc ~0.8m trong tong ~3.08m
than cay), lech khoi truc chinh ~25-30cm, so diem nho hon han cum chinh
(80-880 so voi 1200-3700). Ngoai dai do (lat 0-10, 19-29) chi co 1 cum duy
nhat -- sach. Ket luan: day la MOT CANH CAY moc ra o giua than (khong phai
than cay thu hai, vi than cay thu hai phai xuat hien xuyen suot tu goc len
ngon), dinh vao than qua mot "co noi" du day khien DBSCAN 3D (eps=0.04,
chay xuyen suot ca khoi 3D) khong the tach -- nhung cat lat mong roi phan
cum 2D TUNG LAT DOC LAP thi tach ro rang, vi cum canh khong lien tuc theo
chieu cao voi cum than trong tung lat rieng (chi "dinh" khi nhin xuyen suot
ca khoi 3D lien tuc).

CO CHE FIX: ham moi remove_off_axis_clusters_per_slice() cat than cay thanh
n_slices lat theo truc doc (geometry_utils.vertical, dung UP_AXIS=1 da
duoc xac nhan dung cho dataset nay -- xem CLAUDE.md muc 5/12 up-axis fix),
trong moi lat chay DBSCAN 2D nho (eps=0.03) tren mat phang ngang (X,Z), chi
giu cum LON NHAT (nhieu diem nhat). Lat qua it diem (< min_points) thi GIU
NGUYEN khong xu ly (tranh doan sai khi khong chac chan -- cung triet ly voi
04b_finalize_mesh.py's sparse-only fix: chi cat khi co tin hieu ro rang,
khong doan mo ho). Chen vao pipeline SAU buoc loc mau xanh (Step 5), TRUOC
DBSCAN 3D (Step 6) -- de DBSCAN 3D con lai chi con phai xu ly du lieu da
sach canh, va van chay duoc binh thuong neu --no-dbscan.

LUU Y (rui ro da biet, chua co each khac phong ngua hoan toan): neu mot lat
that su la than cay bi vo tinh tach thanh 2 cum do vung quet thua/lo hong
that (khong phai canh), ham nay se giu nham cum nho hon la "phu" va bo cum
kia -- rui ro nay von co san trong bat ky "giu cum lon nhat" nao (kem ca
keep_largest_cluster() 3D goc), khong phai rui ro moi. Da kiem tra tren du
lieu that: voi n_slices=40, khong lat nao cua than cay that (ngoai dai co
canh) bi tach thanh 2 cum canh nhau kich thuoc tuong duong.

CACH DUNG
---------
    python patch_trunk_isolation_slice_declutter.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_trunk_isolation.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- Site 1: new function, inserted after keep_largest_cluster() -----------
SNIPPETS.append((
    "new_function_remove_off_axis_clusters_per_slice",
    '''    kept = pcd.select_by_index(np.where(mask)[0])
    print(f"  DBSCAN largest cluster: {counts.max()} / {len(pcd.points)} pts "
          f"({len(unique)} clusters total)")
    return kept


# ---------------------------------------------------------------------------
# Main isolation pipeline
# ---------------------------------------------------------------------------

def isolate_trunk(''',
    '''    kept = pcd.select_by_index(np.where(mask)[0])
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

def isolate_trunk(''',
))

# --- Site 2: isolate_trunk() signature: add declutter_slices flag ----------
SNIPPETS.append((
    "isolate_trunk_signature",
    '''def isolate_trunk(
    dataset_dir: str,
    trunk_radius: float = 0.30,
    remove_green: bool = True,
    use_dbscan: bool = True,
) -> bool:''',
    '''def isolate_trunk(
    dataset_dir: str,
    trunk_radius: float = 0.30,
    remove_green: bool = True,
    use_dbscan: bool = True,
    declutter_slices: bool = True,
) -> bool:''',
))

# --- Site 3: call the new step between green filter and DBSCAN -------------
SNIPPETS.append((
    "call_declutter_between_green_and_dbscan",
    '''    # Step 5: Loại thực vật xanh
    if remove_green:
        pcd = remove_green_vegetation(pcd)

    # Step 6: DBSCAN — giữ cluster lớn nhất (thân cây)
    if use_dbscan and len(pcd.points) > 100:
        pcd = keep_largest_cluster(pcd, eps=0.04, min_points=30)''',
    '''    # Step 5: Loại thực vật xanh
    if remove_green:
        pcd = remove_green_vegetation(pcd)

    # Step 5b: Per-slice declutter -- loai canh cay/vat the phu dinh vao
    # giua than (xem CLAUDE.md muc 12). Chay TRUOC DBSCAN 3D vi canh dinh
    # lien tuc theo chieu cao se bi keep_largest_cluster() (chay xuyen suot
    # ca khoi 3D) nhap chung vao cum than -- xu ly TUNG LAT truoc se tach no
    # ra som, giup DBSCAN 3D phia sau lam viec tren du lieu da sach hon.
    if declutter_slices and len(pcd.points) > 100:
        pcd = remove_off_axis_clusters_per_slice(pcd, n_slices=40, eps=0.03, min_points=10)

    # Step 6: DBSCAN — giữ cluster lớn nhất (thân cây)
    if use_dbscan and len(pcd.points) > 100:
        pcd = keep_largest_cluster(pcd, eps=0.04, min_points=30)''',
))

# --- Site 4: CLI flag + threading into single-scan call ---------------------
SNIPPETS.append((
    "cli_flag_no_slice_declutter",
    '''    parser.add_argument(
        "--no-dbscan", action="store_true",
        help="Disable DBSCAN largest-cluster selection"
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
        )
        if not ok:
            sys.exit(1)''',
    '''    parser.add_argument(
        "--no-dbscan", action="store_true",
        help="Disable DBSCAN largest-cluster selection"
    )
    parser.add_argument(
        "--no-slice-declutter", action="store_true",
        help="Disable per-slice off-axis cluster removal (branch/neighbor-object filter)"
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
        )
        if not ok:
            sys.exit(1)''',
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
        print(f"{TARGET_NAME} (per-slice declutter) da duoc va truoc do roi. Khong can lam gi them.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "de tranh lam hong file. Bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_slicedeclutter")
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
    print(f"Da va thanh cong {len(applied)} doan (per-slice declutter): {applied}")
    print("Chay lai buoc trunk isolation (buoc 3) de kiem tra ket qua, vd:")
    print(f'  python {TARGET_NAME} <scan_dir>')
    print("De tat buoc moi (so sanh/rollback): them --no-slice-declutter")


if __name__ == "__main__":
    main()
