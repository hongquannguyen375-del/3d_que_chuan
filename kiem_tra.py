# -*- coding: utf-8 -*-
"""Cac cong cu kiem tra va chan doan, gop tu 6 script rieng le.

Dung:
    python kiem_tra.py <lenh> [tham so...]
    python kiem_tra.py <lenh> --help

Cac lenh:
    anh         anh depth/rgb/confidence hong (doc thu bang ca cv2 va PIL)
    pointcloud  so diem, NaN/Inf, hop bao cua mot dam may diem
    ply         cau truc PLY nhi phan -- quet ca corpus tim file hong o muc byte
    lech        khoang cach tam giua dam may / than / quy dao camera / mesh
    truc        so truc than uoc tinh tu dam may tho va dam may than
    odometry    thu tu cot that cua odometry.csv so voi cot ma code gia dinh

Moi lenh giu NGUYEN tham so cua script cu -- chi thay
"python check_pointcloud.py X" bang "python kiem_tra.py pointcloud X".

Chi "ply" duoc dung nhu thu vien: export_lichen_excel.py goi
ply_is_sane() truoc khi giao mesh cho open3d (xem lenh "ply" ben duoi).
Nap module nay KHONG chay gi ca -- moi cong cu chi chay khi goi lenh.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
import argparse
import os
import struct
import numpy as np
import open3d as o3d
import geometry_utils
import csv


# ===========================================================================
#  kiem_tra.py anh  (truoc day: check_image_integrity_v2.py)
# ===========================================================================
# check_image_integrity_v2.py
# =============================
# Ban nang cap cua check_image_integrity.py: kiem tra tung file anh bang
# CA HAI thu vien cv2 va Pillow (PIL), vi 2 thu vien co the khong thong nhat
# ve viec 1 file da bi hong hay chua (02_pointcloud_and_mesh.py dung cv2,
# 02_mesh_tsdf.py dung Pillow - can kiem tra ca 2 de khong bo sot).
#
# CACH DUNG
# ---------
#     python kiem_tra.py anh "E:/Thucdia-18May2026/cay_0003_1805"
#
# Script chi DOC, khong sua/xoa gi ca.

def readable_cv2(path: Path) -> bool:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return img is not None
    except Exception:
        return False


def readable_pil(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def check_folder(folder: Path, exts: tuple[str, ...], label: str) -> list[str]:
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    if not files:
        print(f"  ({label}: khong co file nao trong {folder})")
        return []

    sizes = [p.stat().st_size for p in files]
    median_size = statistics.median(sizes)

    bad = []
    print(f"  Dang kiem tra {len(files)} file trong {folder} (cv2 + PIL) ...")
    for i, p in enumerate(files, 1):
        if i % 500 == 0:
            print(f"    ... {i}/{len(files)}")
        ok_cv2 = readable_cv2(p)
        ok_pil = readable_pil(p)
        if not (ok_cv2 and ok_pil):
            size = p.stat().st_size
            ratio = size / median_size if median_size else 0
            which = []
            if not ok_cv2:
                which.append("cv2")
            if not ok_pil:
                which.append("PIL")
            bad.append(
                f"{p.name}  ({size:,} bytes, ~{ratio:.0%} so voi trung vi "
                f"{median_size:,.0f} bytes)  -- hong theo: {'+'.join(which)}"
            )
    return bad


def _main_anh() -> None:
    # Kiem thu vien o day, khong o cap module -- xem ghi chu dau file.
    global Image, cv2
    try:
        import cv2
    except ImportError:
        print("Can cai opencv-python: pip install opencv-python")
        sys.exit(1)
    try:
        from PIL import Image
    except ImportError:
        print("Can cai pillow: pip install pillow")
        sys.exit(1)
    if len(sys.argv) != 2:
        print('Cach dung: python kiem_tra.py anh "duong/dan/toi/thu_muc_scan"')
        sys.exit(1)

    scan_dir = Path(sys.argv[1]).resolve()
    if not scan_dir.exists():
        print(f"Khong tim thay thu muc: {scan_dir}")
        sys.exit(1)

    print("=" * 70)
    print(f"  Kiem tra tinh toan ven anh (cv2 + PIL) trong: {scan_dir}")
    print("=" * 70)

    all_bad: dict[str, list[str]] = {}

    depth_dir = scan_dir / "depth"
    if depth_dir.is_dir():
        all_bad["depth/"] = check_folder(depth_dir, (".png",), "depth")

    rgb_dir = scan_dir / "rgb"
    if rgb_dir.is_dir():
        all_bad["rgb/"] = check_folder(rgb_dir, (".jpg", ".jpeg", ".png"), "rgb")

    conf_dir = scan_dir / "confidence"
    if conf_dir.is_dir():
        all_bad["confidence/"] = check_folder(conf_dir, (".png",), "confidence")

    print()
    print("=" * 70)
    total_bad = sum(len(v) for v in all_bad.values())
    if total_bad == 0:
        print("  KET QUA: Khong phat hien file nao bi hong (ca cv2 va PIL deu doc duoc). Du lieu OK.")
    else:
        print(f"  KET QUA: Phat hien {total_bad} file bi hong / khong doc duoc:")
        for folder_label, bad_list in all_bad.items():
            if not bad_list:
                continue
            print(f"\n  --- {folder_label} ({len(bad_list)} file hong) ---")
            for line in bad_list:
                print(f"    - {line}")
    print("=" * 70)


# ===========================================================================
#  kiem_tra.py pointcloud  (truoc day: check_pointcloud.py)
# ===========================================================================
# check_pointcloud.py
# ====================
# Chan doan nhanh file output/pointcloud.ply cua 1 scan: dem so diem, kiem tra
# co toa do NaN/Inf hay khong, va uoc luong RAM can de xu ly - de biet truoc
# 05_trunk_isolation.py bi crash do DU LIEU HONG hay do HET RAM.
#
# Script CHI DOC, khong tinh toan nang (khong goi KDTree/DBSCAN), nen chay
# nhanh (vai giay den vai chuc giay) va an toan, khong lam may bi treo.
#
# CACH DUNG
# ---------
#     python kiem_tra.py pointcloud "E:/Thucdia-18May2026/cay_0003_1805"

def _main_pointcloud() -> None:
    # Kiem thu vien o day, khong o cap module -- xem ghi chu dau file.
    global np, o3d
    try:
        import numpy as np
        import open3d as o3d
    except ImportError as e:
        print(f"Thieu thu vien: {e}")
        sys.exit(1)
    if len(sys.argv) != 2:
        print('Cach dung: python kiem_tra.py pointcloud "duong/dan/toi/thu_muc_scan"')
        sys.exit(1)

    scan_dir = Path(sys.argv[1]).resolve()
    pcd_path = scan_dir / "output" / "pointcloud.ply"

    if not pcd_path.exists():
        print(f"Khong tim thay file: {pcd_path}")
        sys.exit(1)

    size_mb = pcd_path.stat().st_size / (1024 * 1024)
    print("=" * 70)
    print(f"  File: {pcd_path}")
    print(f"  Kich thuoc file: {size_mb:.1f} MB")
    print("=" * 70)

    print("\n  Dang doc file (chi doc, khong tinh toan)...")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)
    n = len(points)

    print(f"\n  So diem: {n:,}")
    if n == 0:
        print("  CANH BAO: point cloud rong!")
        return

    # Uoc luong RAM can (Open3D + numpy + cac buffer trung gian khi loc/KDTree
    # thuong can gap 5-10 lan kich thuoc du lieu goc)
    bytes_per_point = 3 * 8  # float64 x,y,z trong numpy
    raw_mb = n * bytes_per_point / (1024 * 1024)
    print(f"  RAM toi thieu cho mang toa do (numpy): ~{raw_mb:.0f} MB")
    print(f"  RAM uoc tinh khi chay KDTree/SOR/DBSCAN: ~{raw_mb*8:.0f}-{raw_mb*15:.0f} MB "
          f"(Open3D thuong can gap 8-15 lan)")

    # Kiem tra NaN / Inf
    nan_mask = np.isnan(points).any(axis=1)
    inf_mask = np.isinf(points).any(axis=1)
    n_nan = int(nan_mask.sum())
    n_inf = int(inf_mask.sum())

    print(f"\n  Diem co toa do NaN : {n_nan:,} ({100*n_nan/n:.4f}%)")
    print(f"  Diem co toa do Inf : {n_inf:,} ({100*n_inf/n:.4f}%)")

    # Bounding box cua cac diem HOP LE (bo NaN/Inf) de xem co outlier khong
    valid_mask = ~(nan_mask | inf_mask)
    valid_pts = points[valid_mask]
    if len(valid_pts) > 0:
        mins = valid_pts.min(axis=0)
        maxs = valid_pts.max(axis=0)
        print(f"\n  Bounding box (diem hop le, met):")
        print(f"    X: {mins[0]:.3f} -> {maxs[0]:.3f}  (rong {maxs[0]-mins[0]:.3f} m)")
        print(f"    Y: {mins[1]:.3f} -> {maxs[1]:.3f}  (rong {maxs[1]-mins[1]:.3f} m)")
        print(f"    Z: {mins[2]:.3f} -> {maxs[2]:.3f}  (rong {maxs[2]-mins[2]:.3f} m)")
        # Canh bao neu bounding box qua lon bat thuong cho 1 than cay
        # (than cay thuong chi rong vai chuc cm, cao vai met)
        if (maxs[0]-mins[0]) > 50 or (maxs[1]-mins[1]) > 50 or (maxs[2]-mins[2]) > 50:
            print("\n  CANH BAO: bounding box qua 50m o mot chieu - co diem outlier"
                  " o rat xa, du lieu depth/pose co the bi loi o mot so khung hinh.")

    print("\n" + "=" * 70)
    print("  KET LUAN:")
    if n_nan > 0 or n_inf > 0:
        print(f"  -> CO {n_nan + n_inf:,} diem toa do khong hop le (NaN/Inf)."
              " Day co the la nguyen nhan lam crash 05_trunk_isolation.py.")
    if n > 5_000_000:
        print(f"  -> Point cloud RAT LON ({n:,} diem). Voi may 8GB RAM,"
              " cac buoc KDTree/SOR/DBSCAN co nguy co het bo nho.")
    if n_nan == 0 and n_inf == 0 and n <= 5_000_000:
        print("  -> Du lieu toa do hop le, so luong diem o muc binh thuong."
              " Nguyen nhan crash co the khac (RAM tai thoi diem chay, dung"
              " luong Open3D/DBSCAN cu the).")
    print("=" * 70)


# ===========================================================================
#  kiem_tra.py ply  (truoc day: ply_check.py)
# ===========================================================================
# Kiem tra file PLY nhi phan TRUOC khi giao cho open3d.
#
# LY DO TON TAI: mot byte hong trong PLY lam open3d chet bang segmentation
# fault -- o tang C, nen try/except cua Python KHONG bat duoc, ca tien trinh
# di theo. Gap that tren cay_0049_1805 (bo 18May): hai byte dem so dinh cua
# mat bi lat thanh 139 va 220 thay vi 3, nen open3d co doc 139 dinh cho mot
# tam giac roi chay vuot vung dem. Kich thuoc file khop chinh xac tung byte
# voi header, toa do deu huu han -- tuc file KHONG bi cat cut hay ghi do,
# dung hai byte bi lat gia tri. Do la hong o muc byte (CLAUDE.md muc 4).
#
# Vi khong bat duoc loi, cach duy nhat la kiem truoc. Kiem tra o day la
# thuan Python, khong nap hinh hoc, nen nhanh -- chi doc header cong voi
# phan ban ghi mat.
#
# Dung nhu module:
#     from ply_check import ply_is_sane
#     ok, ly_do = ply_is_sane(path)
#
# Hoac chay truc tiep de quet ca corpus:
#     python kiem_tra.py ply "D:\Backup\Thucdia-18May2026"
#     python kiem_tra.py ply "D:\Backup\Thucdia-18May2026" --name trunk_mesh_final.ply

# Kich thuoc tung kieu du lieu PLY, tinh theo byte.
TYPE_SIZE = {
    "char": 1, "uchar": 1, "int8": 1, "uint8": 1,
    "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
    "int": 4, "uint": 4, "int32": 4, "uint32": 4,
    "float": 4, "float32": 4,
    "double": 8, "float64": 8,
}


MAX_HEADER = 65536      # header PLY that khong bao gio gan muc nay


# File nam tren duong chay that: mot file trong nhom nay hong la phai dung lai
# ban quet. Moi file .ply khac trong output/ la san pham phu khong ai doc, va
# hong o do KHONG anh huong ket qua -- gop chung vao mot danh sach "can dung
# lai" la bao dong gia, vi 63 trong 64 file hong dau tien do duoc la loai nay.
ACTIVE_FILES = {
    "pointcloud.ply",             # buoc 2 -> buoc 3
    "trunk_pointcloud.ply",       # nguon mau, mat na, nhan cho buoc 5/6/8
    "trunk_mesh_poisson.ply",     # nguon hinh hoc hien tai (buoc 4b)
    "trunk_mesh_recolored.ply",   # buoc 5 -> 6
    "trunk_mesh_trimmed.ply",     # buoc 6 -> 7
    "trunk_mesh_final.ply",       # buoc 7 -> 8, va export_lichen_excel.py doc
    "trunk_mesh_detected.ply",    # ket qua cuoi
}


# Vi sao tung file duoi day khong con quan trong -- de nguoi doc khong phai
# tra cuu CLAUDE.md moi biet co nen lo hay khong.
INERT_REASON = {
    "mesh.ply": "mesh nhanh, khong buoc nao tu 3->8 doc; nay da co --no-mesh",
    "trunk_mesh_tsdf.ply": "Poisson da thay lam nguon hinh hoc, chi de lui ve",
    "trunk_mesh_detected_CHEO.ply": "file doi chieu cua annotate_lichen",
    "trunk_labeled.ply": "mau hoa theo huong doc, khong chay vao ket qua",
}


def _parse_header(raw):
    """Doc header PLY. Tra ve (offset_du_lieu, dinh_dang, danh_sach_element).

    Moi element la dict: {"name", "count", "props"}, trong do props la danh
    sach ("scalar", kieu) hoac ("list", kieu_dem, kieu_phan_tu).
    """
    end = raw.find(b"end_header\n", 0, MAX_HEADER)
    if end < 0:
        return None, None, None
    offset = end + len(b"end_header\n")
    try:
        text = raw[:end].decode("ascii")
    except UnicodeDecodeError:
        return None, None, None

    fmt = None
    elements = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            try:
                elements.append({"name": parts[1], "count": int(parts[2]),
                                 "props": []})
            except ValueError:
                return None, None, None
        elif parts[0] == "property" and elements:
            if parts[1] == "list" and len(parts) >= 5:
                elements[-1]["props"].append(("list", parts[2], parts[3]))
            elif len(parts) >= 3:
                elements[-1]["props"].append(("scalar", parts[1]))
    return offset, fmt, elements


def ply_is_sane(path, check_lists=True):
    """Kiem tra mot file PLY nhi phan co an toan de open3d doc khong.

    Tra ve (True, "") neu khong tim thay van de, hoac (False, ly_do).

    Chi kiem duoc file 'binary_little_endian'. Voi ascii hoac big-endian thi
    tra ve True kem ly do "khong kiem" -- khong gia vo la da kiem.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except (IOError, OSError) as e:
        return False, "khong doc duoc file: %s" % e

    if not raw.startswith(b"ply"):
        return False, "khong phai file PLY (thieu chu ky 'ply')"

    offset, fmt, elements = _parse_header(raw)
    if offset is None:
        return False, "header hong hoac khong co 'end_header'"
    if fmt != "binary_little_endian":
        return True, "khong kiem (dang %s, chi kiem binary_little_endian)" % fmt
    if not elements:
        return False, "header khong khai element nao"

    pos = offset
    total = len(raw)
    for el in elements:
        fixed = 0
        has_list = False
        for p in el["props"]:
            if p[0] == "scalar":
                if p[1] not in TYPE_SIZE:
                    return True, "khong kiem (kieu la '%s')" % p[1]
                fixed += TYPE_SIZE[p[1]]
            else:
                has_list = True

        if not has_list:
            need = el["count"] * fixed
            if pos + need > total:
                return False, ("element '%s' can %d byte nhung file chi con %d"
                               % (el["name"], need, total - pos))
            pos += need
            continue

        # Element co truong danh sach (thuong la mat: 1 byte dem + N chi so).
        # Phai di tung ban ghi, vi do dai moi ban ghi phu thuoc byte dem do.
        if not check_lists:
            return True, ""
        for i in range(el["count"]):
            rec = pos
            for p in el["props"]:
                if p[0] == "scalar":
                    rec += TYPE_SIZE[p[1]]
                    continue
                cnt_t, item_t = p[1], p[2]
                if cnt_t not in TYPE_SIZE or item_t not in TYPE_SIZE:
                    return True, "khong kiem (kieu danh sach la)"
                cs = TYPE_SIZE[cnt_t]
                if rec + cs > total:
                    return False, ("het du lieu o ban ghi '%s' thu %d/%d"
                                   % (el["name"], i, el["count"]))
                n = int.from_bytes(raw[rec:rec + cs], "little")
                # Mesh tam giac tu pipeline nay luon la 3. Byte dem khac 3 la
                # dau hieu hong -- va la thu lam open3d chet.
                if el["name"] == "face" and n != 3:
                    return False, ("ban ghi 'face' thu %d/%d khai %d dinh "
                                   "(dang le 3) -- byte dem bi hong, open3d "
                                   "se doc vuot vung dem va chet"
                                   % (i, el["count"], n))
                if n > 255:
                    return False, ("ban ghi '%s' thu %d khai %d phan tu -- "
                                   "vo ly" % (el["name"], i, n))
                rec += cs + n * TYPE_SIZE[item_t]
            if rec > total:
                return False, ("ban ghi '%s' thu %d/%d vuot qua cuoi file"
                               % (el["name"], i, el["count"]))
            pos = rec

    if pos != total:
        return False, ("thua %d byte sau element cuoi (file dai hon header "
                       "khai)" % (total - pos))
    return True, ""


# --------------------------------------------------------------------------- #
def _main_ply():
    ap = argparse.ArgumentParser(
        description="Quet cac file PLY trong mot thu muc corpus, tim file "
                    "hong o muc byte truoc khi no lam open3d chet.")
    ap.add_argument("raw_data", help="Thu muc goc chua cac ban quet")
    ap.add_argument("--name", default=None,
                    help="Chi kiem mot ten file trong output/ "
                         "(mac dinh: moi file .ply)")
    args = ap.parse_args()

    if not os.path.isdir(args.raw_data):
        sys.exit("Khong thay thu muc: %s" % args.raw_data)

    n_ok = n_skip = 0
    active_bad, inert_bad = [], []
    for scan in sorted(os.listdir(args.raw_data)):
        out = os.path.join(args.raw_data, scan, "output")
        if not os.path.isdir(out):
            continue
        names = [args.name] if args.name else sorted(
            f for f in os.listdir(out) if f.lower().endswith(".ply"))
        for nm in names:
            p = os.path.join(out, nm)
            if not os.path.exists(p):
                continue
            ok, why = ply_is_sane(p)
            if ok and why:
                n_skip += 1
            elif ok:
                n_ok += 1
            elif nm in ACTIVE_FILES:
                active_bad.append((scan, nm, why))
                print("  HONG  %-22s %-26s %s" % (scan, nm, why), flush=True)
            else:
                inert_bad.append((scan, nm, why))

    print("")
    print("Ket qua: %d file lanh, %d file khong kiem duoc"
          % (n_ok, n_skip))
    print("  HONG tren duong chay that : %d" % len(active_bad))
    print("  HONG o file khong ai doc  : %d" % len(inert_bad))

    if inert_bad:
        # Tach rieng vi day la bao dong gia: lan quet dau tien bo 18May cho
        # 64 file hong ma 63 thuoc loai nay. Gop chung thi thanh "14 cay can
        # dung lai", trong khi that ra chi 1 cay.
        print("")
        print("File hong KHONG anh huong ket qua (co the xoa):")
        by_name = {}
        for _, nm, _ in inert_bad:
            by_name[nm] = by_name.get(nm, 0) + 1
        for nm in sorted(by_name):
            print("  %4d x %-28s %s"
                  % (by_name[nm], nm, INERT_REASON.get(nm, "khong co buoc nao doc")))

    if active_bad:
        print("")
        print("Cac ban quet CAN DUNG LAI:")
        for scan in sorted(set(b[0] for b in active_bad)):
            files = sorted(b[1] for b in active_bad if b[0] == scan)
            print("  %-24s %s" % (scan, ", ".join(files)))
        sys.exit(1)


# ===========================================================================
#  kiem_tra.py lech  (truoc day: diagnose_spatial_mismatch.py)
# ===========================================================================
# diagnose_spatial_mismatch.py
# ==============================
# Script chan doan DOC LAP, chi doc du lieu, khong sua file nao trong pipeline.
#
# So sanh vung khong gian (bounding box + trong tam) cua 4 thu:
#   1. pointcloud.ply       - point cloud GOC, chua co lap than cay (Buoc 2)
#   2. trunk_pointcloud.ply - point cloud SAU khi co lap than cay (Buoc 3)
#   3. Quy dao camera       - vi tri camera tu odometry.csv (camera luon o
#                             gan than cay dang quet, trong pham vi ~1-3m)
#   4. trunk_mesh_final.ply (hoac trunk_mesh_detected.ply neu co) - mesh cuoi
#
# Neu trunk_pointcloud.ply (2) nam XA quy dao camera (3) trong khi mesh (4)
# lai o GAN quy dao camera - day la bang chung truc tiep cho thay buoc co
# lap than cay (05_trunk_isolation.py) da chon NHAM cum diem (vd nhat nham
# 1 cay khac/vat the nen o gan do), khong phai loi xoay he truc.
#
# CACH DUNG
# ---------
#     python kiem_tra.py lech "D:/Backup/Thucdia-18May2026/cay_0031_1805"

def bbox_center(points: np.ndarray):
    if len(points) == 0:
        return None, None, None
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = (mn + mx) / 2
    return mn, mx, center


def report(label: str, points: np.ndarray):
    mn, mx, center = bbox_center(points)
    if mn is None:
        print(f"  {label:28s}: (khong co diem nao)")
        return None
    size = mx - mn
    print(f"  {label:28s}: {len(points):>10,} diem")
    print(f"    {'':28s}  min={mn}")
    print(f"    {'':28s}  max={mx}")
    print(f"    {'':28s}  tam={center}  kich_thuoc={size}")
    return center


def _main_lech() -> None:
    # Kiem thu vien o day, khong o cap module -- xem ghi chu dau file.
    global o3d
    try:
        import open3d as o3d
    except ImportError:
        print("Can cai open3d: pip install open3d==0.19.0")
        sys.exit(1)
    if len(sys.argv) != 2:
        print('Cach dung: python kiem_tra.py lech "duong/dan/toi/thu_muc_scan"')
        sys.exit(1)

    scan_dir = Path(sys.argv[1]).resolve()
    out_dir = scan_dir / "output"

    print("=" * 70)
    print(f"  Chan doan vung khong gian: {scan_dir}")
    print("=" * 70)
    print()

    centers = {}

    # 1. pointcloud.ply goc
    p1 = out_dir / "pointcloud.ply"
    if p1.exists():
        pcd = o3d.io.read_point_cloud(str(p1))
        pts = np.asarray(pcd.points)
        centers["pointcloud.ply (goc)"] = report("pointcloud.ply (goc, chua co lap)", pts)
    else:
        print(f"  (khong tim thay {p1})")
    print()

    # 2. trunk_pointcloud.ply
    p2 = out_dir / "trunk_pointcloud.ply"
    if p2.exists():
        pcd = o3d.io.read_point_cloud(str(p2))
        pts = np.asarray(pcd.points)
        centers["trunk_pointcloud.ply"] = report("trunk_pointcloud.ply (da co lap)", pts)
    else:
        print(f"  (khong tim thay {p2})")
    print()

    # 3. Quy dao camera tu odometry.csv
    odo_path = scan_dir / "odometry.csv"
    if odo_path.exists():
        import csv
        cam_pos = []
        with open(odo_path, "r", newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                try:
                    cam_pos.append([float(row["x"]), float(row["y"]), float(row["z"])])
                except (KeyError, ValueError):
                    continue
        cam_pos = np.array(cam_pos)
        centers["Quy dao camera"] = report("Quy dao camera (odometry.csv)", cam_pos)
    else:
        print(f"  (khong tim thay {odo_path})")
    print()

    # 4. Mesh cuoi cung
    for mesh_name in ["trunk_mesh_final.ply", "trunk_mesh_detected.ply",
                       "trunk_mesh_trimmed.ply", "trunk_mesh_tsdf.ply"]:
        p4 = out_dir / mesh_name
        if p4.exists():
            mesh = o3d.io.read_triangle_mesh(str(p4))
            verts = np.asarray(mesh.vertices)
            centers[mesh_name] = report(f"{mesh_name} (mesh)", verts)
            break
    else:
        print("  (khong tim thay file mesh nao)")
    print()

    # So sanh khoang cach giua cac trong tam
    print("=" * 70)
    print("  Khoang cach giua cac trong tam (met):")
    print("=" * 70)
    names = list(centers.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c1, c2 = centers[names[i]], centers[names[j]]
            if c1 is None or c2 is None:
                continue
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            flag = "  <-- XA BAT THUONG!" if dist > 1.0 else ""
            print(f"  {names[i]:28s} <-> {names[j]:28s}: {dist:.3f} m{flag}")

    print()
    print("  Ghi chu: camera quet thuong cach than cay dang quet ~0.3-1.5m.")
    print("  Neu trong tam 'trunk_pointcloud.ply' cach 'Quy dao camera' xa hon")
    print("  ~1.5-2m trong khi trong tam mesh lai gan quy dao camera hon nhieu,")
    print("  day la bang chung 05_trunk_isolation.py co the da chon nham cum")
    print("  diem (vd nham sang cay khac/vat the nen) chu khong phai loi xoay.")
    print("=" * 70)


# ===========================================================================
#  kiem_tra.py truc  (truoc day: diagnostic_trunk_axis.py)
# ===========================================================================
# diagnostic_trunk_axis.py (SIMPLIFIED VERSION - NO MATPLOTLIB)
#
# Phân tích xem trục X, Z của thân cây được xác định có chính xác không.
# So sánh:
# - Raw pointcloud → phân tích hình dáng
# - Trunk pointcloud → xem trục đã xác định chính xác không

def load_ply(path):
    """Load PLY file"""
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return None, None
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors


def analyze_trunk_axis_detection(points_raw, points_trunk):
    """
    Phân tích cách trục X, Z được xác định
    """
    print("\n" + "="*70)
    print("TRUNK AXIS DETECTION ANALYSIS")
    print("="*70)
    
    # Raw pointcloud — phân tích hình dáng
    print(f"\n1️⃣  RAW POINTCLOUD ({len(points_raw):,} points)")
    print("-" * 70)
    
    z_raw = geometry_utils.vertical(points_raw)
    z_min_raw = z_raw.min()
    z_max_raw = z_raw.max()
    z_range = z_max_raw - z_min_raw
    
    print(f"   Z range: {z_min_raw:.3f}m → {z_max_raw:.3f}m (range={z_range:.3f}m)")
    
    # Upper 40% (mặc định trong code)
    z_threshold_40 = z_max_raw - z_range * 0.4
    upper_mask_40 = z_raw >= z_threshold_40
    upper_pts_40 = points_raw[upper_mask_40]
    
    print(f"\n   [Upper 40% - mặc định code]")
    print(f"   Z threshold: {z_threshold_40:.3f}m")
    print(f"   Points in upper 40%: {upper_mask_40.sum():,} / {len(points_raw):,}")
    
    # Tính trung vị XY theo cách code làm
    cx_40, cz_40 = geometry_utils.horizontal_center(upper_pts_40)
    print(f"   Median XZ: ({cx_40:.3f}, {cz_40:.3f})")
    
    # Tính mean XY để so sánh
    xz_mean_40 = geometry_utils.horizontal(upper_pts_40)
    cx_mean_40 = np.mean(xz_mean_40[:, 0])
    cz_mean_40 = np.mean(xz_mean_40[:, 1])
    print(f"   Mean XZ:   ({cx_mean_40:.3f}, {cz_mean_40:.3f})")
    
    # Test các % khác
    print(f"\n   Testing different upper fractions:")
    for upper_frac in [0.3, 0.5, 0.6]:
        z_thresh = z_max_raw - z_range * upper_frac
        upper_mask = z_raw >= z_thresh
        upper_pts = points_raw[upper_mask]
        if len(upper_pts) > 0:
            cx, cz = geometry_utils.horizontal_center(upper_pts)
            print(f"\n   [Upper {upper_frac*100:.0f}%]")
            print(f"   Points: {upper_mask.sum():,}")
            print(f"   Median XZ: ({cx:.3f}, {cz:.3f})")
    
    # So sánh với trunk pointcloud
    print(f"\n2️⃣  TRUNK POINTCLOUD ({len(points_trunk):,} points)")
    print("-" * 70)
    
    # Xem thân cây đã được cắt gọt ở đâu XY
    x_trunk = points_trunk[:, 0]
    z_trunk = points_trunk[:, 2]
    
    print(f"   X range: {x_trunk.min():.3f}m → {x_trunk.max():.3f}m (range={x_trunk.max()-x_trunk.min():.3f}m)")
    print(f"   Z range: {z_trunk.min():.3f}m → {z_trunk.max():.3f}m (range={z_trunk.max()-z_trunk.min():.3f}m)")
    
    cx_trunk, cz_trunk = geometry_utils.horizontal_center(points_trunk)
    print(f"\n   Median XZ: ({cx_trunk:.3f}, {cz_trunk:.3f})")
    
    # Tính khoảng cách từ điểm đến tâm
    dist_xz = geometry_utils.horizontal_distance(points_trunk, (cx_trunk, cz_trunk))
    print(f"\n   Distance from center to trunk points (XZ):")
    print(f"   Max:   {dist_xz.max():.3f}m")
    print(f"   Mean:  {dist_xz.mean():.3f}m")
    print(f"   Std:   {dist_xz.std():.3f}m")
    print(f"   Pct95: {np.percentile(dist_xz, 95):.3f}m")
    
    # 3️⃣  So sánh trục được xác định
    print(f"\n3️⃣  AXIS COMPARISON")
    print("-" * 70)
    print(f"   Raw upper 40% median: ({cx_40:.3f}, {cz_40:.3f})")
    print(f"   Trunk median:         ({cx_trunk:.3f}, {cz_trunk:.3f})")
    
    dist_axis = np.sqrt((cx_40 - cx_trunk)**2 + (cz_40 - cz_trunk)**2)
    print(f"   Distance between axes: {dist_axis:.3f}m ({dist_axis*100:.1f}cm)")
    
    if dist_axis > 0.10:
        print(f"\n   ⚠️  CẢNH BÁO: Trục được xác định có khác nhau {dist_axis*100:.1f}cm!")
        print(f"   → Có thể trục X, Z bị xác định sai!")
        print(f"   → Kiểm tra lại tham số `upper_fraction` hoặc có cây khác lân cận")
    else:
        print(f"\n   ✓ Trục X, Z xác định hợp lý")
    
    # 4️⃣  Kiểm tra phân bố XY trong raw
    print(f"\n4️⃣  XY DISTRIBUTION IN RAW POINTCLOUD")
    print("-" * 70)
    
    # Tính distance từ từng điểm đến tâm predicted (cx_40, cz_40)
    dist_from_axis_40 = geometry_utils.horizontal_distance(points_raw, (cx_40, cz_40))
    
    print(f"   Distance from predicted axis ({cx_40:.3f}, {cz_40:.3f}):")
    print(f"   Max:    {dist_from_axis_40.max():.3f}m")
    print(f"   Mean:   {dist_from_axis_40.mean():.3f}m")
    print(f"   Std:    {dist_from_axis_40.std():.3f}m")
    print(f"   Pct50:  {np.percentile(dist_from_axis_40, 50):.3f}m")
    print(f"   Pct75:  {np.percentile(dist_from_axis_40, 75):.3f}m")
    print(f"   Pct90:  {np.percentile(dist_from_axis_40, 90):.3f}m")
    print(f"   Pct95:  {np.percentile(dist_from_axis_40, 95):.3f}m")
    
    # Phân tích: có vẻ như trục có lệch không?
    n_within_0_2 = (dist_from_axis_40 <= 0.20).sum()
    n_within_0_3 = (dist_from_axis_40 <= 0.30).sum()
    n_within_0_4 = (dist_from_axis_40 <= 0.40).sum()
    
    print(f"\n   Points within radius:")
    print(f"   0.20m: {n_within_0_2:,} ({100*n_within_0_2/len(points_raw):.1f}%)")
    print(f"   0.30m: {n_within_0_3:,} ({100*n_within_0_3/len(points_raw):.1f}%)")
    print(f"   0.40m: {n_within_0_4:,} ({100*n_within_0_4/len(points_raw):.1f}%)")
    
    # Nếu radius 0.30m chỉ chứa ~50% thì có lẽ trục bị lệch
    pct_0_3 = 100*n_within_0_3/len(points_raw)
    if pct_0_3 < 50:
        print(f"\n   ⚠️  NGHI NGỜ: Trục X, Z có vẻ bị lệch!")
        print(f"   → Chỉ {pct_0_3:.1f}% điểm trong radius 0.30m")
        print(f"   → Cây quế thường ~20cm đường kính → expected 70-85%")
        print(f"   → Khả năng: (1) trục bị lệch, (2) cây quá lớn, (3) có cây khác")
    else:
        print(f"\n   ✓ Phân bố XZ có vẻ hợp lý ({pct_0_3:.1f}% trong 0.30m)")


def _main_truc():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <scan_dir>")
        print(f"\nExample:")
        print(f"  python {sys.argv[0]} Raw_data/cay_0007_1805")
        sys.exit(1)
    
    scan_dir = sys.argv[1]
    raw_ply = os.path.join(scan_dir, "output", "pointcloud.ply")
    trunk_ply = os.path.join(scan_dir, "output", "trunk_pointcloud.ply")
    
    print(f"\n{'#'*70}")
    print(f"# TRUNK AXIS DETECTION DIAGNOSTIC")
    print(f"# Checking if X, Y trunk axis is correctly detected")
    print(f"{'#'*70}")
    
    # Load files
    points_raw, _ = load_ply(raw_ply)
    points_trunk, _ = load_ply(trunk_ply)
    
    if points_raw is None or points_trunk is None:
        print(f"\n❌ Error: Cannot load files")
        print(f"   Expected:")
        print(f"     - {raw_ply}")
        print(f"     - {trunk_ply}")
        sys.exit(1)
    
    # Analyze
    analyze_trunk_axis_detection(points_raw, points_trunk)
    
    print(f"\n{'#'*70}\n")


# ===========================================================================
#  kiem_tra.py odometry  (truoc day: diagnose_odometry_columns.py)
# ===========================================================================
# diagnose_odometry_columns.py
# ==============================
# Script chan doan DOC LAP, khong dung chung/sua doi bat ky file nao trong
# pipeline chinh. Muc dich: tim nguyen nhan tai sao trunk_pointcloud.ply va
# mesh TSDF lai nam o 2 he truc khac nhau cho mot so scan.
#
# Kiem tra 2 dieu:
#   1. In ra HEADER THAT cua odometry.csv (ten cot + thu tu that su trong
#      file) - so sanh voi gia dinh "timestamp,frame,x,y,z,qx,qy,qz,qw" ma
#      02_mesh_tsdf.py dang dung khi doc bang np.loadtxt(usecols=range(9)).
#   2. Doc CUNG 1 dong du lieu bang 2 CACH:
#      a) Theo TEN cot (giong 02_pointcloud_and_mesh.py dung csv.DictReader)
#      b) Theo VI TRI cot co dinh 0-8 (giong 02_mesh_tsdf.py dung np.loadtxt)
#      Neu 2 ket qua quaternion/position KHAC NHAU -> xac nhan dung nguyen
#      nhan: thu tu cot thuc te khong khop voi gia dinh vi tri co dinh.
#
# CACH DUNG
# ---------
#     python kiem_tra.py odometry "D:/Backup/Thucdia-18May2026/cay_0031_1805"

def _main_odometry() -> None:
    if len(sys.argv) != 2:
        print('Cach dung: python kiem_tra.py odometry "duong/dan/toi/thu_muc_scan"')
        sys.exit(1)

    scan_dir = Path(sys.argv[1]).resolve()
    odo_path = scan_dir / "odometry.csv"
    if not odo_path.exists():
        print(f"Khong tim thay: {odo_path}")
        sys.exit(1)

    print("=" * 70)
    print(f"  Kiem tra: {odo_path}")
    print("=" * 70)

    # ---- 1. In header that ----------------------------------------------
    with open(odo_path, "r", newline="") as f:
        raw_header_line = f.readline().rstrip("\n").rstrip("\r")
    raw_fields = raw_header_line.split(",")
    print(f"\n  Header THAT trong file (theo dung thu tu, {len(raw_fields)} cot):")
    for i, name in enumerate(raw_fields):
        print(f"    Cot {i:2d}: {name!r}")

    assumed = ["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw"]
    print(f"\n  Thu tu 02_mesh_tsdf.py GIA DINH (khong kiem tra lai):")
    for i, name in enumerate(assumed):
        print(f"    Cot {i:2d}: {name!r}")

    # So sanh ten cot (bo khoang trang, khong phan biet hoa/thuong) giua
    # header that va gia dinh, tung vi tri mot.
    print(f"\n  So sanh tung vi tri (header that vs gia dinh):")
    mismatch_found = False
    for i in range(min(len(raw_fields), len(assumed))):
        real_clean = raw_fields[i].strip().lower()
        assumed_clean = assumed[i].strip().lower()
        match = "OK" if real_clean == assumed_clean else "!! KHAC NHAU !!"
        if real_clean != assumed_clean:
            mismatch_found = True
        print(f"    Cot {i:2d}: that={raw_fields[i]!r:15s} gia_dinh={assumed[i]!r:12s}  {match}")

    if mismatch_found:
        print("\n  ==> PHAT HIEN LECH THU TU COT. Day rat co the la nguyen nhan")
        print("      khien 02_mesh_tsdf.py doc nham gia tri quaternion/position,")
        print("      dan den mesh TSDF bi xoay/lech so voi trunk_pointcloud.ply.")
    else:
        print("\n  ==> Thu tu cot khop voi gia dinh. Van de co the o cho khac,")
        print("      xem tiep phan so sanh pose cu the ben duoi.")

    # ---- 2. So sanh pose cu the cho vai dong dau ------------------------
    print(f"\n{'='*70}")
    print("  So sanh POSE cua vai dong dau, doc theo TEN cot vs VI TRI cot")
    print(f"{'='*70}")

    with open(odo_path, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        rows_by_name = []
        for i, row in enumerate(reader):
            if i >= 3:
                break
            try:
                rows_by_name.append({
                    "frame": row.get("frame", row.get(" frame")),
                    "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]),
                    "qx": float(row["qx"]), "qy": float(row["qy"]),
                    "qz": float(row["qz"]), "qw": float(row["qw"]),
                })
            except KeyError as e:
                print(f"  LOI doc theo ten cot: thieu cot {e}. "
                      "Header co the dung ten khac (vd co khoang trang).")
                rows_by_name.append(None)

    raw_data = np.loadtxt(str(odo_path), delimiter=",", skiprows=1,
                          usecols=range(9), max_rows=3)

    for i in range(min(3, len(raw_data))):
        print(f"\n  --- Dong du lieu {i} ---")
        by_name = rows_by_name[i] if i < len(rows_by_name) else None
        if by_name:
            print(f"    [Theo TEN cot]  frame={by_name['frame']}  "
                  f"pos=({by_name['x']:.4f}, {by_name['y']:.4f}, {by_name['z']:.4f})  "
                  f"quat=({by_name['qx']:.4f}, {by_name['qy']:.4f}, "
                  f"{by_name['qz']:.4f}, {by_name['qw']:.4f})")
        row_pos = raw_data[i]
        print(f"    [Theo VI TRI]   frame={int(row_pos[1])}  "
              f"pos=({row_pos[2]:.4f}, {row_pos[3]:.4f}, {row_pos[4]:.4f})  "
              f"quat=({row_pos[5]:.4f}, {row_pos[6]:.4f}, "
              f"{row_pos[7]:.4f}, {row_pos[8]:.4f})")

        if by_name:
            pos_diff = abs(by_name['x'] - row_pos[2]) + abs(by_name['y'] - row_pos[3]) + abs(by_name['z'] - row_pos[4])
            quat_diff = (abs(by_name['qx'] - row_pos[5]) + abs(by_name['qy'] - row_pos[6])
                        + abs(by_name['qz'] - row_pos[7]) + abs(by_name['qw'] - row_pos[8]))
            if pos_diff > 1e-4 or quat_diff > 1e-4:
                print(f"    ==> LECH! pos_diff={pos_diff:.6f}  quat_diff={quat_diff:.6f}")
            else:
                print(f"    ==> Khop nhau (sai so nho).")

    print(f"\n{'='*70}")
    print("  XONG. Gui toan bo output nay cho tro ly de phan tich.")
    print(f"{'='*70}")


# =========================================================================== #
#  Dieu phoi
# =========================================================================== #
COMMANDS = {
    "anh": _main_anh,
    "pointcloud": _main_pointcloud,
    "ply": _main_ply,
    "lech": _main_lech,
    "truc": _main_truc,
    "odometry": _main_odometry,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 else 1)
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print("Lenh la: %r\n" % cmd)
        print(__doc__)
        sys.exit(2)
    # Moi cong cu cu tu doc sys.argv; dat lai de no thay dung tham so cua
    # minh, voi ten chuong trinh hien ro lenh con trong thong bao --help.
    sys.argv = ["kiem_tra.py %s" % cmd] + sys.argv[2:]
    COMMANDS[cmd]()


if __name__ == "__main__":
    main()
