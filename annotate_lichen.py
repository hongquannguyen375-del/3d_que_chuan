#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
annotate_lichen.py -- khoanh dia y bang tay tren anh 2D, chieu nguoc len model.

TAI SAO CAN
-----------
Bo do mau tu dong khong dang tin. Do duoc tren cay_0004 (CLAUDE.md muc 12
item 18): bien thien mau quanh chu vi than khop mot hinh sin giai thich 95-96%
-- do la anh sang co huong tren mat tru -- va bien do cua no tren truc a* la
1.62, gap ~10 lan chenh lech dia y/vo. Bo do hien tai bam vao MAT KHUAT NANG
cua than: tuong quan giua do sang cua mot goc va ty le no bao la dia y do duoc
r = -0.75.

Khong co nhan that thi moi nguong deu tu quyet dinh cau tra loi. Cong cu nay
tao ra nhan that.

HAI RANG BUOC KHI CHON KHUNG, CA HAI DEU DO DUOC
------------------------------------------------
1. KHONG TRUNG VUNG. Khung lien nhau nhin gan nhu cung mot cho. Do tren
   cay_0004: 12 khung NET NHAT chi phu 50/192 o luoi (26%), vi chung chum vao
   mot doan. Chon tham lam theo "o moi" cho 179/192 (93%). Rai deu theo thoi
   gian duoc 143/192 (74%).
2. UU TIEN KHUNG NET. Do net (phuong sai Laplacian trong vung than) dao dong
   20 lan giua cac khung (30 den 659). Khung nhoe xoa mat ranh gioi mang.

DUNG
----
    python annotate_lichen.py pick <scan> [-n 12]
        -> D:\\Lichen_project\\annotate\\<scan>\\  (anh + HUONGDAN.txt)

    (nguoi dung to mau hong canh sen 255,0,255 len vung dia y, luu de len)

    python annotate_lichen.py read <scan>
        -> annotate\\<scan>\\labels3d.npy   (-1 chua thay, 0 vo, 1 dia y)
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d

RAW_DATA_DIR = r"D:\Backup\Thucdia-18May2026"
ANN_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotate")

FRONT_DOT = -0.35     # phap tuyen . huong nhin: duoi muc nay la "quay mat ve camera"
N_BAND    = 16
N_SECTOR  = 12
MIN_SHARP = 60.0      # p10 do duoc tren cay_0004


# --------------------------------------------------------------------------- #
#  Hinh hoc
# --------------------------------------------------------------------------- #

def quat_to_R(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_poses(scan_dir):
    path = os.path.join(scan_dir, "odometry.csv")
    rows = np.genfromtxt(path, delimiter=",", skip_header=1,
                         usecols=range(9), invalid_raise=False)
    out = {}
    with open(path) as f:
        f.readline()
        for i, line in enumerate(f):
            fid = line.split(",")[1].strip()
            if i < len(rows) and not np.isnan(rows[i][2]):
                out[fid] = (rows[i][2:5], rows[i][5:9])
    return out


def project(pts, pos, quat, K):
    P = (pts - pos) @ quat_to_R(quat)
    z = P[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * P[:, 0] / z + K[0, 2]
        v = K[1, 1] * P[:, 1] / z + K[1, 2]
    return u, v, z


def visible(u, v, z, w, h, cell=4):
    """Giu diem gan nhat trong moi o luoi pixel -- khu mat sau cua than."""
    ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return idx
    key = (v[idx] // cell).astype(np.int64) * (w // cell + 2) + (u[idx] // cell).astype(np.int64)
    order = np.lexsort((z[idx], key))
    idx_s, key_s = idx[order], key[order]
    first = np.ones(len(key_s), bool)
    first[1:] = key_s[1:] != key_s[:-1]
    return idx_s[first]


def build_grid(pts):
    """Phap tuyen be mat (tu tam lat ra) + chi so o (dai cao x cung goc)."""
    y = pts[:, 1]
    lo, hi = y.min(), y.max()
    band = np.clip(((y - lo) / ((hi - lo) / N_BAND)).astype(int), 0, N_BAND - 1)
    cx = np.zeros(N_BAND); cz = np.zeros(N_BAND)
    for b in range(N_BAND):
        m = band == b
        if m.sum() > 20:
            cx[b], cz[b] = np.median(pts[m, 0]), np.median(pts[m, 2])
    nrm = np.c_[pts[:, 0] - cx[band], np.zeros(len(pts)), pts[:, 2] - cz[band]]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    ang = np.degrees(np.arctan2(nrm[:, 2], nrm[:, 0])) % 360
    sec = np.clip((ang / (360.0 / N_SECTOR)).astype(int), 0, N_SECTOR - 1)
    return nrm, band * N_SECTOR + sec


def front_visible(pts, nrm, pos, quat, K, w, h):
    u, v, z = project(pts, pos, quat, K)
    vs = visible(u, v, z, w, h)
    if len(vs) == 0:
        return vs, u, v
    d = pts[vs] - pos
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    return vs[(nrm[vs] * d).sum(1) < FRONT_DOT], u, v


# --------------------------------------------------------------------------- #
#  pick
# --------------------------------------------------------------------------- #

def cmd_pick(scan, n_frames, step, raw_data):
    sd = os.path.join(raw_data, scan)
    pcd = o3d.io.read_point_cloud(os.path.join(sd, "output", "trunk_pointcloud.ply"))
    pts = np.asarray(pcd.points)
    K = np.loadtxt(os.path.join(sd, "camera_matrix.csv"), delimiter=",")
    poses = load_poses(sd)
    nrm, cell = build_grid(pts)
    n_cells = len(set(cell.tolist()))

    print(f"Quet khung hinh cua {scan} (moi {step} khung 1)...")
    cands = []
    fids = [f for f in sorted(poses)
            if os.path.exists(os.path.join(sd, "rgb", f"{f}.jpg"))][::step]
    for i, fid in enumerate(fids):
        img = cv2.imread(os.path.join(sd, "rgb", f"{fid}.jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        pos, q = poses[fid]
        vs, u, v = front_visible(pts, nrm, pos, q, K, w, h)
        if len(vs) < 200:
            continue
        uu = u[vs].astype(int); vv = v[vs].astype(int)
        crop = img[vv.min():vv.max() + 1, uu.min():uu.max() + 1]
        if crop.size < 15000:
            continue
        sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if sharp < MIN_SHARP:
            continue
        cs, cnt = np.unique(cell[vs], return_counts=True)
        cands.append({"fid": fid, "sharp": sharp, "cells": set(cs[cnt >= 15].tolist())})
        if i % 200 == 0:
            print(f"   {i}/{len(fids)}", flush=True)
    print(f"   {len(cands)} khung du net va du diem\n")
    if not cands:
        sys.exit("Khong co khung nao dung duoc")

    smax = max(c["sharp"] for c in cands)
    covered, chosen = set(), []
    while len(chosen) < n_frames:
        best, best_score = None, 0.0
        for c in cands:
            if c in chosen:
                continue
            gain = len(c["cells"] - covered)
            if gain == 0:
                continue
            score = gain * (0.75 + 0.25 * c["sharp"] / smax)
            if score > best_score:
                best, best_score = c, score
        if best is None:
            break
        chosen.append(best)
        covered |= best["cells"]

    out = os.path.join(ANN_ROOT, scan)
    os.makedirs(out, exist_ok=True)
    print(f"Chon {len(chosen)} khung, phu {len(covered)}/{n_cells} o "
          f"({100.0 * len(covered) / n_cells:.0f}% be mat)\n")

    index = []
    for i, c in enumerate(chosen):
        fid = c["fid"]
        img = cv2.imread(os.path.join(sd, "rgb", f"{fid}.jpg"))
        h, w = img.shape[:2]
        pos, q = poses[fid]
        vs, u, v = front_visible(pts, nrm, pos, q, K, w, h)
        pad = 25
        x0, x1 = max(0, int(u[vs].min()) - pad), min(w, int(u[vs].max()) + pad)
        y0, y1 = max(0, int(v[vs].min()) - pad), min(h, int(v[vs].max()) + pad)
        name = f"{i + 1:02d}_{fid}"
        cv2.imwrite(os.path.join(out, f"{name}.png"), img[y0:y1, x0:x1])
        index.append({"n": i + 1, "frame": fid, "crop": [int(x0), int(y0), int(x1), int(y1)],
                      "sharp": round(float(c["sharp"]), 1), "n_pts": int(len(vs))})
        print(f"  {name}.png   {len(vs):>5,} diem than   do net {c['sharp']:>5.0f}")

    json.dump({"scan": scan, "n_cells": n_cells, "covered": len(covered),
               "frames": index}, open(os.path.join(out, "INDEX.json"), "w"), indent=2)
    _write_guide(out, len(chosen), len(covered), n_cells)
    print(f"\nDoc {os.path.join(out, 'HUONGDAN.txt')} roi bat dau khoanh.")


def _write_guide(out, n, covered, n_cells):
    with open(os.path.join(out, "HUONGDAN.txt"), "w") as f:
        f.write(f"""HUONG DAN KHOANH DIA Y BANG TAY
================================

Co {n} khung hinh. Chung duoc chon de PHU DEU than cay: nhin thay
{covered}/{n_cells} o luoi ({100.0 * covered / n_cells:.0f}% be mat). Chon theo do net
khong lam duoc viec nay -- cac khung net nhat thuong chum vao mot doan.

CACH KHOANH
-----------
1. Mo tung file .png bang Paint hoac phan mem ve bat ky.

2. To de len MOI vung dia y nhin thay tren THAN CAY.
   Dung mau HONG CANH SEN thuan:  R=255  G=0  B=255
   (Paint: Edit colors -> go 255, 0, 255)
   Chon mau nay vi anh rung that khong bao gio co no, nen may tach duoc chac
   chan dau la net ve cua anh.

3. To phu kin ca mang. Khong can chinh xac tung mm; sai vai pixel o ria
   khong anh huong.

4. CHI to phan tren than cay. Bo qua la, dat, cay khac o nen.

5. Luu de len chinh file do (Ctrl+S), GIU NGUYEN ten va duoi .png.

KHOANH TAT CA, KHONG CHI CHO MAY BO SOT
---------------------------------------
Bo nhan nay dung de day may. Neu chi khoanh phan may bo sot, may se hoc nham
rang nhung cho no da nhan ra la "khong phai dia y".

KHUNG KHONG CO DIA Y
--------------------
Cu de nguyen, khong to gi. Khung sach cung la du lieu -- no day may biet cho
nao KHONG phai dia y.

XONG THI CHAY
-------------
    python annotate_lichen.py read <ten_scan>
""")


# --------------------------------------------------------------------------- #
#  read
# --------------------------------------------------------------------------- #

def mark_mask(painted, clean):
    b = painted[:, :, 0].astype(int)
    g = painted[:, :, 1].astype(int)
    r = painted[:, :, 2].astype(int)
    m = (r > 180) & (b > 180) & (g < 90)
    if clean is not None and clean.shape == painted.shape:
        m &= np.abs(painted.astype(int) - clean.astype(int)).sum(2) > 60
    return m


def cmd_read(scan, raw_data, min_votes):
    sd = os.path.join(raw_data, scan)
    ann = os.path.join(ANN_ROOT, scan)
    meta = json.load(open(os.path.join(ann, "INDEX.json")))
    pts = np.asarray(o3d.io.read_point_cloud(
        os.path.join(sd, "output", "trunk_pointcloud.ply")).points)
    K = np.loadtxt(os.path.join(sd, "camera_matrix.csv"), delimiter=",")
    poses = load_poses(sd)
    nrm, _ = build_grid(pts)

    yes = np.zeros(len(pts), int)
    no = np.zeros(len(pts), int)
    print(f"{'khung':>14s} {'px da to':>10s} {'diem thay':>11s} {'trung vung to':>14s}")
    print("-" * 54)
    n_done = 0
    for fr in meta["frames"]:
        name = f"{fr['n']:02d}_{fr['frame']}"
        pp = os.path.join(ann, f"{name}.png")
        if not os.path.exists(pp):
            print(f"{name:>14s}   (khong thay file)")
            continue
        painted = cv2.imread(pp)
        x0, y0, x1, y1 = fr["crop"]
        full = cv2.imread(os.path.join(sd, "rgb", f"{fr['frame']}.jpg"))
        mk = mark_mask(painted, full[y0:y1, x0:x1])
        h, w = full.shape[:2]
        pos, q = poses[fr["frame"]]
        vs, u, v = front_visible(pts, nrm, pos, q, K, w, h)
        uu = (u[vs] - x0).astype(int); vv = (v[vs] - y0).astype(int)
        ok = (uu >= 0) & (uu < mk.shape[1]) & (vv >= 0) & (vv < mk.shape[0])
        vs, uu, vv = vs[ok], uu[ok], vv[ok]
        hit = mk[vv, uu]
        yes[vs[hit]] += 1
        no[vs[~hit]] += 1
        n_done += 1
        print(f"{name:>14s} {int(mk.sum()):>10,} {len(vs):>11,} {int(hit.sum()):>14,}")

    if n_done == 0:
        sys.exit("Chua co khung nao duoc khoanh.")
    lab = np.full(len(pts), -1, np.int8)
    seen = (yes + no) > 0
    lab[seen] = 0
    lab[(yes >= min_votes) | ((yes > 0) & (yes >= no))] = 1
    print("-" * 54)
    print(f"\n  {n_done} khung da doc")
    print(f"  diem duoc it nhat 1 khung nhin thay : {int((lab >= 0).sum()):,} / {len(pts):,}")
    print(f"  diem khoanh la dia y                : {int((lab == 1).sum()):,} "
          f"({100.0 * (lab == 1).sum() / max((lab >= 0).sum(), 1):.1f}%)")
    np.save(os.path.join(ann, "labels3d.npy"), lab)
    print(f"\n  da luu {os.path.join(ann, 'labels3d.npy')}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pick", help="Chon va xuat khung de khoanh")
    a.add_argument("scan")
    a.add_argument("-n", type=int, default=12, help="So khung (mac dinh 12)")
    a.add_argument("--step", type=int, default=5, help="Xet moi N khung (mac dinh 5)")
    a.add_argument("--raw-data", default=RAW_DATA_DIR)
    b = sub.add_parser("read", help="Doc phan da khoanh, chieu len model")
    b.add_argument("scan")
    b.add_argument("--raw-data", default=RAW_DATA_DIR)
    b.add_argument("--min-votes", type=int, default=2,
                   help="So khung toi thieu cung khoanh mot diem (mac dinh 2)")
    args = p.parse_args()
    if args.cmd == "pick":
        cmd_pick(args.scan, args.n, args.step, args.raw_data)
    else:
        cmd_read(args.scan, args.raw_data, args.min_votes)


if __name__ == "__main__":
    main()
