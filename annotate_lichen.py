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
    _score(sd, pts, lab)


def _score(sd, pts, lab):
    """Cham diem bo do hien tai tren cac diem DA CO NHAN.

    Chi cham tren diem lab >= 0 -- tuc co it nhat mot khung da khoanh nhin
    thay no. Diem lab == -1 khong khung nao nhin thay, khong biet dung sai,
    nen khong duoc tinh vao.
    """
    import contextlib
    import importlib.util
    import io as _io
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "det", os.path.join(here, "05_detect_lichen.py"))
    det = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(det)
        cols = np.asarray(o3d.io.read_point_cloud(
            os.path.join(sd, "output", "trunk_pointcloud.ply")).colors)
        with contextlib.redirect_stdout(_io.StringIO()):
            auto = det.classify_lichen_local_contrast(pts, cols) == 2
    except Exception as e:
        print(f"  (khong cham diem duoc: {e})")
        return

    m = lab >= 0
    t = lab[m] == 1
    a = auto[m]
    tp = int((a & t).sum()); fp = int((a & ~t).sum())
    fn = int((~a & t).sum()); tn = int((~a & ~t).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)

    print("")
    print("  " + "=" * 58)
    print(f"  CHAM DIEM BO DO HIEN TAI tren {int(m.sum()):,} diem co nhan")
    print("  " + "=" * 58)
    print(f"    {'':20s} {'anh khoanh DIA Y':>18s} {'anh khoanh VO':>15s}")
    print(f"    {'may bao dia y':20s} {tp:>18,} {fp:>15,}")
    print(f"    {'may bao vo':20s} {fn:>18,} {tn:>15,}")
    print("")
    print(f"    BO SOT : {fn:,}/{tp + fn:,} dia y that "
          f"({100.0 * fn / max(tp + fn, 1):.1f}%)")
    print(f"    NHAM   : {fp:,}/{tp + fp:,} cho may bao la dia y "
          f"({100.0 * fp / max(tp + fp, 1):.1f}%)")
    print(f"    do chinh xac {100 * prec:.1f}%   do bao phu {100 * rec:.1f}%   "
          f"F1 {100 * 2 * prec * rec / max(prec + rec, 1e-9):.1f}%")
    print("")
    print(f"    ty le dia y THAT (anh khoanh) : {100.0 * t.mean():.1f}%")
    print(f"    ty le may dang bao            : {100.0 * a.mean():.1f}%")


# --------------------------------------------------------------------------- #
#  fit -- chinh bo do bang nhan tay, roi cham tren vung chua dung de chinh
# --------------------------------------------------------------------------- #

FEATURES = ["L", "a", "b", "sL", "rL", "ra", "rb", "lcL", "lca",
            "nL300", "na300", "nL1000", "na1000"]

# Bo dac trung mac dinh. Chon bang do, khong bang cam tinh -- xem CLAUDE.md
# muc 12 item 20. Hai diem dang chu y:
#
#   1. KHONG co dac trung nao theo DO SANG tho. Bo "rL" ra khoi bo nay lam F1
#      tren cay la tang tu 37.9% len 41.3%: do sang mang thong tin anh sang
#      nhieu hon thong tin dia y, va do la dung cai bay da giet bo do cu.
#      Rieng "ra","rb" (du mau, khong phai do sang) da cho 39.1%, cao hon ca
#      bo 9 dac trung cu (35.1%).
#   2. Tuong phan CUNG HUONG PHAP TUYEN o hai be rong khac nhau. Xem
#      _normal_contrast() ve ly do no khac tuong phan 3D thong thuong.
#
# Ca vung quanh bo nay deu do duoc 39-42%, tuc la mot cao nguyen chu khong
# phai mot dinh nhon -- nen lua chon chinh xac trong vung do la khong quan
# trong. Dung doi theo cam tinh; do lai bang 'cross' truoc.
FEATURES_ILLUM = ["ra", "rb", "nL300", "na300", "nL1000", "na1000"]
MIN_OBS = 5           # duoi so khung nay thi mau chua du tin


def collect_colors(sd, pts, nrm, step=2, cache=None, verbose=True):
    """Gop L,a,b cho tung diem tu ANH GOC, chi tinh khung nhin CHINH DIEN.

    Khac voi mau trong trunk_pointcloud.ply: mau do da gop qua moi goc nhin,
    ke ca goc liec va mat sau, nen tin hieu dia y bi hoa tan. Loc chinh dien
    dua kha nang tach tu 0.32 len 0.78-0.99 (do tren cay_0004).
    """
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if len(z["cnt"]) == len(pts):
            if verbose:
                print("  dung lai mau da gop: " + cache)
            return z["mean"], z["std"], z["cnt"]

    K = np.loadtxt(os.path.join(sd, "camera_matrix.csv"), delimiter=",")
    poses = load_poses(sd)
    fids = [f for f in sorted(poses)
            if os.path.exists(os.path.join(sd, "rgb", f + ".jpg"))][::step]
    n = len(pts)
    s1 = np.zeros((n, 3)); s2 = np.zeros((n, 3)); cnt = np.zeros(n)
    used = blur = 0
    for i, fid in enumerate(fids):
        img = cv2.imread(os.path.join(sd, "rgb", fid + ".jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        pos, q = poses[fid]
        vs, u, v = front_visible(pts, nrm, pos, q, K, w, h)
        if len(vs) < 100:
            continue
        uu = u[vs].astype(int); vv = v[vs].astype(int)
        x0, x1 = uu.min(), uu.max() + 1
        y0, y1 = vv.min(), vv.max() + 1
        crop = img[y0:y1, x0:x1]
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if g.size < 5000 or cv2.Laplacian(g, cv2.CV_64F).var() < MIN_SHARP:
            blur += 1
            continue
        lb = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float64)
        pu = uu - x0; pv = vv - y0
        val = np.c_[lb[pv, pu, 0] * 100.0 / 255.0,
                    lb[pv, pu, 1] - 128.0,
                    lb[pv, pu, 2] - 128.0]
        s1[vs] += val; s2[vs] += val ** 2; cnt[vs] += 1
        used += 1
        if verbose and i % 200 == 0:
            print("   %d/%d" % (step * i, step * len(fids)), flush=True)
    if verbose:
        print("  dung %d khung, bo %d khung nhoe" % (used, blur))
    c = np.maximum(cnt, 1)[:, None]
    keep = (cnt >= MIN_OBS)[:, None]
    mean = np.where(keep, s1 / c, np.nan)
    std = np.where(keep, np.sqrt(np.maximum(s2 / c - (s1 / c) ** 2, 0.0)), np.nan)
    if cache:
        # Cay chua bao gio qua 'pick' thi chua co thu muc nay. Truoc khi sua,
        # do mot cay chua to lam ca 645 khung roi moi sap o dong nay.
        d = os.path.dirname(cache)
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez(cache, mean=mean, std=std, cnt=cnt)
    return mean, std, cnt


def _by_normal(val, nrm, n=16):
    """Tru trung vi cua cac diem CUNG HUONG PHAP TUYEN.

    Anh sang co huong tren mat tru lam mau bien thien theo goc quanh than (mot
    hinh sin giai thich 95%). Tru theo huong khu duoc no ma khong xoa mang dia
    y -- vi mang trai tren nhieu do cao khac nhau o cung mot huong. Tru theo
    DO CAO thi nguoc lai se xoa mat mang: da thu, kha nang tach roi 0.32 -> 0.09.
    """
    ang = np.degrees(np.arctan2(nrm[:, 2], nrm[:, 0])) % 360
    s = np.clip((ang / (360.0 / n)).astype(int), 0, n - 1)
    out = np.full(len(val), np.nan)
    for i in range(n):
        m = (s == i) & ~np.isnan(val)
        if m.sum() >= 30:
            out[m] = val[m] - np.median(val[m])
    return out


def _local_contrast(pts, val, k=40):
    """Gia tri cua diem tru trung vi cua k diem gan nhat trong khong gian 3D."""
    from scipy.spatial import cKDTree
    _, nb = cKDTree(pts).query(pts, k=k)
    out = np.full(len(pts), np.nan)
    ok = ~np.isnan(val)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.where(ok[nb], val[nb], np.nan), axis=1)
    out[ok] = val[ok] - med[ok]
    return out


def _normal_contrast(pts, nrm, val, k=300, dot_min=0.90, chunk=4000):
    """Tru trung vi cua lan can CUNG HUONG PHAP TUYEN, trong ban kinh lon.

    Tuong phan 3D thong thuong (_local_contrast) so mot diem voi k lang gieng
    gan nhat bat ke huong. Than cay chi rong 4-12cm, nen o k lon vung lan can
    VONG QUANH than va tron lan cac mat co do chieu sang khac han nhau -- dung
    cai bay da giet bo do cu. Loc theo phap tuyen giu lai nhung diem cung mat,
    gan nhu chi khac ve DO CAO, nen hieu so con lai la dia y chu khong phai
    nang.

    Do duoc tren 3 cay: them dai luong nay o hai be rong (k=300 va k=1000)
    dua F1 tren cay CHUA TUNG THAY tu 39.1% len 41.3%.

    Tinh theo tung khuc vi k=1000 tren 56k diem se can ~1.8 GB neu lam mot
    lan -- may nay chi co 8 GB.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    out = np.full(len(pts), np.nan)
    for i in range(0, len(pts), chunk):
        j = min(i + chunk, len(pts))
        _, nb = tree.query(pts[i:j], k=min(k, len(pts)), workers=-1)
        same = np.einsum("nkj,nj->nk", nrm[nb], nrm[i:j]) >= dot_min
        v = np.where(same, val[nb], np.nan)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(v, axis=1)
        n_ok = (~np.isnan(v)).sum(axis=1)
        ok = (n_ok >= 10) & ~np.isnan(val[i:j])
        out[i:j] = np.where(ok, val[i:j] - med, np.nan)
    return out


def build_features(pts, nrm, mean, std):
    L, A, B = mean[:, 0], mean[:, 1], mean[:, 2]
    F = {"L": L, "a": A, "b": B, "sL": std[:, 0],
         "rL": _by_normal(L, nrm), "ra": _by_normal(A, nrm),
         "rb": _by_normal(B, nrm),
         "lcL": _local_contrast(pts, L), "lca": _local_contrast(pts, A)}
    for k in (300, 1000):
        F["nL%d" % k] = _normal_contrast(pts, nrm, L, k=k)
        F["na%d" % k] = _normal_contrast(pts, nrm, A, k=k)
    return F


def _logistic(X, y, iters=400, lr=2.0, l2=1e-3):
    """Hoi quy logistic viet tay, co can bang lop.

    Co tinh de don gian: vai nghin diem nhan tay cua MOT cay khong du de nuoi
    mot mo hinh lon -- no se hoc thuoc long cay nay.
    """
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    pos = max(y.mean(), 1e-6)
    sw = np.where(y, 1.0 / pos, 1.0 / max(1 - pos, 1e-6))
    sw /= sw.mean()
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z.dot(w)))
        w -= lr * (Z.T.dot(sw * (p - y)) / len(Z) + l2 * np.r_[w[:-1], 0.0])
    return w, mu, sd


def _apply(X, w, mu, sd):
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    return 1.0 / (1.0 + np.exp(-Z.dot(w)))


def _best_threshold(score, y):
    best = (0.0, float(np.nanmedian(score)))
    for t in np.nanpercentile(score, np.arange(1, 100)):
        p = score >= t
        tp = (p & y).sum(); fp = (p & ~y).sum(); fn = (~p & y).sum()
        f1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1], best[0]


def _pr(pred, truth, tag):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print("    %-16s chinh xac %5.1f%%  bao phu %5.1f%%  F1 %5.1f%%"
          "   (dung %s nham %s sot %s)"
          % (tag, 100 * prec, 100 * rec, 100 * f1,
             format(tp, ","), format(fp, ","), format(fn, ",")))
    return f1


def _current_detector(sd, pts):
    """Cham diem bo do dang chay, tren cung tap diem."""
    import contextlib
    import importlib.util
    import io as _io
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        if here not in sys.path:
            sys.path.insert(0, here)
        spec = importlib.util.spec_from_file_location(
            "det", os.path.join(here, "05_detect_lichen.py"))
        det = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(det)
        cols = np.asarray(o3d.io.read_point_cloud(
            os.path.join(sd, "output", "trunk_pointcloud.ply")).colors)
        with contextlib.redirect_stdout(_io.StringIO()):
            return det.classify_lichen_local_contrast(pts, cols) == 2
    except Exception as e:
        print("    (khong cham duoc bo do hien tai: %s)" % e)
        return None


def _shade_report(L, sec, te, y, pred):
    """Bo do moi co dang bam vao anh sang khong -- cai bay da giet bo do cu.

    Bo do hien tai bam vao mat khuat nang: tuong quan do sang <-> ty le bao la
    dia y do duoc r = -0.75. Neu bo do moi cung tuong quan manh voi do sang
    (manh hon chinh su that), no dang hoc "cho sang/toi = dia y" chu khong hoc
    dia y -- va se sai o cay co huong nang khac.
    """
    rows = []
    print("")
    print("  %5s %9s %11s %12s %7s" % ("goc", "do sang", "dia y THAT",
                                       "may MOI bao", "n"))
    print("  " + "-" * 50)
    for s in range(N_SECTOR):
        mm = te & (sec == s)
        if mm.sum() < 100:
            continue
        rows.append((L[mm].mean(), y[mm].mean(), pred[mm].mean()))
        print("  %5d %9.1f %10.1f%% %11.1f%% %7s"
              % (int(s * 360 / N_SECTOR), L[mm].mean(), 100 * y[mm].mean(),
                 100 * pred[mm].mean(), format(int(mm.sum()), ",")))
    if len(rows) < 4:
        return
    r = np.array(rows)
    c_true = np.corrcoef(r[:, 0], r[:, 1])[0, 1]
    c_pred = np.corrcoef(r[:, 0], r[:, 2])[0, 1]
    print("")
    print("    tuong quan do sang <-> dia y THAT : %+.2f" % c_true)
    print("    tuong quan do sang <-> may MOI bao: %+.2f" % c_pred)
    if abs(c_pred) > abs(c_true) + 0.15:
        print("    ^ may bam vao anh sang manh hon chinh su that."
              " Bo dac trung 'illum' bam it hon.")


def cmd_fit(scan, raw_data, feature_set, step, fresh):
    """Chinh nguong tren mot nua so o luoi, cham diem tren nua kia.

    Chia theo O LUOI kieu ban co, khong chia ngau nhien theo diem: hai diem
    canh nhau gan nhu chac chan cung nhan, chia ngau nhien se cho diem cao gia
    tao vi mo hinh da nhin thay hang xom cua moi diem dem kiem.
    """
    sd = os.path.join(raw_data, scan)
    ann = os.path.join(ANN_ROOT, scan)
    lab_path = os.path.join(ann, "labels3d.npy")
    if not os.path.exists(lab_path):
        sys.exit("Chua co nhan. Chay truoc:\n"
                 "    python annotate_lichen.py read " + scan)
    pcd = o3d.io.read_point_cloud(os.path.join(sd, "output", "trunk_pointcloud.ply"))
    pts = np.asarray(pcd.points)
    lab = np.load(lab_path)
    if len(lab) != len(pts):
        sys.exit("Nhan %s diem nhung model co %s -- trunk_pointcloud.ply da doi"
                 " sau khi khoanh. Chay lai 'read'."
                 % (format(len(lab), ","), format(len(pts), ",")))
    nrm, cell = build_grid(pts)
    cache = os.path.join(ann, "colors.npz")
    if fresh and os.path.exists(cache):
        os.remove(cache)
    mean, std, cnt = collect_colors(sd, pts, nrm, step=step, cache=cache)

    F = build_features(pts, nrm, mean, std)
    keys = FEATURES_ILLUM if feature_set == "illum" else FEATURES
    ok = np.ones(len(pts), bool)
    for k in FEATURES:
        ok &= ~np.isnan(F[k])
    band = cell // N_SECTOR
    sec = cell % N_SECTOR
    m = (lab >= 0) & ok
    y = lab == 1
    tr = m & (((band + sec) % 2) == 0)
    te = m & (((band + sec) % 2) == 1)
    print("")
    print("  o CHAN, dung de CHINH : %s diem, %.1f%% dia y"
          % (format(int(tr.sum()), ","), 100 * y[tr].mean()))
    print("  o LE,   dung de KIEM  : %s diem, %.1f%% dia y"
          % (format(int(te.sum()), ","), 100 * y[te].mean()))
    if tr.sum() < 200 or te.sum() < 200:
        sys.exit("  Qua it diem co nhan de chia doi. Khoanh them khung.")

    X = np.c_[tuple(F[k] for k in keys)]
    w, mu, sd_ = _logistic(X[tr], y[tr])
    thr, f1tr = _best_threshold(_apply(X[tr], w, mu, sd_), y[tr])
    pred = _apply(X, w, mu, sd_) >= thr

    print("")
    print("  " + "=" * 64)
    print("  KET QUA tren o LE -- chua he dung de chinh  [%s]" % feature_set)
    print("  " + "=" * 64)
    _pr(pred[te], y[te], "bo do MOI")
    auto = _current_detector(sd, pts)
    if auto is not None:
        _pr(auto[te], y[te], "bo do HIEN TAI")
    print("    (F1 tren chinh o CHAN la %.1f%% -- lech nhieu so voi o LE"
          " nghia la hoc thuoc)" % (100 * f1tr))

    _shade_report(F["L"], sec, te, y, pred)

    out = os.path.join(ann, "model.json")
    with open(out, "w") as f:
        json.dump({"scan": scan, "feature_set": feature_set, "keys": keys,
                   "w": w.tolist(), "mu": mu.tolist(), "sd": sd_.tolist(),
                   "threshold": thr}, f, indent=2)
    print("\n  da luu " + out)
    print("  CANH BAO: mo hinh nay chinh tren MOT cay, mot huong nang. Truoc"
          " khi dung cho ca corpus")
    print("  phai khoanh them vai cay co huong nang khac roi kiem cheo giua"
          " cac cay.")



# --------------------------------------------------------------------------- #
#  apply -- to model bang mo hinh da chinh, de nhin bang mat
# --------------------------------------------------------------------------- #

# Cung bang mau voi 05_detect_lichen.py, de so sanh hai model canh nhau duoc.
VIZ_UP   = np.array([1.00, 0.50, 0.00])   # dia y phia tren doc  = CAM
VIZ_DOWN = np.array([1.00, 0.95, 0.00])   # dia y phia duoi doc  = VANG
END_ZONE_FRAC = 0.05                      # cat 5% moi dau -- mat cat gia tao


def predict(scan, raw_data, step=2, verbose=True, fit_from=None,
            feature_set="illum", model=None):
    """Chay mo hinh len TUNG DIEM cua trunk_pointcloud.ply.

    fit_from = None  -> dung model.json cua chinh cay do (da chinh tren no).
    fit_from = [...] -> chinh mo hinh MOI tren cac cay duoc ke ten, roi cham
                        len cay nay. Cay dich bi loai khoi tap chinh du co
                        duoc ke ten, nen ket qua luon la "mo hinh chua tung
                        thay cay nay" -- dung thu de nhin bang mat.
    """
    sd = os.path.join(raw_data, scan)
    ann = os.path.join(ANN_ROOT, scan)
    if model is not None:
        M = model
    elif fit_from:
        keys = FEATURES_ILLUM if feature_set == "illum" else FEATURES
        srcs = [s for s in fit_from if s != scan]
        if not srcs:
            sys.exit("Khong con cay nao de chinh sau khi loai " + scan)
        datas = []
        for s in srcs:
            if verbose:
                print("  chinh tren " + s + " ...")
            d = _load_scan(s, raw_data, step)
            if d:
                datas.append(d)
        if not datas:
            sys.exit("Khong cay nao trong danh sach co nhan.")
        (w, mu, sd_, thr), f1 = _fit_on(datas, keys)
        M = {"keys": keys, "w": list(w), "mu": list(mu), "sd": list(sd_),
             "threshold": thr,
             "fit_from": [d["scan"] for d in datas]}
        if verbose:
            print("  mo hinh cheo tu %d cay (F1 tren chinh cac cay do %.1f%%)"
                  % (len(datas), 100 * f1))
    else:
        mp = os.path.join(ann, "model.json")
        if not os.path.exists(mp):
            sys.exit("Chua co mo hinh. Chay truoc:\n"
                     "    python annotate_lichen.py fit " + scan)
        with open(mp) as f:
            M = json.load(f)
    pts = np.asarray(o3d.io.read_point_cloud(
        os.path.join(sd, "output", "trunk_pointcloud.ply")).points)
    nrm, cell = build_grid(pts)
    mean, std, cnt = collect_colors(sd, pts, nrm, step=step,
                                    cache=os.path.join(ann, "colors.npz"),
                                    verbose=verbose)
    F = build_features(pts, nrm, mean, std)
    X = np.c_[tuple(F[k] for k in M["keys"])]
    ok = ~np.isnan(X).any(1)
    p = np.zeros(len(pts))
    p[ok] = _apply(X[ok], np.array(M["w"]), np.array(M["mu"]), np.array(M["sd"]))
    return pts, (p >= M["threshold"]) & ok, ok, M


def _transfer(pts, lich, verts, k=15, min_frac=0.5):
    """Chuyen nhan tu diem sang dinh mesh bang DA SO phieu, khong phai 1 phieu.

    Buoc 8 cua pipeline nhan mot dinh la dia y khi CHI CAN 1 trong k diem gan
    nhat la dia y. Do duoc (CLAUDE.md muc 12 item 16): cach do thoi ty le bao
    cao len 4-6 lan so voi ty le that trong dam may diem, va con ty le thuan
    voi do min cua mesh -- cung mot cay doc 2.50% o depth 7 va 4.78% o depth 9.
    Da so phieu khong co tinh chat do.
    """
    from scipy.spatial import cKDTree
    _, idx = cKDTree(pts).query(verts, k=k, workers=-1)
    frac = lich[idx].mean(axis=1)
    return frac >= min_frac, frac


def cmd_apply(scan, raw_data, step, out_name, fit_from=None,
              feature_set="illum", model=None, quiet=False, stats=None):
    sd = os.path.join(raw_data, scan)
    out_dir = os.path.join(sd, "output")
    mesh_path = os.path.join(out_dir, "trunk_mesh_final.ply")
    if not os.path.exists(mesh_path):
        sys.exit("Khong thay " + mesh_path)

    pts, lich, ok, M = predict(scan, raw_data, step=step, fit_from=fit_from,
                               feature_set=feature_set, model=model)
    if M.get("fit_from") and not quiet:
        print("")
        print("  MO HINH CHEO -- chinh tren %s, chua tung thay %s"
              % (", ".join(M["fit_from"]), scan))
    if quiet:
        _P = lambda *a, **k: None
    else:
        _P = print
    _P("")
    _P("  tren DAM MAY DIEM (khong phu thuoc do min cua mesh)")
    _P("    diem du mau de ket luan : %s / %s"
          % (format(int(ok.sum()), ","), format(len(pts), ",")))
    _P("    ty le dia y             : %.1f%%" % (100.0 * lich[ok].mean()))

    lab_path = os.path.join(ANN_ROOT, scan, "labels3d.npy")
    if os.path.exists(lab_path):
        lab = np.load(lab_path)
        if len(lab) == len(pts):
            m = (lab >= 0) & ok
            print("    ty le dia y THAT (khoanh): %.1f%%"
                  % (100.0 * (lab[m] == 1).mean()))
            _pr(lich[m], lab[m] == 1, "bo do MOI")
            auto = _current_detector(sd, pts)
            if auto is not None:
                _pr(auto[m], lab[m] == 1, "bo do HIEN TAI")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    cols = np.asarray(mesh.vertex_colors)
    if len(cols) != len(verts):
        cols = np.full((len(verts), 3), 0.5)

    vl, frac = _transfer(pts, lich, verts)
    vl_any = frac > 0
    _P("")
    _P("  tren MESH")
    _P("    da so phieu (dung o day) : %.1f%% so dinh" % (100.0 * vl.mean()))
    _P("    >=1 phieu (kieu pipeline): %.1f%% so dinh  <- thoi phong"
          % (100.0 * vl_any.mean()))

    # huong len doc, dung y het buoc 8
    trunk_axis = _pca_axis(verts)
    g_up = _gravity_up(sd)
    up_perp = g_up - np.dot(g_up, trunk_axis) * trunk_axis
    n = np.linalg.norm(up_perp)
    up_perp = np.array([1.0, 0.0, 0.0]) if n < 0.1 else up_perp / n

    fc = verts[tris].mean(axis=1)
    centroid = verts.mean(axis=0)
    fup = (fc - centroid) @ up_perp
    side = np.zeros(len(verts), np.float32)
    for c in range(3):
        np.add.at(side, tris[:, c], fup)
    is_up = side >= 0

    # cat hai dau: mat cat phang khong phai vo cay
    t = verts @ trunk_axis
    ez = END_ZONE_FRAC * max(t.max() - t.min(), 0.01)
    vl = vl & (t >= t.min() + ez) & (t <= t.max() - ez)

    new = cols.copy()
    new[vl & is_up] = VIZ_UP
    new[vl & ~is_up] = VIZ_DOWN
    mesh.vertex_colors = o3d.utility.Vector3dVector(new.clip(0, 1))
    out = os.path.join(out_dir, out_name)
    o3d.io.write_triangle_mesh(out, mesh, write_vertex_colors=True)

    nu = int((vl & is_up).sum()); nd = int((vl & ~is_up).sum())
    if stats is not None:
        tri = verts[tris]
        area = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        f_lich = vl[tris].mean(axis=1) >= 0.5
        stats.update({
            "scan": scan,
            "n_pts": int(len(pts)),
            "n_pts_used": int(ok.sum()),
            "pct_points": round(100.0 * float(lich[ok].mean()), 2),
            "pct_mesh": round(100.0 * float(vl.mean()), 2),
            "total_cm2": round(float(area.sum()) * 10000.0, 1),
            "lichen_cm2": round(float(area[f_lich].sum()) * 10000.0, 1),
            "n_up": nu, "n_down": nd,
            "fit_from": M.get("fit_from"),
        })
    _P("    tren doc %s dinh / duoi doc %s dinh  (ty le %.2f)"
          % (format(nu, ","), format(nd, ","), nu / max(nd, 1)))
    _P("")
    _P("  da luu " + out)
    _P("  CAM = dia y phia tren doc, VANG = phia duoi doc, con lai giu mau vo.")
    _P("  File nay KHONG thay trunk_mesh_detected.ply cua pipeline -- de"
          " canh nhau ma so.")



def cmd_apply_all(raw_data, step, out_name, feature_set, skip_existing,
                  fresh_summary):
    """Chay bo do len TOAN BO corpus, ghi bang ty le dia y tung cay.

    Mo hinh duoc chinh MOT LAN tren cac cay da khoanh, roi dung lai cho moi
    cay -- tru cay nao chinh no co nhan, cay do duoc chinh lai voi chinh no bi
    loai ra, de con so cua no van la "cay chua tung thay".
    """
    import time
    from pipeline_io import dump_summary, merged_results

    keys = FEATURES_ILLUM if feature_set == "illum" else FEATURES
    labelled = sorted(d for d in os.listdir(ANN_ROOT)
                      if os.path.exists(os.path.join(ANN_ROOT, d,
                                                     "labels3d.npy")))
    if not labelled:
        sys.exit("Chua cay nao co nhan. Chay 'read' truoc.")
    print("  chinh mo hinh tren %d cay da khoanh: %s"
          % (len(labelled), ", ".join(labelled)))
    datas = []
    for s in labelled:
        d = _load_scan(s, raw_data, step)
        if d:
            datas.append(d)
    if not datas:
        sys.exit("Khong doc duoc cay nao co nhan.")

    def build_model(exclude=None):
        use = [d for d in datas if d["scan"] != exclude]
        if not use:
            return None
        (w, mu, sd_, thr), f1 = _fit_on(use, keys)
        return {"keys": keys, "w": list(w), "mu": list(mu), "sd": list(sd_),
                "threshold": thr, "fit_from": [d["scan"] for d in use]}

    full = build_model()
    holdout = {d["scan"]: build_model(exclude=d["scan"]) for d in datas}

    scans = sorted(d for d in os.listdir(raw_data)
                   if d.startswith("cay_")
                   and os.path.exists(os.path.join(raw_data, d, "output",
                                                   "trunk_mesh_final.ply")))
    print("  %d cay co trunk_mesh_final.ply\n" % len(scans))

    rows, fails = [], []
    t_all = time.time()
    for i, s in enumerate(scans, 1):
        out_p = os.path.join(raw_data, s, "output", out_name)
        if skip_existing and os.path.exists(out_p):
            print("  [%d/%d] %s  -- da co, bo qua" % (i, len(scans), s))
            continue
        t0 = time.time()
        st = {}
        try:
            cmd_apply(s, raw_data, step, out_name,
                      model=holdout.get(s, full), quiet=True, stats=st)
        except Exception as e:
            print("  [%d/%d] %s  -- LOI: %s" % (i, len(scans), s, e))
            fails.append((s, str(e)))
            continue
        st["seconds"] = round(time.time() - t0, 1)
        rows.append(st)
        print("  [%d/%d] %-18s diem %5.1f%%   mesh %5.1f%%   %8s cm2  [%.0fs]"
              % (i, len(scans), s, st["pct_points"], st["pct_mesh"],
                 format(st["lichen_cm2"], ","), st["seconds"]))

    print("\n  " + "=" * 74)
    print("  %-18s %8s %8s %11s %11s %7s" %
          ("cay", "diem %", "mesh %", "dia y cm2", "tong cm2", "tren/duoi"))
    print("  " + "-" * 74)
    for r in sorted(rows, key=lambda r: -r["pct_points"]):
        print("  %-18s %7.1f%% %7.1f%% %11s %11s %7.2f"
              % (r["scan"], r["pct_points"], r["pct_mesh"],
                 format(r["lichen_cm2"], ","), format(r["total_cm2"], ","),
                 r["n_up"] / max(r["n_down"], 1)))
    if rows:
        p = np.array([r["pct_points"] for r in rows])
        print("  " + "-" * 74)
        print("  %-18s %7.1f%% (trung vi)   thap nhat %.1f%%   cao nhat %.1f%%"
              % ("%d cay" % len(rows), float(np.median(p)), p.min(), p.max()))
    if fails:
        print("\n  %d cay LOI:" % len(fails))
        for s, e in fails:
            print("    %s: %s" % (s, e))

    path = os.path.join(raw_data, "lichen_moi_summary.json")
    dump_summary(path, merged_results(path, rows, id_key="scan",
                                      fresh=fresh_summary))
    print("\n  da luu %s" % path)
    print("  Cot 'diem %%' khong phu thuoc do min cua mesh -- dung cot do khi so")
    print("  giua cac cay. Cot 'mesh %%' va 'cm2' phu thuoc do min (CLAUDE.md")
    print("  muc 12 item 16). Tong %.0f phut." % ((time.time() - t_all) / 60))


def _pca_axis(v):
    x = v - v.mean(0)
    w, V = np.linalg.eigh(np.cov(x.T))
    a = V[:, np.argmax(w)]
    return a / np.linalg.norm(a)


def _gravity_up(sd):
    """Huong len, lay tu imu.csv neu co; thieu thi dung +Y cua the gioi.

    ARKit gan truc the gioi theo trong luc, do duoc lech toi da 0.63 do tren
    ca 49 ban quet (CLAUDE.md muc 6), nen +Y la mac dinh an toan.
    """
    p = os.path.join(sd, "imu.csv")
    if os.path.exists(p):
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1, usecols=(1, 2, 3))
            g = np.median(a, axis=0)
            if np.linalg.norm(g) > 1e-6:
                return -g / np.linalg.norm(g)
        except Exception:
            pass
    return np.array([0.0, 1.0, 0.0])



# --------------------------------------------------------------------------- #
#  sun -- do huong nang tung cay, de chon cay khoanh tiep
# --------------------------------------------------------------------------- #

def sun_azimuth(sd, n_sec=16):
    """Huong sang nhat quanh than, va hinh sin do khop den dau.

    Anh sang co huong tren mat tru lam do sang bien thien theo goc quanh than
    theo mot hinh sin. Dinh cua hinh sin la huong nang. R2 cao nghia la anh
    sang that su co huong ro; R2 thap nghia la troi am hoac cay trong bong ram,
    va cay do khong dung de kiem tra tinh tong quat theo anh sang.

    Do tren mau da gop trong trunk_pointcloud.ply chu khong gop lai tu anh goc:
    gop lai tu anh goc mat ~7 phut moi cay, qua cham cho viec CHON. Mau da gop
    lam nhat bien thien nhung khong doi huong cua no -- kiem lai tren
    cay_0004_1805 cho 78 do, so voi 90 do do tu anh goc, lech duoi mot cung.
    """
    p = os.path.join(sd, "output", "trunk_pointcloud.ply")
    if not os.path.exists(p):
        return None
    pcd = o3d.io.read_point_cloud(p)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if len(pts) < 2000 or len(cols) != len(pts):
        return None

    y = pts[:, 1]
    lo, hi = y.min(), y.max()
    band = np.clip(((y - lo) / ((hi - lo) / N_BAND)).astype(int), 0, N_BAND - 1)
    cx = np.zeros(N_BAND); cz = np.zeros(N_BAND)
    for b in range(N_BAND):
        m = band == b
        if m.sum() > 20:
            cx[b], cz[b] = np.median(pts[m, 0]), np.median(pts[m, 2])
    ang = np.degrees(np.arctan2(pts[:, 2] - cz[band], pts[:, 0] - cx[band])) % 360
    sec = np.clip((ang / (360.0 / n_sec)).astype(int), 0, n_sec - 1)

    v = cols.mean(axis=1)
    mu = np.full(n_sec, np.nan)
    for s in range(n_sec):
        m = sec == s
        if m.sum() >= 50:
            mu[s] = v[m].mean()
    ok = ~np.isnan(mu)
    if ok.sum() < 10:
        return None
    th = np.radians(np.arange(n_sec) * 360.0 / n_sec)[ok]
    A = np.c_[np.cos(th), np.sin(th), np.ones(int(ok.sum()))]
    coef, _, _, _ = np.linalg.lstsq(A, mu[ok], rcond=None)
    res = mu[ok] - A.dot(coef)
    var = ((mu[ok] - mu[ok].mean()) ** 2).sum()
    return {"az": float(np.degrees(np.arctan2(coef[1], coef[0])) % 360),
            "amp": float(np.hypot(coef[0], coef[1])),
            "r2": float(1 - (res ** 2).sum() / max(var, 1e-12)),
            "n": len(pts), "h": float(hi - lo)}


def cmd_sun(raw_data, ref):
    """Bang huong nang ca corpus, sap theo goc -- chon cay cho khac nhau."""
    scans = sorted(d for d in os.listdir(raw_data)
                   if d.startswith("cay_")
                   and os.path.isdir(os.path.join(raw_data, d)))
    rows = []
    for s in scans:
        r = sun_azimuth(os.path.join(raw_data, s))
        if r:
            r["scan"] = s
            rows.append(r)
    if not rows:
        sys.exit("Khong ban quet nao co trunk_pointcloud.ply")
    rows.sort(key=lambda r: r["az"])
    ref_az = None
    for r in rows:
        if r["scan"] == ref:
            ref_az = r["az"]
    print("")
    print("  %-18s %6s %8s %6s %8s %6s %8s"
          % ("ban quet", "huong", "bien do", "R2", "diem", "cao m", "lech"))
    print("  " + "-" * 70)
    for r in rows:
        if ref_az is None:
            d = ""
        else:
            dd = (r["az"] - ref_az + 180) % 360 - 180
            d = "%+.0f" % dd
        print("  %-18s %5.0f %9.3f %6.2f %8s %6.2f %8s"
              % (r["scan"], r["az"], r["amp"], r["r2"],
                 format(r["n"], ","), r["h"], d))
    print("\n  %d ban quet do duoc." % len(rows))
    if ref_az is not None:
        print("  'lech' la goc so voi %s (%.0f do)." % (ref, ref_az))
    print("  Chon cay de khoanh tiep: lech CANG LON cang kiem duoc manh rang mo")
    print("  hinh bam vao dia y chu khong bam vao nang. Bo qua cay co R2 thap --")
    print("  anh sang o do khong co huong ro, khong kiem duoc gi.")



# --------------------------------------------------------------------------- #
#  cross -- kiem cheo giua cac cay: chinh tren cay nay, cham tren cay khac
# --------------------------------------------------------------------------- #

def _load_scan(scan, raw_data, step=2):
    """Doc mot cay da co nhan: dac trung, nhan, huong nang."""
    sd = os.path.join(raw_data, scan)
    ann = os.path.join(ANN_ROOT, scan)
    lab_path = os.path.join(ann, "labels3d.npy")
    if not os.path.exists(lab_path):
        return None
    pts = np.asarray(o3d.io.read_point_cloud(
        os.path.join(sd, "output", "trunk_pointcloud.ply")).points)
    lab = np.load(lab_path)
    if len(lab) != len(pts):
        print("  bo qua %s: nhan %s diem, model %s diem"
              % (scan, format(len(lab), ","), format(len(pts), ",")))
        return None
    nrm, cell = build_grid(pts)
    mean, std, cnt = collect_colors(sd, pts, nrm, step=step,
                                    cache=os.path.join(ann, "colors.npz"),
                                    verbose=True)
    F = build_features(pts, nrm, mean, std)
    ok = np.ones(len(pts), bool)
    for k in FEATURES:
        ok &= ~np.isnan(F[k])
    m = (lab >= 0) & ok
    sa = sun_azimuth(sd)
    return {"scan": scan, "F": F, "m": m, "y": lab == 1,
            "L": F["L"], "sec": cell % N_SECTOR,
            "az": (sa or {}).get("az", float("nan"))}


def _fit_on(datas, keys):
    X = np.concatenate([np.c_[tuple(d["F"][k][d["m"]] for k in keys)]
                        for d in datas])
    y = np.concatenate([d["y"][d["m"]] for d in datas])
    w, mu, sd = _logistic(X, y)
    thr, f1 = _best_threshold(_apply(X, w, mu, sd), y)
    return (w, mu, sd, thr), f1


def _test_on(model, d, keys):
    w, mu, sd, thr = model
    X = np.c_[tuple(d["F"][k][d["m"]] for k in keys)]
    p = _apply(X, w, mu, sd) >= thr
    t = d["y"][d["m"]]
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return (2 * prec * rec / max(prec + rec, 1e-9), prec, rec, p)


def cmd_cross(scans, raw_data, feature_set, step):
    """Chinh tren mot tap cay, cham tren cay CHUA HE THAY.

    Day la phep thu that su. Cham tren chinh cay da dung de chinh chi cho biet
    mo hinh khop du lieu den dau, khong cho biet no co dung duoc cho cay tiep
    theo khong -- ma do moi la thu ca corpus can.
    """
    keys = FEATURES_ILLUM if feature_set == "illum" else FEATURES
    if not scans:
        scans = sorted(d for d in os.listdir(ANN_ROOT)
                       if os.path.exists(os.path.join(ANN_ROOT, d, "labels3d.npy")))
    datas = []
    for s in scans:
        print("  doc %s ..." % s)
        d = _load_scan(s, raw_data, step)
        if d:
            datas.append(d)
    if len(datas) < 2:
        sys.exit("Can it nhat 2 cay da khoanh. Moi co: %d" % len(datas))

    print("")
    print("  CAC CAY DA KHOANH")
    print("  %-18s %8s %9s %8s" % ("cay", "huong", "diem nhan", "dia y"))
    print("  " + "-" * 48)
    for d in datas:
        print("  %-18s %7.0f %9s %7.1f%%"
              % (d["scan"], d["az"], format(int(d["m"].sum()), ","),
                 100 * d["y"][d["m"]].mean()))

    print("")
    print("  " + "=" * 68)
    print("  BANG CHEO -- hang = chinh tren cay nay, cot = cham tren cay kia")
    print("  " + "=" * 68)
    hdr = "  %-18s" % "chinh \\ cham"
    for d in datas:
        hdr += " %10s" % d["scan"].replace("cay_", "").replace("_1805", "")
    print(hdr)
    print("  " + "-" * (18 + 11 * len(datas)))
    for a in datas:
        model, _ = _fit_on([a], keys)
        row = "  %-18s" % a["scan"].replace("_1805", "")
        for b in datas:
            f1, _, _, _ = _test_on(model, b, keys)
            mark = "*" if a is b else " "
            row += " %9.1f%s" % (100 * f1, mark)
        print(row)
    print("  (* = cham tren chinh cay da dung de chinh -- khong phai phep thu)")

    print("")
    print("  " + "=" * 68)
    print("  BO MOT CAY RA -- chinh tren tat ca cay con lai, cham tren cay do")
    print("  " + "=" * 68)
    print("  %-18s %9s %9s %9s %11s"
          % ("cay bi bo ra", "chinh xac", "bao phu", "F1", "bo do cu"))
    print("  " + "-" * 62)
    f1s = []
    for b in datas:
        rest = [d for d in datas if d is not b]
        model, _ = _fit_on(rest, keys)
        f1, prec, rec, pred = _test_on(model, b, keys)
        f1s.append(f1)
        old_s = "  -"
        auto = _current_detector_pts(b, raw_data)
        if auto is not None:
            t = b["y"][b["m"]]; a = auto[b["m"]]
            tp = int((a & t).sum()); fp = int((a & ~t).sum()); fn = int((~a & t).sum())
            p2 = tp / max(tp + fp, 1); r2 = tp / max(tp + fn, 1)
            old_s = "%9.1f%%" % (100 * 2 * p2 * r2 / max(p2 + r2, 1e-9))
        print("  %-18s %8.1f%% %8.1f%% %8.1f%% %11s"
              % (b["scan"].replace("_1805", ""), 100 * prec, 100 * rec,
                 100 * f1, old_s))
        _shade_report(b["L"], b["sec"], b["m"], b["y"], _expand(pred, b["m"]))
    print("")
    print("  F1 trung binh khi gap cay chua tung thay: %.1f%%"
          % (100 * float(np.mean(f1s))))


def _expand(pred_on_masked, m):
    out = np.zeros(len(m), bool)
    out[m] = pred_on_masked
    return out


def _current_detector_pts(d, raw_data):
    sd = os.path.join(raw_data, d["scan"])
    pts = np.asarray(o3d.io.read_point_cloud(
        os.path.join(sd, "output", "trunk_pointcloud.ply")).points)
    return _current_detector(sd, pts)



def main():
    global MIN_SHARP
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pick", help="Chon va xuat khung de khoanh")
    a.add_argument("scan")
    a.add_argument("-n", type=int, default=12, help="So khung (mac dinh 12)")
    a.add_argument("--step", type=int, default=5, help="Xet moi N khung (mac dinh 5)")
    a.add_argument("--raw-data", default=RAW_DATA_DIR)
    a.add_argument("--min-sharp", type=float, default=MIN_SHARP,
                   help="San do net (mac dinh %g). Nang len cho ban quay bi "
                        "rung: khung nhoe xoa mat ranh gioi mang, nguoi to se "
                        "to thieu -- do duoc tren cay_0036_1805, khung to nhoe "
                        "gap 3 lan hai cay kia." % MIN_SHARP)
    b = sub.add_parser("read", help="Doc phan da khoanh, chieu len model")
    b.add_argument("scan")
    b.add_argument("--raw-data", default=RAW_DATA_DIR)
    b.add_argument("--min-votes", type=int, default=2,
                   help="So khung toi thieu cung khoanh mot diem (mac dinh 2)")
    c = sub.add_parser("fit", help="Chinh bo do theo nhan, kiem tren vung chua dung")
    c.add_argument("scan")
    c.add_argument("--raw-data", default=RAW_DATA_DIR)
    c.add_argument("--features", choices=["illum", "all"], default="illum",
                   help="illum = chi dac trung da khu anh sang (mac dinh); "
                        "all = them ca mau tho, F1 cao hon nhung bam vao nang")
    c.add_argument("--step", type=int, default=2,
                   help="Gop mau tu moi N khung (mac dinh 2)")
    c.add_argument("--fresh", action="store_true", help="Gop lai mau, bo cache")
    d = sub.add_parser("apply", help="To model bang mo hinh da chinh")
    d.add_argument("scan", nargs="?", help="Bo trong khi dung --all")
    d.add_argument("--raw-data", default=RAW_DATA_DIR)
    d.add_argument("--step", type=int, default=2)
    d.add_argument("--out", default=None,
                   help="Ten file trong output/ (mac dinh trunk_mesh_detected_MOI.ply,"
                        " hoac _CHEO.ply khi co --fit-from)")
    d.add_argument("--fit-from", nargs="*", default=None,
                   help="Chinh mo hinh tren cac cay NAY roi cham len cay dich;"
                        " bo trong sau co = moi cay da khoanh. Cay dich luon bi"
                        " loai khoi tap chinh.")
    d.add_argument("--features", choices=["illum", "all"], default="illum")
    d.add_argument("--all", action="store_true",
                   help="Chay het corpus, ghi lichen_moi_summary.json")
    d.add_argument("--skip-existing", action="store_true")
    d.add_argument("--fresh-summary", action="store_true")
    f_ = sub.add_parser("cross", help="Kiem cheo giua cac cay da khoanh")
    f_.add_argument("scans", nargs="*",
                    help="Ten cac cay; bo trong = moi cay da co labels3d.npy")
    f_.add_argument("--raw-data", default=RAW_DATA_DIR)
    f_.add_argument("--features", choices=["illum", "all"], default="illum")
    f_.add_argument("--step", type=int, default=2)
    e = sub.add_parser("sun", help="Do huong nang ca corpus, de chon cay khoanh tiep")
    e.add_argument("--raw-data", default=RAW_DATA_DIR)
    e.add_argument("--ref", default="cay_0004_1805",
                   help="Cay lay lam moc de tinh do lech (mac dinh cay_0004_1805)")
    args = p.parse_args()
    if args.cmd == "pick":
        MIN_SHARP = args.min_sharp
        cmd_pick(args.scan, args.n, args.step, args.raw_data)
    elif args.cmd == "fit":
        cmd_fit(args.scan, args.raw_data, args.features, args.step, args.fresh)
    elif args.cmd == "apply":
        if args.all:
            cmd_apply_all(args.raw_data, args.step,
                          args.out or "trunk_mesh_detected_CHEO.ply",
                          args.features, args.skip_existing,
                          args.fresh_summary)
            return
        ff = args.fit_from
        if ff is not None and len(ff) == 0:
            ff = sorted(d for d in os.listdir(ANN_ROOT)
                        if os.path.exists(os.path.join(ANN_ROOT, d,
                                                       "labels3d.npy")))
        out = args.out or ("trunk_mesh_detected_CHEO.ply" if ff
                           else "trunk_mesh_detected_MOI.ply")
        cmd_apply(args.scan, args.raw_data, args.step, out, ff, args.features)
    elif args.cmd == "sun":
        cmd_sun(args.raw_data, args.ref)
    elif args.cmd == "cross":
        cmd_cross(args.scans, args.raw_data, args.features, args.step)
    else:
        cmd_read(args.scan, args.raw_data, args.min_votes)


if __name__ == "__main__":
    main()
