#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_mesh_poisson_trunk.py
========================
Dung mesh than cay bang Poisson Surface Reconstruction TU trunk_pointcloud.ply
(dau ra cua 05_trunk_isolation.py), thay cho mesh TSDF o buoc 4.

Input : <scan>/output/trunk_pointcloud.ply
Output: <scan>/output/trunk_mesh_poisson.ply

TAI SAO CAN BUOC NAY (xem CLAUDE.md muc 12 item 15)
---------------------------------------------------
trunk_mesh_tsdf.ply co nhung lo THUNG LON tren dung phan than cay DA DUOC
QUET. Nguyen nhan da duoc do va khoanh vung: TSDF cua Open3D do khoang cach
DOC THEO TIA CAMERA. O goc nhin xien theta, khoang cach doc tia bi keo dai
1/cos(theta) -- 60 do la gap doi, 80 do gap 5.8 lan -- nen voxel nam sat be
mat lai bi tinh la xa, roi ra ngoai dai sdf_trunc va KHONG BAO GIO duoc tich
hop. Khong tich luy trong so thi khong co mat dang tri.

Do duoc: goc nhin tot nhat ma bat ky camera nao dat duoc tren phan be mat bi
mat la 52-58 do (p50), so voi 11-22 do tren phan con nguyen. Va so lo tuong
quan -0.93 voi do manh cua than tren cay_0004 (tren r_p90 6.0cm khong co lo
nao; duoi 5.8cm thi tang deu), vi than cang manh thi ty le be mat nam o goc
xien cang lon.

DA THU VA LOAI TRU 11 CAU HINH TSDF: sdf_trunc 0.010/0.015/0.025/0.05/0.08/
0.12/0.16, voxel 0.005 va 0.008, conf_thresh 0 va 1, cong them mot bo loc
"diem bay" o mep do sau. Trong pham vi than cay so vong bien gan nhu khong
doi (29-32 vong, 5-9 vong >=30 dinh) trong khi ban kinh mesh phinh tu 1.28x
len 2.44x. Bo loc mep con lam mat 11-24% be mat ma khong giam lo. Tuc cac lo
nay do DU LIEU quyet dinh, khong phai do tham so.

Poisson khong gap van de do vi no KHOP MOT MAT VAO DAM DIEM thay vi hoa doc
theo tia camera, nen dung duoc chinh nhung diem ma TSDF phai bo. Diem quan
trong: du lieu diem trong cac lo VAN DAY DU va khong he te hon noi khac (do
duoc: lop vo diem day 2.7cm trong lo so voi 2.6-5.6cm vung lanh; mat do
33 diem/do so voi 20 diem/do).

DO TREN CA 46 SCAN, Poisson bam diem tot hon TSDF o MOI cay:
  - 95% dinh mesh nam trong 0.7-0.9cm cua mot diem that (TSDF: 2.6-44cm)
  - phinh ban kinh 0.95-1.07x (TSDF: 0.95-15.3x)

CHON --depth BANG SO DO, KHONG PHAI BANG MAT
--------------------------------------------
Moc doi chieu la dien tich hinh tru suy ra tu chinh point cloud
(2*pi*r_med*h). Day la so quan trong vi 05_detect_lichen.py tinh total_cm2 tu
dien tich TAM GIAC cua mesh. Do tren 7 scan dai dien, dien tich sau khi lam
muot Taubin (dung tham so ma 04b_finalize_mesh.py van dung) chia cho moc tru:

    depth   lo lon nhat   phinh ban kinh   dien tich / tru
      6        8 -  36      1.02-1.04         0.92-1.27      (mesh qua thua)
      7       18 -  75      0.98-1.12         0.92-1.52      <-- CHON
      8       42 - 216      1.01-1.13         1.04-1.91
      9      136 - 651      1.00-1.07         1.16-3.10      (bam nhieu)
    TSDF     374 -1383      0.95-1.62         0.35-1.25      (thieu be mat)

Depth cao bam theo nhieu cua lop vo diem day 2.7-5.6cm va tao mat nhan nhum,
lam PHONG dien tich; depth 6 thi mesh qua thua (co cay duoi 1,000 dinh).
Luu y: chi so "khoang cach toi diem gan nhat" DANH LUA o day -- depth 9 dat
0.7cm chinh vi no bam vao nhieu; moc hinh tru moi la thuoc do dang tin.

Canh bao: TSDF hien dang do THIEU dien tich rat nhieu tren mot so cay
(cay_0009 chi bang 0.35x moc tru, tuc mat 65% be mat; cay_0013 0.57x;
cay_0010 0.75x), nen total_cm2 trong lichen_stats.json hien co DA SAI SAN,
khong phai se sai sau khi doi.

PHAP TUYEN
----------
Poisson can phap tuyen co huong nhat quan. Thay vi lan truyen theo mat phang
tiep tuyen (khong dang tin tren lop vo diem day ~3cm), o day phap tuyen duoc
lat ra NGOAI so voi truc cua CHINH LAT CAT do -- huong "ra ngoai" dung ve mat
vat ly cho mot than cay.

CACH DUNG
---------
    python 02_mesh_poisson_trunk.py --scan cay_0007_1805
    python 02_mesh_poisson_trunk.py --all
    python 02_mesh_poisson_trunk.py --all --depth 8 --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import open3d as o3d

RAW_DATA_DIR = r"D:\Backup\Thucdia-18May2026"

DEFAULT_DEPTH = 7
N_NORMAL_BANDS = 40
DENSITY_QUANTILE = 0.05

# --- Ring trim (cat lat khong khep kin) -- xem docstring patch_poisson_ring_trim.py
RING_BINS        = 36     # bin 10 do quanh chu vi
RING_MAX_GAP_DEG = 20.0   # khe goc trong lon nhat cho phep trong mot lat
RING_SLICE_M     = 0.06   # do day lat (m) -- do tren 46 cay, xem docstring
RING_BRIDGE      = 1      # so lat hong LIEN TIEP duoc phep bac cau
RING_MIN_PTS     = 10     # lat it hon nguong nay coi nhu hong
RING_WARN_FRAC   = 0.70   # giu duoi muc nay -> canh bao + co trong JSON

# --- Cat phang hai dau -- xem docstring patch_poisson_flat_rims.py
RIM_FLAT_TOL  = 0.015   # vanh trai dai qua muc nay (m) thi coi la rang cua
RIM_MAX_ITER  = 6       # cat lap; thuc te hoi tu sau 2-3 lan


# ---------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------

def orient_normals_outward(pcd: o3d.geometry.PointCloud,
                           n_bands: int = N_NORMAL_BANDS) -> o3d.geometry.PointCloud:
    """
    Uoc tinh phap tuyen roi lat chung ra NGOAI so voi truc cua tung lat cat.

    Khong dung orient_normals_consistent_tangent_plane(): lop vo diem cua scan
    that day toi 2.7-5.6cm, nen lan truyen huong theo mat phang tiep tuyen rat
    de bi lat nham va tao ra mat Poisson lon xon. Truc than cay thi luon biet
    duoc, va "ra xa truc" chinh la huong ngoai that su.
    """
    pts = np.asarray(pcd.points)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))

    y = pts[:, 1]
    edges = np.linspace(y.min(), y.max(), n_bands + 1)
    band = np.clip(np.searchsorted(edges, y) - 1, 0, n_bands - 1)

    normals = np.asarray(pcd.normals).copy()
    for i in range(n_bands):
        k = band == i
        if k.sum() < 30:
            continue
        cx = float(np.median(pts[k, 0]))
        cz = float(np.median(pts[k, 2]))
        radial = np.stack(
            [pts[k, 0] - cx, np.zeros(int(k.sum())), pts[k, 2] - cz], axis=1)
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
        sub = normals[k]
        sub[np.einsum("ij,ij->i", sub, radial) < 0] *= -1
        normals[k] = sub

    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# ---------------------------------------------------------------------------
# Ring trim -- cat cac lat cat khong khep kin thanh vong tron
# ---------------------------------------------------------------------------

def slice_ring_gaps(pts: np.ndarray,
                    slice_m: float = RING_SLICE_M,
                    n_bins: int = RING_BINS,
                    min_pts: int = RING_MIN_PTS):
    """
    Do "khe goc trong lon nhat" cua tung lat cat theo chieu cao.

    Tam cua moi lat lay RIENG (median X, median Z cua chinh lat do), khong dung
    mot tam chung -- than cay nghieng va co phinh goc, nen mot tam chung se lam
    lech goc do o hai dau.

    Tra ve (edges, gaps_deg, counts). Lat khong du diem -> khe = 360 deg.
    """
    y = pts[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    n = max(4, int(round((y_max - y_min) / max(slice_m, 1e-6))))

    # Cung idiom searchsorted+clip nhu orient_normals_outward(): xu ly bin cuoi
    # dung, khac voi cac vong (y>=lo)&(y<hi) rai rac noi khac.
    edges = np.linspace(y_min, y_max, n + 1)
    band = np.clip(np.searchsorted(edges, y) - 1, 0, n - 1)

    gaps = np.full(n, 360.0)
    counts = np.zeros(n, dtype=int)
    step = 360.0 / n_bins

    for i in range(n):
        k = band == i
        c = int(k.sum())
        counts[i] = c
        if c < min_pts:
            continue
        px, pz = pts[k, 0], pts[k, 2]
        cx, cz = float(np.median(px)), float(np.median(pz))
        ang = np.arctan2(pz - cz, px - cx)
        b = ((ang + np.pi) / (2.0 * np.pi) * n_bins).astype(int) % n_bins
        occ = np.bincount(b, minlength=n_bins) > 0
        if occ.all():
            gaps[i] = 0.0
            continue
        idx = np.where(occ)[0]
        if len(idx) == 0:
            continue
        # Dai bin trong lien tiep dai nhat THEO VONG TRON
        runs = np.diff(np.r_[idx, idx[0] + n_bins]) - 1
        gaps[i] = float(runs.max() * step)

    return edges, gaps, counts


def longest_closed_run(ok: np.ndarray, bridge: int = RING_BRIDGE):
    """
    Doan lien tuc dai nhat cua cac lat DAT, cho phep bac cau qua toi da `bridge`
    lat hong LIEN TIEP.

    Ly do bac cau: do tren 46 cay, o nguong 20 deg co 28 cay bi lat hong nam
    GIUA than. Cat roi tung lat lam 25/46 cay vo thanh 2-4 manh, va cac buoc sau
    deu goi keep_largest_component() nen se am tham vut cac manh thua. Bac cau 1
    lat giu duoc 91.6% chieu cao (so voi 92.0% neu cat roi) MA VAN LIEN MOT KHOI.

    Tra ve (i0, i1) chi so lat dau/cuoi (bao gom ca hai), hoac None.
    """
    n = len(ok)
    best = None
    best_len = 0
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        last_ok = i
        while j < n:
            if ok[j]:
                last_ok = j
                j += 1
            else:
                r = j
                while r < n and not ok[r]:
                    r += 1
                if r < n and (r - j) <= bridge:
                    j = r          # bac cau qua khoang hong ngan nay
                else:
                    break
        if last_ok - i + 1 > best_len:
            best_len = last_ok - i + 1
            best = (i, last_ok)
        i = j if j > i else i + 1
    return best


def closed_band_range(pts: np.ndarray,
                      slice_m: float = RING_SLICE_M,
                      n_bins: int = RING_BINS,
                      max_gap_deg: float = RING_MAX_GAP_DEG,
                      bridge: int = RING_BRIDGE,
                      min_pts: int = RING_MIN_PTS):
    """
    Tra ve (y_lo, y_hi, kept_frac, n_bad, n_slices) -- dai chieu cao chi gom cac
    lat khep kin -- hoac None neu khong tim duoc lat nao dat.
    """
    edges, gaps, _counts = slice_ring_gaps(pts, slice_m, n_bins, min_pts)
    ok = gaps <= max_gap_deg
    run = longest_closed_run(ok, bridge)
    if run is None:
        return None
    i0, i1 = run
    y_lo, y_hi = float(edges[i0]), float(edges[i1 + 1])
    total = float(edges[-1] - edges[0])
    kept = (y_hi - y_lo) / total if total > 1e-9 else 0.0
    return y_lo, y_hi, kept, int((~ok).sum()), int(len(ok))


def _boundary_rims(verts: np.ndarray, tris: np.ndarray):
    """
    Tim VANH o hai dau mesh va do do "phang" cua chung.

    Vanh = thanh phan lien thong cua do thi CANH BIEN chua dinh cao nhat (hoac
    thap nhat). Phai lam theo thanh phan lien thong, KHONG duoc lay "moi dinh
    bien trong vung 25% tren/duoi": lo hong gan goc se bi tinh nham vao vanh
    day va gay cat oan (do duoc: cay_0030 mat 50.2%, cay_0007 mat 21.4% voi
    cach sai, so voi 9.6% va 0.0% voi cach dung).

    Union-find thay cho scipy.sparse.csgraph de khong them phu thuoc moi.

    Tra ve (spread_top, spread_bot, y_cut_hi, y_cut_lo) -- don vi met.
    """
    y_lo = float(verts[:, 1].min())
    y_hi = float(verts[:, 1].max())
    if len(tris) == 0:
        return 0.0, 0.0, y_hi, y_lo

    e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]),
                axis=1)
    uq, cnt = np.unique(e, axis=0, return_counts=True)
    b = uq[cnt == 1]                       # canh chi thuoc 1 tam giac = canh bien
    if len(b) == 0:
        return 0.0, 0.0, y_hi, y_lo

    bv, inv = np.unique(b.ravel(), return_inverse=True)
    inv = inv.reshape(-1, 2)

    parent = list(range(len(bv)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # nen duong
            x = parent[x]
        return x

    for a, c in inv:
        ra, rc = _find(int(a)), _find(int(c))
        if ra != rc:
            parent[ra] = rc

    lab = np.array([_find(i) for i in range(len(bv))])
    y = verts[bv, 1]
    yt = y[lab == lab[int(np.argmax(y))]]   # vanh chua dinh cao nhat
    yb = y[lab == lab[int(np.argmin(y))]]   # vanh chua dinh thap nhat

    return (float(yt.max() - yt.min()), float(yb.max() - yb.min()),
            float(yt.min()), float(yb.max()))


def flatten_open_ends(mesh: o3d.geometry.TriangleMesh,
                      tol: float = RIM_FLAT_TOL,
                      max_iter: int = RIM_MAX_ITER):
    """
    Cat phang hai dau: bo phan vanh rang cua o ngon (va o goc).

    Ly do can buoc nay rieng, ngoai ring trim: ring trim do POINT CLOUD nen chi
    bat duoc truong hop camera khong quay het vong. Ngon cay THUA DIEM thi van
    qua duoc phep thu do, nhung bo cat mat do cua Poisson lai gam mat khong deu
    o do -> vanh rang cua. Do duoc tren cay_0004 SAU ring trim: vanh ngon trai
    dai 21.7cm trong khi vanh day 0.0cm.

    Tra ve (mesh, cut_top_m, cut_bot_m).
    """
    v0 = np.asarray(mesh.vertices)
    if len(v0) == 0:
        return mesh, 0.0, 0.0
    y0_lo, y0_hi = float(v0[:, 1].min()), float(v0[:, 1].max())

    for _ in range(max_iter):
        V = np.asarray(mesh.vertices)
        T = np.asarray(mesh.triangles)
        if len(T) == 0:
            break
        s_top, s_bot, cut_hi, cut_lo = _boundary_rims(V, T)
        lo, hi = float(V[:, 1].min()), float(V[:, 1].max())
        need_hi, need_lo = s_top > tol, s_bot > tol
        if not need_hi and not need_lo:
            break
        keep = ((V[:, 1] >= (cut_lo if need_lo else lo)) &
                (V[:, 1] <= (cut_hi if need_hi else hi)))
        if int(keep.sum()) < 100:
            break                          # khong cat den muc lam rong mesh
        mesh.remove_vertices_by_mask(~keep)
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()

    V = np.asarray(mesh.vertices)
    if len(V) == 0:
        return mesh, 0.0, 0.0
    return (mesh,
            y0_hi - float(V[:, 1].max()),
            float(V[:, 1].min()) - y0_lo)


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

def poisson_trunk_mesh(pcd: o3d.geometry.PointCloud,
                       depth: int = DEFAULT_DEPTH,
                       verbose: bool = True) -> o3d.geometry.TriangleMesh:
    """Poisson + cat vung mat do thap + cat phan ngoai tam cao cua point cloud."""
    pts = np.asarray(pcd.points)
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=False)
    densities = np.asarray(densities)
    n_raw = len(mesh.vertices)

    # Poisson luon tra ve mot mat KIN, nen no "doan" them phan khong co diem
    # nao chong lung. Cat theo mat do de bo dung phan bia do.
    if len(densities):
        thr = float(np.quantile(densities, DENSITY_QUANTILE))
        mesh.remove_vertices_by_mask(densities <= thr)
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()

    # Poisson cung keo dai qua dau va qua goc (nap kin). Cat ve dung tam cao
    # cua point cloud.
    verts = np.asarray(mesh.vertices)
    if len(verts):
        outside = (verts[:, 1] < y_min) | (verts[:, 1] > y_max)
        if outside.any():
            mesh.remove_vertices_by_mask(outside)
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_triangles()

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()

    if verbose:
        print(f"  Poisson depth={depth}: {n_raw:,} -> {len(mesh.vertices):,} verts  "
              f"{len(mesh.triangles):,} tris", flush=True)
    return mesh


def keep_largest_component(mesh: o3d.geometry.TriangleMesh):
    """Giu connected component nhieu tam giac nhat."""
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts) <= 1:
        return mesh, 0
    keep = int(np.argmax(counts))
    drop = labels != keep
    n_drop = int(drop.sum())
    mesh.remove_triangles_by_mask(drop)
    mesh.remove_unreferenced_vertices()
    return mesh, n_drop


# ---------------------------------------------------------------------------
# Per-scan
# ---------------------------------------------------------------------------

def process_scan(scan_dir: str,
                 depth: int = DEFAULT_DEPTH,
                 skip_existing: bool = False,
                 ring_trim: bool = True,
                 max_gap_deg: float = RING_MAX_GAP_DEG,
                 ring_slice: float = RING_SLICE_M,
                 ring_bridge: int = RING_BRIDGE,
                 rim_flatten: bool = True,
                 rim_tol: float = RIM_FLAT_TOL,
                 verbose: bool = True) -> dict:
    name = os.path.basename(scan_dir.rstrip("/\\"))
    out_dir = os.path.join(scan_dir, "output")
    result = {"name": name, "ok": False, "msg": ""}

    pcd_path = os.path.join(out_dir, "trunk_pointcloud.ply")
    out_path = os.path.join(out_dir, "trunk_mesh_poisson.ply")

    if not os.path.exists(pcd_path):
        result["msg"] = "trunk_pointcloud.ply not found -- chay 05_trunk_isolation.py truoc"
        return result
    if skip_existing and os.path.exists(out_path):
        result.update({"ok": True, "msg": "skip (da co)"})
        return result

    t0 = time.time()
    if verbose:
        print(f"\n{'='*60}")
        print(f"Poisson trunk mesh: {name}")
        print(f"{'='*60}", flush=True)

    pcd = o3d.io.read_point_cloud(pcd_path)
    n_pts = len(pcd.points)
    if n_pts < 1000:
        result["msg"] = f"point cloud qua it diem ({n_pts})"
        return result
    if verbose:
        print(f"  Loaded {n_pts:,} trunk points", flush=True)

    _p0 = np.asarray(pcd.points)
    h_orig = float(_p0[:, 1].max() - _p0[:, 1].min())

    # --- Ring trim: bo cac lat cat khong khep kin ---------------------------
    # Chi cat BAN SAO TRONG BO NHO. KHONG ghi de trunk_pointcloud.ply tren dia:
    # file do con la nguon mau cho buoc 5, mat na cho buoc 6, nguon nhan cho
    # buoc 8. Diem ngoai dai don gian khong co vertex mesh nao o gan.
    ring = None
    if ring_trim:
        _pts = np.asarray(pcd.points)
        ring = closed_band_range(_pts, slice_m=ring_slice, n_bins=RING_BINS,
                                 max_gap_deg=max_gap_deg, bridge=ring_bridge)
        if ring is None:
            if verbose:
                print("  Ring trim: KHONG co lat nao khep kin -- bo qua buoc cat",
                      flush=True)
        else:
            _y_lo, _y_hi, _kept, _n_bad, _n_sl = ring
            _keep = (_pts[:, 1] >= _y_lo) & (_pts[:, 1] <= _y_hi)
            _n_cut = int((~_keep).sum())
            if verbose:
                print(f"  Ring trim: {_n_sl} lat day {ring_slice*100:.0f}cm, "
                      f"{_n_bad} lat co khe > {max_gap_deg:.0f} deg", flush=True)
                print(f"             giu y=[{_y_lo:.3f}, {_y_hi:.3f}] = "
                      f"{_kept*100:.0f}% chieu cao, bo {_n_cut:,} diem",
                      flush=True)
            if _kept < RING_WARN_FRAC:
                print(f"  *** CANH BAO: chi giu duoc {_kept*100:.0f}% chieu cao "
                      f"(< {RING_WARN_FRAC*100:.0f}%) -- scan nay nen QUET LAI ***",
                      flush=True)
            if _n_cut:
                pcd = pcd.select_by_index(np.where(_keep)[0])
            if len(pcd.points) < 1000:
                result["msg"] = (f"con {len(pcd.points)} diem sau ring trim -- "
                                 f"noi long --max-gap-deg hoac dung --no-ring-trim")
                return result

    orient_normals_outward(pcd)
    mesh = poisson_trunk_mesh(pcd, depth=depth, verbose=verbose)

    if len(mesh.vertices) < 100:
        result["msg"] = "mesh rong sau khi cat mat do -- thu tang --depth"
        return result

    mesh, n_drop = keep_largest_component(mesh)
    if verbose and n_drop:
        print(f"  Keep largest component: removed {n_drop:,} fragment tris", flush=True)

    # --- Cat phang hai dau (vanh rang cua) ---------------------------------
    cut_top = cut_bot = 0.0
    if rim_flatten:
        mesh, cut_top, cut_bot = flatten_open_ends(mesh, tol=rim_tol)
        if verbose and (cut_top > 1e-4 or cut_bot > 1e-4):
            print(f"  Cat phang hai dau: bo {cut_top*100:.1f}cm o ngon, "
                  f"{cut_bot*100:.1f}cm o goc", flush=True)
        if len(mesh.vertices) < 100:
            result["msg"] = ("mesh rong sau khi cat phang hai dau -- "
                             "dung --no-rim-flatten de kiem tra")
            return result

    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, mesh)

    elapsed = time.time() - t0
    result.update({
        "ok": True,
        "msg": f"verts:{len(mesh.vertices):,} tris:{len(mesh.triangles):,} [{elapsed:.0f}s]",
    })
    if ring is not None:
        result.update({
            "ring_y_lo":      round(ring[0], 4),
            "ring_y_hi":      round(ring[1], 4),
            "ring_kept_frac": round(ring[2], 4),
            "ring_n_bad":     ring[3],
            "ring_n_slices":  ring[4],
        })
    result["rim_cut_top_cm"] = round(cut_top * 100, 1)
    result["rim_cut_bot_cm"] = round(cut_bot * 100, 1)

    # mesh_kept_frac = chieu cao mesh CUOI CUNG / chieu cao point cloud GOC.
    # Day moi la con so quyet dinh co gan co hay khong: ring_kept_frac chi tinh
    # den buoc cat point cloud, chua tinh phan vanh rang cua bi cat sau do.
    _mv = np.asarray(mesh.vertices)
    _h_final = float(_mv[:, 1].max() - _mv[:, 1].min()) if len(_mv) else 0.0
    result["mesh_height_m"] = round(_h_final, 4)
    if h_orig > 1e-9:
        result["mesh_kept_frac"] = round(_h_final / h_orig, 4)
        result["ring_flag"] = bool(_h_final / h_orig < RING_WARN_FRAC)
    if verbose:
        print(f"  Saved {out_path}  [{elapsed:.0f}s]", flush=True)
    return result


def run_batch(raw_data_dir: str, scan_filter=None, depth: int = DEFAULT_DEPTH,
              skip_existing: bool = False, ring_trim: bool = True,
              max_gap_deg: float = RING_MAX_GAP_DEG,
              ring_slice: float = RING_SLICE_M,
              ring_bridge: int = RING_BRIDGE,
              rim_flatten: bool = True,
              rim_tol: float = RIM_FLAT_TOL) -> None:
    scan_dirs = sorted(
        os.path.join(raw_data_dir, d)
        for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
        and os.path.exists(os.path.join(raw_data_dir, d, "output", "trunk_pointcloud.ply"))
    )
    if scan_filter:
        scan_dirs = [d for d in scan_dirs if os.path.basename(d) == scan_filter]

    print(f"Found {len(scan_dirs)} scans with trunk_pointcloud.ply  (depth={depth})")

    results = []
    for scan_dir in scan_dirs:
        try:
            results.append(process_scan(scan_dir, depth=depth,
                                        skip_existing=skip_existing,
                                        ring_trim=ring_trim,
                                        max_gap_deg=max_gap_deg,
                                        ring_slice=ring_slice,
                                        ring_bridge=ring_bridge,
                                        rim_flatten=rim_flatten,
                                        rim_tol=rim_tol))
        except Exception as e:
            results.append({"name": os.path.basename(scan_dir), "ok": False,
                            "msg": f"loi: {e}"})

    ok = sum(1 for r in results if r["ok"])
    print(f"\n{'='*60}")
    print(f"Batch done: {ok} ok, {len(results)-ok} failed")
    for r in results:
        if not r["ok"]:
            print(f"  - {r['name']}: {r['msg']}")

    kept = [r["ring_kept_frac"] for r in results if "ring_kept_frac" in r]
    if kept:
        print(f"Ring trim: giu trung binh {100*sum(kept)/len(kept):.1f}% chieu cao "
              f"(thap nhat {100*min(kept):.0f}%)")
    mk = [r["mesh_kept_frac"] for r in results if "mesh_kept_frac" in r]
    if mk:
        print(f"Sau khi cat phang hai dau: mesh giu trung binh "
              f"{100*sum(mk)/len(mk):.1f}% chieu cao point cloud goc "
              f"(thap nhat {100*min(mk):.0f}%)")

    # Quality gate: liet ke cac scan bi cat qua nhieu -- xem CLAUDE.md muc 12
    # item 14(b) ("khong co quality gate nao ton tai").
    flagged = [r for r in results if r.get("ring_flag")]
    if flagged:
        print("")
        print(f"CAC CAY CAN QUET LAI (giu < {RING_WARN_FRAC*100:.0f}% chieu cao):")
        for r in flagged:
            print(f"  - {r['name']}: mesh giu "
                  f"{r.get('mesh_kept_frac', 0)*100:.0f}% chieu cao goc  "
                  f"({r.get('ring_n_bad', 0)}/{r.get('ring_n_slices', 0)} "
                  f"lat khong khep kin, cat phang "
                  f"{r.get('rim_cut_top_cm', 0):.0f}cm ngon + "
                  f"{r.get('rim_cut_bot_cm', 0):.0f}cm goc)")

    with open(os.path.join(raw_data_dir, "poisson_summary.json"), "w") as f:
        json.dump({"depth": depth,
                   "ring_trim": ring_trim,
                   "max_gap_deg": max_gap_deg,
                   "ring_slice_m": ring_slice,
                   "ring_bridge": ring_bridge,
                   "results": results}, f, indent=2)


def main():
    p = argparse.ArgumentParser(
        description="Dung mesh than cay bang Poisson tu trunk_pointcloud.ply")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", metavar="NAME")
    g.add_argument("--all", action="store_true")
    p.add_argument("--raw-data", default=RAW_DATA_DIR)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                   help=f"Poisson octree depth (default {DEFAULT_DEPTH}). Cao hon = "
                        f"bam nhieu hon va PHONG dien tich; xem docstring")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--max-gap-deg", type=float, default=RING_MAX_GAP_DEG,
                   help=f"Khe goc trong lon nhat cho phep trong mot lat cat "
                        f"(default {RING_MAX_GAP_DEG:.0f} deg). Lon hon = cat it hon")
    p.add_argument("--ring-slice", type=float, default=RING_SLICE_M,
                   help=f"Do day lat cat (m) khi do do khep kin "
                        f"(default {RING_SLICE_M})")
    p.add_argument("--ring-bridge", type=int, default=RING_BRIDGE,
                   help=f"So lat hong lien tiep duoc phep bac cau "
                        f"(default {RING_BRIDGE})")
    p.add_argument("--no-ring-trim", action="store_true",
                   help="Khong cat lat khong khep kin (nhu truoc ban va)")
    p.add_argument("--no-rim-flatten", action="store_true",
                   help="Khong cat phang hai dau (giu ca vanh rang cua o ngon)")
    p.add_argument("--rim-tol", type=float, default=RIM_FLAT_TOL,
                   help=f"Vanh trai dai qua muc nay (m) thi coi la rang cua "
                        f"(default {RIM_FLAT_TOL})")
    args = p.parse_args()

    run_batch(args.raw_data, scan_filter=args.scan, depth=args.depth,
              skip_existing=args.skip_existing,
              ring_trim=not args.no_ring_trim,
              max_gap_deg=args.max_gap_deg,
              ring_slice=args.ring_slice,
              ring_bridge=args.ring_bridge,
              rim_flatten=not args.no_rim_flatten,
              rim_tol=args.rim_tol)


if __name__ == "__main__":
    main()
