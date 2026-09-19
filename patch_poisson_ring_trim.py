#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_poisson_ring_trim.py
==========================
Va 02_mesh_poisson_trunk.py (buoc 4b): cat bo cac lat cat theo chieu cao ma
mat cat KHONG KHEP KIN thanh vong tron, truoc khi dung Poisson.

VAN DE
------
Poisson luon tra ve mot mat KIN, nen o nhung doan camera khong quay het quanh
than, no NOI SUY mot mat bia ra bac qua khe. Mat bia do:
  - van duoc 05_detect_lichen.py tinh vao total_cm2 (dien tich tam giac), va
  - luon duoc gan nhan "vo cay sach, khong dia y"
=> lam LOANG ti le dia y bao cao. Do tren cay_0004: mesh co 57 canh bien, trong
do 18 o lat day (dau ho do Y-clip, dung thiet ke) va 4/6/6/10/13 o lat 34-38
(phan ngon). Lat 1-33 khong mot canh bien nao.

Khong the va duoc nhung cho do -- khong co du lieu de va. Cach xu ly la CAT BO.

CHI SO: KHE GOC TRONG LON NHAT, KHONG PHAI "% BIN CO DIEM"
----------------------------------------------------------
"% bin co diem" bi nhieu boi MAT DO DIEM: lat o ngon chi co 90-300 diem so voi
700+ o giua, nen de trot mot bin trong do lay mau chu khong phai do thieu quet.
Khe goc trong lon nhat sach hon han -- do tren 46 cay x 40 lat = 1840 lat:
    khe = 0 deg : 78.6% so lat
    p90 = 20 deg,  p95 = 30 deg,  p99 = 60 deg
Phan bo tach bach, khong phai mot dai lien tuc.

CHON THAM SO BANG SO DO, KHONG PHAI BANG MAT
--------------------------------------------
Do day lat (quet 2/3/5/6/10cm), % chieu cao giu duoc o nguong 20 deg:
    2cm  87.3%  (qua nhieu, chi 268 diem/lat)
    3cm  88.8%
    5cm  91.6%
    6cm  92.0%  <-- CHON (~800 diem/lat)
   10cm  94.6%  (qua tho de cat chinh xac)

Chinh sach ghep doan -- lat hong KHONG CHI O NGON: o nguong 20 deg, 28/46 cay
co lat hong nam GIUA than. Neu cat roi tung lat thi 25/46 cay vo thanh 2-4
manh, va moi keep_largest_component() o buoc 5/6/7 se am tham vut cac manh thua
(dung loi da ghi o CLAUDE.md muc 12 item 7). Do 4 phuong an:
    doan lien tuc, khong bac cau : giu TB 87.1%,  4 cay <70%,  lien khoi
    doan lien tuc, BAC CAU 1 lat : giu TB 91.6%,  1 cay <70%,  lien khoi  <-- CHON
    doan lien tuc, bac cau 2 lat : giu TB 93.2%,  1 cay <70%,  lien khoi
    cat roi tung lat             : giu TB 92.0%,  1 cay <70%,  2-4 MANH
Bac cau 1 lat dat gan bang phuong an cat roi MA VAN LIEN MOT KHOI, nen khong
phai sua bat ky keep_largest_component() nao o cac buoc sau.

RUI RO LECH SO LIEU DIA Y DA DUOC LOAI TRU
-------------------------------------------
Day dung la cai bay o CLAUDE.md muc 12 item 10: tieu chi cu cua trim_ragged_top()
vo tinh tuong quan voi vung CO dia y (vo cay ram tai vung dia y dung mesh kem
min hon), nen no cat mat dung thu can do. Kiem tra bang cach chay
classify_lichen_local_contrast() tren 8 cay, so ti le dia y cua phan giu lai voi
toan than:
    cay_0004 +0.00pp   cay_0005 -0.07pp   cay_0007 +0.00pp   cay_0009 +0.04pp
    cay_0013 -0.00pp   cay_0025 -0.04pp   cay_0033 +0.37pp   cay_0041 +0.79pp
Lech toi da +0.79 diem phan tram (cay_0041: 11.06% so voi 10.27%). Phan bi cat
NGHEO dia y hon phan giu lai, khong phai giau hon.

TAI SAO DAT O BUOC 4B CHU KHONG PHAI BUOC 7
--------------------------------------------
04b_finalize_mesh.py co san trim_ragged_top() dung cho viec nay, va hien thuc te
la NO-OP (tieu chi duy nhat con song la slice_total < 20 vertex/lat, trong khi
lat that co 649-7439 vertex). Van khong dung cho do vi:
  - No dung truc IMU (read_gravity_up, lech 4.5 deg so voi world-Y), trong khi
    phep do do phu chu vi can dung truc world-Y nhu buoc 3.
  - No chi xet 65% phia tren, khong bat duoc lat hong o giua than.
  - No chay SAU Taubin smoothing va sau buoc 5-6, tuc mesh noi suy sai da kip
    anh huong toi to mau va mat na cat.
  - Cat point cloud TRUOC Poisson thi Poisson khong he dung mat bac qua khe; cat
    mesh sau thi mat bia do da kip tham gia vao uoc luong normal va nguong mat do
    cua cac lat lan can.
trim_ragged_top() KHONG bi dung toi boi ban va nay.

LUU Y
-----
Chi cat BAN SAO TRONG BO NHO cua point cloud. KHONG ghi de trunk_pointcloud.ply
tren dia -- file do con la nguon mau cho buoc 5, mat na cho buoc 6 va nguon nhan
cho buoc 8. Diem ngoai dai don gian la khong co vertex mesh nao o gan nen khong
duoc dung toi.

CACH DUNG
---------
    python patch_poisson_ring_trim.py

Sau khi va:
    python 02_mesh_poisson_trunk.py --all --raw-data <DIR>
    python 02_mesh_poisson_trunk.py --scan <NAME> --max-gap-deg 30
    python 02_mesh_poisson_trunk.py --scan <NAME> --no-ring-trim   # nhu truoc
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "02_mesh_poisson_trunk.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- Site 1: constants ------------------------------------------------------
SNIPPETS.append((
    "constants",
    '''DEFAULT_DEPTH = 7
N_NORMAL_BANDS = 40
DENSITY_QUANTILE = 0.05''',
    '''DEFAULT_DEPTH = 7
N_NORMAL_BANDS = 40
DENSITY_QUANTILE = 0.05

# --- Ring trim (cat lat khong khep kin) -- xem docstring patch_poisson_ring_trim.py
RING_BINS        = 36     # bin 10 do quanh chu vi
RING_MAX_GAP_DEG = 20.0   # khe goc trong lon nhat cho phep trong mot lat
RING_SLICE_M     = 0.06   # do day lat (m) -- do tren 46 cay, xem docstring
RING_BRIDGE      = 1      # so lat hong LIEN TIEP duoc phep bac cau
RING_MIN_PTS     = 10     # lat it hon nguong nay coi nhu hong
RING_WARN_FRAC   = 0.70   # giu duoi muc nay -> canh bao + co trong JSON''',
))

# --- Site 2: new functions --------------------------------------------------
SNIPPETS.append((
    "ring_functions",
    '''# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

def poisson_trunk_mesh(pcd: o3d.geometry.PointCloud,''',
    '''# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

def poisson_trunk_mesh(pcd: o3d.geometry.PointCloud,''',
))

# --- Site 3: process_scan signature ----------------------------------------
SNIPPETS.append((
    "process_scan_signature",
    '''def process_scan(scan_dir: str,
                 depth: int = DEFAULT_DEPTH,
                 skip_existing: bool = False,
                 verbose: bool = True) -> dict:''',
    '''def process_scan(scan_dir: str,
                 depth: int = DEFAULT_DEPTH,
                 skip_existing: bool = False,
                 ring_trim: bool = True,
                 max_gap_deg: float = RING_MAX_GAP_DEG,
                 ring_slice: float = RING_SLICE_M,
                 ring_bridge: int = RING_BRIDGE,
                 verbose: bool = True) -> dict:''',
))

# --- Site 4: the ring trim call --------------------------------------------
SNIPPETS.append((
    "ring_trim_call",
    '''    if verbose:
        print(f"  Loaded {n_pts:,} trunk points", flush=True)

    orient_normals_outward(pcd)''',
    '''    if verbose:
        print(f"  Loaded {n_pts:,} trunk points", flush=True)

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

    orient_normals_outward(pcd)''',
))

# --- Site 5: result dict ----------------------------------------------------
SNIPPETS.append((
    "result_ring_fields",
    '''    elapsed = time.time() - t0
    result.update({
        "ok": True,
        "msg": f"verts:{len(mesh.vertices):,} tris:{len(mesh.triangles):,} [{elapsed:.0f}s]",
    })''',
    '''    elapsed = time.time() - t0
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
            "ring_flag":      bool(ring[2] < RING_WARN_FRAC),
        })''',
))

# --- Site 6: run_batch signature -------------------------------------------
SNIPPETS.append((
    "run_batch_signature",
    '''def run_batch(raw_data_dir: str, scan_filter=None, depth: int = DEFAULT_DEPTH,
              skip_existing: bool = False) -> None:''',
    '''def run_batch(raw_data_dir: str, scan_filter=None, depth: int = DEFAULT_DEPTH,
              skip_existing: bool = False, ring_trim: bool = True,
              max_gap_deg: float = RING_MAX_GAP_DEG,
              ring_slice: float = RING_SLICE_M,
              ring_bridge: int = RING_BRIDGE) -> None:''',
))

# --- Site 7: run_batch forwards --------------------------------------------
SNIPPETS.append((
    "run_batch_call",
    '''            results.append(process_scan(scan_dir, depth=depth,
                                        skip_existing=skip_existing))''',
    '''            results.append(process_scan(scan_dir, depth=depth,
                                        skip_existing=skip_existing,
                                        ring_trim=ring_trim,
                                        max_gap_deg=max_gap_deg,
                                        ring_slice=ring_slice,
                                        ring_bridge=ring_bridge))''',
))

# --- Site 8: batch summary + flagged list + JSON ---------------------------
SNIPPETS.append((
    "batch_summary",
    '''    ok = sum(1 for r in results if r["ok"])
    print(f"\\n{'='*60}")
    print(f"Batch done: {ok} ok, {len(results)-ok} failed")
    for r in results:
        if not r["ok"]:
            print(f"  - {r['name']}: {r['msg']}")

    with open(os.path.join(raw_data_dir, "poisson_summary.json"), "w") as f:
        json.dump({"depth": depth, "results": results}, f, indent=2)''',
    '''    ok = sum(1 for r in results if r["ok"])
    print(f"\\n{'='*60}")
    print(f"Batch done: {ok} ok, {len(results)-ok} failed")
    for r in results:
        if not r["ok"]:
            print(f"  - {r['name']}: {r['msg']}")

    kept = [r["ring_kept_frac"] for r in results if "ring_kept_frac" in r]
    if kept:
        print(f"Ring trim: giu trung binh {100*sum(kept)/len(kept):.1f}% chieu cao "
              f"(thap nhat {100*min(kept):.0f}%)")

    # Quality gate: liet ke cac scan bi cat qua nhieu -- xem CLAUDE.md muc 12
    # item 14(b) ("khong co quality gate nao ton tai").
    flagged = [r for r in results if r.get("ring_flag")]
    if flagged:
        print("")
        print(f"CAC CAY CAN QUET LAI (giu < {RING_WARN_FRAC*100:.0f}% chieu cao):")
        for r in flagged:
            print(f"  - {r['name']}: giu {r['ring_kept_frac']*100:.0f}%  "
                  f"({r.get('ring_n_bad', 0)}/{r.get('ring_n_slices', 0)} "
                  f"lat khong khep kin)")

    with open(os.path.join(raw_data_dir, "poisson_summary.json"), "w") as f:
        json.dump({"depth": depth,
                   "ring_trim": ring_trim,
                   "max_gap_deg": max_gap_deg,
                   "ring_slice_m": ring_slice,
                   "ring_bridge": ring_bridge,
                   "results": results}, f, indent=2)''',
))

# --- Site 9: CLI ------------------------------------------------------------
SNIPPETS.append((
    "cli",
    '''    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    run_batch(args.raw_data, scan_filter=args.scan, depth=args.depth,
              skip_existing=args.skip_existing)''',
    '''    p.add_argument("--skip-existing", action="store_true")
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
    args = p.parse_args()

    run_batch(args.raw_data, scan_filter=args.scan, depth=args.depth,
              skip_existing=args.skip_existing,
              ring_trim=not args.no_ring_trim,
              max_gap_deg=args.max_gap_deg,
              ring_slice=args.ring_slice,
              ring_bridge=args.ring_bridge)''',
))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if all(new in content for _, _, new in SNIPPETS):
        print(f"{TARGET_NAME} (ring trim) da duoc va truoc do roi.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Dung doan, dung sua tay. Kiem tra lai file goc truoc.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_ringtrim")
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
    print(f"Da va thanh cong {len(applied)} doan (ring trim): {applied}")
    print("")
    print("Kiem tra: python -m py_compile 02_mesh_poisson_trunk.py")
    print("Chay    : python 02_mesh_poisson_trunk.py --scan <NAME> --raw-data <DIR>")
    print("Tat     : python 02_mesh_poisson_trunk.py --scan <NAME> --no-ring-trim")


if __name__ == "__main__":
    main()
