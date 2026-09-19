#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_poisson_flat_rims.py
==========================
Va 02_mesh_poisson_trunk.py (buoc 4b): cat phang hai dau mesh, bo phan VANH
RANG CUA o ngon (va o goc) ma ring trim khong bat duoc.

VAN DE -- VA TAI SAO RING TRIM KHONG DU
---------------------------------------
patch_poisson_ring_trim.py do PONT CLOUD: no bo cac lat ma camera khong quay
het vong (khe goc > 20 do). Dung, nhung do la MOT trong HAI nguyen nhan:

  1. Camera khong quay het vong  -> khe goc trong point cloud   -> ring trim bat
  2. Ngon cay THUA DIEM (90-312 diem/lat so voi 700+ o giua)    -> KHONG BAT
     -> bo cat mat do cua Poisson (DENSITY_QUANTILE = 5%) gam mat khong deu
     -> vanh tren khong phang ma rang cua

Do duoc tren cay_0004 sau khi da co ring trim: mesh chi con 2 vong bien (day va
ngon), nghe thi sach, NHUNG vong o ngon trai dai 21.7cm theo phuong doc, trong
khi vong o day trai dai 0.0cm. Vanh phang thi trai dai phai ~0. 21.7cm chinh la
phan khuyet lom chom nguoi dung van nhin thay.

Dem SO vong bien la chua du -- phai do HINH DANG cua chung.

DO TREN CA 46 CAY
-----------------
Trai dai cua vanh ngon: trung vi 3.8cm, p90 16.8cm, toi da 35.5cm (cay_0038).
  21/46 cay co vanh ngon rang cua > 5cm.
Trai dai cua vanh day : trung vi 2.1cm, p90  5.0cm, toi da 19.3cm (cay_0030).
  5/46 cay > 5cm.

DA THU VA LOAI BO MOT TIEU CHI NGAT HON
---------------------------------------
"Giu dai chieu cao KHONG chua canh bien nao" (dai do theo dinh nghia la ong
kin) nghe hap dan nhung do ra chi giu 84.6% trung binh va 8/46 cay tut duoi
70% -- mot lo hong nho o giua cat doi ca cay (cay_0025 chi con 16%). Bi loai.

CACH LAM
--------
Xac dinh VANH bang THANH PHAN LIEN THONG cua do thi canh bien, roi cat phang
tai day vanh do. Lan nguyen mau dau tien dung "moi dinh bien trong vung 25%
tren/duoi" va SAI: lo hong gan goc bi tinh nham vao vanh day, lam cay_0030 mat
50.2% va cay_0007 mat 21.4% mot cach oan uong. Voi thanh phan lien thong dung
thi hai cay do chi mat 9.6% va 0.0%.

Lap toi da RIM_MAX_ITER lan vi moi lan cat bo mot hang tam giac lai lo ra vanh
moi; thuc te hoi tu sau 2-3 lan.

Dung union-find thay vi scipy.sparse.csgraph de khong them phu thuoc moi vao
file nay (run_full_pipeline.py khai bao module can cho buoc 4b la numpy+open3d).

KET QUA DO DUOC
---------------
  cay_0004  ngon 21.7cm -> 0.4cm   mat 9.5%
  cay_0038  ngon 35.5cm -> 0.0cm   mat 15.6%
  cay_0030  day  19.3cm -> 0.0cm   mat 9.6%
  cay_0007  von da phang           mat 0.0%
  cay_0051  von da phang           mat 0.0%
Tren 46 cay: mat them trung binh 4.9% chieu cao, trung vi 3.6%, toi da 26.8%
(cay_0003 -- capture hong da biet). Chi 3 cay mat >10%, 1 cay mat >20%.

CACH DUNG
---------
    python patch_poisson_flat_rims.py

Sau khi va:
    python 02_mesh_poisson_trunk.py --all --raw-data <DIR>
    python 02_mesh_poisson_trunk.py --scan <NAME> --no-rim-flatten   # tat
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
    '''RING_WARN_FRAC   = 0.70   # giu duoi muc nay -> canh bao + co trong JSON''',
    '''RING_WARN_FRAC   = 0.70   # giu duoi muc nay -> canh bao + co trong JSON

# --- Cat phang hai dau -- xem docstring patch_poisson_flat_rims.py
RIM_FLAT_TOL  = 0.015   # vanh trai dai qua muc nay (m) thi coi la rang cua
RIM_MAX_ITER  = 6       # cat lap; thuc te hoi tu sau 2-3 lan''',
))

# --- Site 2: new functions --------------------------------------------------
SNIPPETS.append((
    "rim_functions",
    '''# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

def poisson_trunk_mesh(pcd: o3d.geometry.PointCloud,''',
    '''def _boundary_rims(verts: np.ndarray, tris: np.ndarray):
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

def poisson_trunk_mesh(pcd: o3d.geometry.PointCloud,''',
))

# --- Site 3: process_scan signature ----------------------------------------
SNIPPETS.append((
    "process_scan_signature",
    '''                 ring_bridge: int = RING_BRIDGE,
                 verbose: bool = True) -> dict:''',
    '''                 ring_bridge: int = RING_BRIDGE,
                 rim_flatten: bool = True,
                 rim_tol: float = RIM_FLAT_TOL,
                 verbose: bool = True) -> dict:''',
))

# --- Site 4: call after keep_largest_component -----------------------------
SNIPPETS.append((
    "flatten_call",
    '''    mesh, n_drop = keep_largest_component(mesh)
    if verbose and n_drop:
        print(f"  Keep largest component: removed {n_drop:,} fragment tris", flush=True)''',
    '''    mesh, n_drop = keep_largest_component(mesh)
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
            return result''',
))

# --- Site 5: result fields --------------------------------------------------
SNIPPETS.append((
    "result_rim_fields",
    '''    if ring is not None:
        result.update({
            "ring_y_lo":      round(ring[0], 4),
            "ring_y_hi":      round(ring[1], 4),
            "ring_kept_frac": round(ring[2], 4),
            "ring_n_bad":     ring[3],
            "ring_n_slices":  ring[4],
            "ring_flag":      bool(ring[2] < RING_WARN_FRAC),
        })''',
    '''    if ring is not None:
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
        result["ring_flag"] = bool(_h_final / h_orig < RING_WARN_FRAC)''',
))

# --- Site 6: record the original cloud height -------------------------------
SNIPPETS.append((
    "record_h_orig",
    '''    if verbose:
        print(f"  Loaded {n_pts:,} trunk points", flush=True)

    # --- Ring trim: bo cac lat cat khong khep kin ---------------------------''',
    '''    if verbose:
        print(f"  Loaded {n_pts:,} trunk points", flush=True)

    _p0 = np.asarray(pcd.points)
    h_orig = float(_p0[:, 1].max() - _p0[:, 1].min())

    # --- Ring trim: bo cac lat cat khong khep kin ---------------------------''',
))

# --- Site 7: run_batch signature -------------------------------------------
SNIPPETS.append((
    "run_batch_signature",
    '''              ring_slice: float = RING_SLICE_M,
              ring_bridge: int = RING_BRIDGE) -> None:''',
    '''              ring_slice: float = RING_SLICE_M,
              ring_bridge: int = RING_BRIDGE,
              rim_flatten: bool = True,
              rim_tol: float = RIM_FLAT_TOL) -> None:''',
))

# --- Site 8: run_batch forwards --------------------------------------------
SNIPPETS.append((
    "run_batch_call",
    '''                                        ring_slice=ring_slice,
                                        ring_bridge=ring_bridge))''',
    '''                                        ring_slice=ring_slice,
                                        ring_bridge=ring_bridge,
                                        rim_flatten=rim_flatten,
                                        rim_tol=rim_tol))''',
))

# --- Site 9: batch summary --------------------------------------------------
SNIPPETS.append((
    "batch_summary",
    '''    kept = [r["ring_kept_frac"] for r in results if "ring_kept_frac" in r]
    if kept:
        print(f"Ring trim: giu trung binh {100*sum(kept)/len(kept):.1f}% chieu cao "
              f"(thap nhat {100*min(kept):.0f}%)")''',
    '''    kept = [r["ring_kept_frac"] for r in results if "ring_kept_frac" in r]
    if kept:
        print(f"Ring trim: giu trung binh {100*sum(kept)/len(kept):.1f}% chieu cao "
              f"(thap nhat {100*min(kept):.0f}%)")
    mk = [r["mesh_kept_frac"] for r in results if "mesh_kept_frac" in r]
    if mk:
        print(f"Sau khi cat phang hai dau: mesh giu trung binh "
              f"{100*sum(mk)/len(mk):.1f}% chieu cao point cloud goc "
              f"(thap nhat {100*min(mk):.0f}%)")''',
))

# --- Site 10: flagged list uses mesh_kept_frac -----------------------------
SNIPPETS.append((
    "flagged_list",
    '''        for r in flagged:
            print(f"  - {r['name']}: giu {r['ring_kept_frac']*100:.0f}%  "
                  f"({r.get('ring_n_bad', 0)}/{r.get('ring_n_slices', 0)} "
                  f"lat khong khep kin)")''',
    '''        for r in flagged:
            print(f"  - {r['name']}: mesh giu "
                  f"{r.get('mesh_kept_frac', 0)*100:.0f}% chieu cao goc  "
                  f"({r.get('ring_n_bad', 0)}/{r.get('ring_n_slices', 0)} "
                  f"lat khong khep kin, cat phang "
                  f"{r.get('rim_cut_top_cm', 0):.0f}cm ngon + "
                  f"{r.get('rim_cut_bot_cm', 0):.0f}cm goc)")''',
))

# --- Site 11: CLI -----------------------------------------------------------
SNIPPETS.append((
    "cli",
    '''    p.add_argument("--no-ring-trim", action="store_true",
                   help="Khong cat lat khong khep kin (nhu truoc ban va)")
    args = p.parse_args()''',
    '''    p.add_argument("--no-ring-trim", action="store_true",
                   help="Khong cat lat khong khep kin (nhu truoc ban va)")
    p.add_argument("--no-rim-flatten", action="store_true",
                   help="Khong cat phang hai dau (giu ca vanh rang cua o ngon)")
    p.add_argument("--rim-tol", type=float, default=RIM_FLAT_TOL,
                   help=f"Vanh trai dai qua muc nay (m) thi coi la rang cua "
                        f"(default {RIM_FLAT_TOL})")
    args = p.parse_args()''',
))

# --- Site 12: CLI forwards --------------------------------------------------
SNIPPETS.append((
    "cli_forward",
    '''              ring_slice=args.ring_slice,
              ring_bridge=args.ring_bridge)''',
    '''              ring_slice=args.ring_slice,
              ring_bridge=args.ring_bridge,
              rim_flatten=not args.no_rim_flatten,
              rim_tol=args.rim_tol)''',
))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if all(new in content for _, _, new in SNIPPETS):
        print(f"{TARGET_NAME} (flat rims) da duoc va truoc do roi.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Ban va nay phai chay SAU patch_poisson_ring_trim.py.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_flatrims")
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
    print(f"Da va thanh cong {len(applied)} doan (flat rims): {applied}")
    print("")
    print("Kiem tra: python -m py_compile 02_mesh_poisson_trunk.py")
    print("Chay    : python 02_mesh_poisson_trunk.py --all --raw-data <DIR>")
    print("Tat     : python 02_mesh_poisson_trunk.py --scan <NAME> --no-rim-flatten")


if __name__ == "__main__":
    main()
