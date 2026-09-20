#!/usr/bin/env python3
"""
Step 03: Re-color TSDF mesh bang mau tu trunk_pointcloud.ply (nearest-neighbor).

Van de cu: TSDF dung RGB 256x192 va weighted-mean nhieu frame -> mau bi nhem.
Fix moi:   trunk_pointcloud.ply da co mau chinh xac tu camera goc (dia y xanh ro).
           -> Voi moi mesh vertex, tim diem PCD gan nhat -> lay mau do.

Pipeline:
  1. Load trunk_mesh_tsdf.ply  (geometry TSDF)
  2. Load trunk_pointcloud.ply (colors chinh xac)
  3. Build cKDTree tren PCD, query nearest neighbor cho tung vertex
  4. Gan mau, luu trunk_mesh_recolored.ply

Usage:
    python 03_recolor_mesh.py --scan 26Q1
    python 03_recolor_mesh.py --all --skip-existing
"""

import argparse, json, os, sys, time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from pipeline_io import dump_summary, merged_results

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "Raw_data")


# ── Per-scan ──────────────────────────────────────────────────────────────────

def process_scan(scan_dir: str, verbose: bool = True,
                 prefer_poisson: bool = True) -> dict:
    name    = os.path.basename(scan_dir)
    out_dir = os.path.join(scan_dir, "output")
    result  = {"name": name, "ok": False, "msg": ""}

    # [Sep 2026] Uu tien mesh Poisson dung tu trunk_pointcloud.ply. Mesh TSDF
    # co lo thung lon tren dung phan than cay DA DUOC QUET, do TSDF do khoang
    # cach doc theo tia camera nen be mat chi nhin duoc o goc xien khong bao
    # gio duoc tich hop (xem docstring patch_recolor_prefer_poisson.py va
    # CLAUDE.md muc 12 item 15). Tu dong lui ve TSDF neu chua chay
    # 02_mesh_poisson_trunk.py.
    poisson_path = os.path.join(out_dir, "trunk_mesh_poisson.ply")
    tsdf_path    = os.path.join(out_dir, "trunk_mesh_tsdf.ply")
    pcd_path     = os.path.join(out_dir, "trunk_pointcloud.ply")

    if prefer_poisson and os.path.exists(poisson_path):
        mesh_path = poisson_path
        mesh_src  = "poisson"
    else:
        mesh_path = tsdf_path
        mesh_src  = "tsdf"

    if not os.path.exists(mesh_path):
        result["msg"] = ("khong tim thay trunk_mesh_poisson.ply lan "
                         "trunk_mesh_tsdf.ply -- chay 02_mesh_poisson_trunk.py "
                         "(hoac 02_mesh_tsdf.py) truoc")
        return result
    if verbose:
        print(f"  Geometry source: {mesh_src} ({os.path.basename(mesh_path)})",
              flush=True)
    if not os.path.exists(pcd_path):
        result["msg"] = "trunk_pointcloud.ply not found — chay 01_clean_pointcloud.py truoc"
        return result

    t0 = time.time()

    # 1. Load TSDF mesh (geometry)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    verts = np.asarray(mesh.vertices, dtype=np.float32)   # (N, 3)
    N = len(verts)
    if verbose:
        print(f"  Mesh : {N:,} verts  {len(mesh.triangles):,} tris", flush=True)

    # 2. Load trunk PCD (colors)
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_pts  = np.asarray(pcd.points, dtype=np.float32)   # (M, 3)
    if not pcd.has_colors():
        result["msg"] = "trunk_pointcloud.ply khong co colors"
        return result
    pcd_cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)  # (M, 3) uint8
    M = len(pcd_pts)
    if verbose:
        print(f"  PCD  : {M:,} pts (source colors)", flush=True)

    # 3. Nearest-neighbor: moi mesh vertex -> diem PCD gan nhat
    if verbose:
        print(f"  Building KDTree + querying {N:,} vertices ...", flush=True)
    tree = cKDTree(pcd_pts)
    dists, idx = tree.query(verts, k=1, workers=-1)

    # 4. Assign colors
    new_colors = pcd_cols[idx].astype(np.float32) / 255.0   # (N, 3) float [0,1]

    # Log: vert nao qua xa PCD (co the la artifact geometry)
    far = (dists > 0.02).sum()
    if verbose and far:
        print(f"  Caution: {far:,} verts cach PCD > 2cm "
              f"(co the la artifact TSDF)", flush=True)

    mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)

    # 5. Save
    out_path = os.path.join(out_dir, "trunk_mesh_recolored.ply")
    o3d.io.write_triangle_mesh(out_path, mesh, write_vertex_colors=True)

    elapsed = time.time() - t0
    result.update({
        "ok":  True,
        "msg": f"verts:{N:,} pcd:{M:,} far:{far:,} [{elapsed:.0f}s]",
    })
    if verbose:
        print(f"  Saved {out_path}  [{elapsed:.0f}s]", flush=True)
    return result


# ── Batch ─────────────────────────────────────────────────────────────────────

def run_batch(raw_data_dir, scan_filter=None, skip_existing=False,
              prefer_poisson=True, fresh_summary=False):
    if scan_filter:
        dirs = [os.path.join(raw_data_dir, scan_filter)]
        if not os.path.isdir(dirs[0]):
            print(f"Scan not found: {dirs[0]}"); sys.exit(1)
    else:
        dirs = sorted([
            os.path.join(raw_data_dir, d)
            for d in os.listdir(raw_data_dir)
            if os.path.isdir(os.path.join(raw_data_dir, d))
        ])

    total = len(dirs)
    print(f"\n{'='*55}")
    print(f"Recolor Mesh (PCD colors)  |  {total} scans")
    print(f"{'='*55}\n")

    ok_count = skip_count = 0
    results  = []

    for i, scan_dir in enumerate(dirs):
        name     = os.path.basename(scan_dir)
        out_path = os.path.join(scan_dir, "output", "trunk_mesh_recolored.ply")

        if skip_existing and os.path.exists(out_path):
            print(f"[{i+1:3d}/{total}] {name} -- SKIP")
            skip_count += 1
            continue

        print(f"[{i+1:3d}/{total}] {name}")
        r = process_scan(scan_dir, verbose=True, prefer_poisson=prefer_poisson)
        results.append(r)
        status = "OK  " if r["ok"] else "FAIL"
        print(f"  {status} {r['msg']}\n")
        if r["ok"]:
            ok_count += 1

    print(f"{'='*55}")
    print(f"DONE: {ok_count}/{total - skip_count} OK  ({skip_count} skipped)")

    # [Sep 2026] Gop thay vi ghi de. Truoc day `--scan X` di qua chinh ham nay
    # voi danh sach loc con 1 cay roi ghi de file bang dung dong do -- do do
    # recolor/trim/finalize/lichen_summary.json tren corpus deu bi rut ve 1/46.
    # Xem pipeline_io.py.
    sm_path = os.path.join(raw_data_dir, "recolor_summary.json")
    dump_summary(sm_path, {"source": "pcd",
                           "results": merged_results(sm_path, results,
                                                     fresh=fresh_summary)})


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Re-color TSDF mesh bang mau tu trunk_pointcloud.ply")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", metavar="NAME")
    g.add_argument("--all",  action="store_true")
    p.add_argument("--raw-data",      default=RAW_DATA_DIR)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-poisson", action="store_true",
                   help="Dung mesh TSDF nhu truoc thay vi trunk_mesh_poisson.ply "
                        "(xem CLAUDE.md muc 12 item 15)")
    p.add_argument("--fresh-summary", action="store_true",
                   help="Ghi de recolor_summary.json thay vi gop vao ban ghi cu")
    args = p.parse_args()
    run_batch(args.raw_data,
              scan_filter    = args.scan if args.scan else None,
              skip_existing  = args.skip_existing,
              prefer_poisson = not args.no_poisson,
              fresh_summary  = args.fresh_summary)


if __name__ == "__main__":
    main()
