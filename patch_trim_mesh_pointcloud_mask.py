#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trim_mesh_pointcloud_mask.py
====================================
Va 04_trim_mesh.py: thay "Step 1b" (gravity_axis_trim + normal_ground_filter +
keep_largest_component som, chay TRUOC PCA) bang mot moc tham chieu dang tin
cay hon: trunk_pointcloud.ply (da duoc 05_trunk_isolation.py xac dinh dung la
than cay -- KDTree camera-anchor + RANSAC ground removal + up-axis fix, xac
nhan tren du lieu that, xem CLAUDE.md muc 5/12).

Da xac nhan tren scan that (cay_0007_1805, D:\\Backup\\Thucdia-18May2026) qua
mot ban replay tung buoc cua chinh cac ham trong file nay, 2 co che gay mat
than cay that o ca hai dau:

  1. gravity_axis_trim() dung MOT truc doc CO DINH song song Y qua 1 centroid
     -- khong bu duoc do nghieng that cua than cay (PCA do duoc ~11 do lech
     Y trong scan nay). Cang xa centroid, truc co dinh nay cang lech khoi vi
     tri than cay that -> keep_largest_component() ben trong no (dong ~143)
     vut nguyen ca mot manh bi dut ket noi do lech truc. Trong log that:
     mat 0.185m dau TREN ngay tai buoc nay.
  2. normal_ground_filter() (thr=0.75) lam vo mesh thanh ~5000 manh (vo mat
     vo/than cay that do TSDF tai tao gan-gui, khong chi mat dat). Ngay sau
     do mot keep_largest_component() thu hai (dong ~311) chi giu DUY NHAT 1
     manh bat ky (manh nhieu tam giac nhat) va vut sach phan con lai. Trong
     log that: mat 0.408m dau DUOI ngay tai buoc nay.

Patch nay:
  - Them ham moi mask_by_trunk_pointcloud(): giu vertex mesh CHI KHI no gan
    (trong ban kinh --mask-radius, mac dinh 5cm) mot diem that trong
    trunk_pointcloud.ply. Loc theo TUNG VERTEX (khong theo component) nen
    khong bao gio vo tinh vut ca mot manh that su la than cay chi vi no bi
    dut ket noi trong mesh.
  - Thay Step 1b: neu trunk_pointcloud.ply ton tai, dung ham tren lam buoc
    loc chinh, KHONG goi keep_largest_component() som (chi con 1 lan duy
    nhat o cuoi ham process_scan(), sau khi PCA + auto_trim_ends + Step 4 da
    don theo truc that). normal_ground_filter() cung khong con chay o nhanh
    chinh nay -- RANSAC ground removal da chay 1 lan khi tao trunk_pointcloud
    .ply roi, nen vertex gan dat se tu dong khong gan diem trunk_pointcloud
    nao va bi loc rot o buoc mask.
  - Fallback: neu khong tim thay trunk_pointcloud.ply (scan cu, hoac co flag
    --no-pointcloud-mask), dung lai gravity_axis_trim + normal_ground_filter
    + keep_largest_component nhu code cu, chi noi radial_factor 2.5 -> 3.5
    de bot nhay cam voi nghieng (nhanh du phong khong co moc nao khac de
    "cuu" mot manh bi dut ket noi).
  - Them 2 CLI flag moi: --mask-radius, --no-pointcloud-mask.

CACH DUNG
---------
    python patch_trim_mesh_pointcloud_mask.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "04_trim_mesh.py"

# Moi snippet la doc lap: (nhan, OLD, NEW). Ap dung tung cai, bao loi ro
# rang neu khong khop thay vi bo qua am tham.
SNIPPETS: list[tuple[str, str, str]] = []

# --- 1. Ham moi mask_by_trunk_pointcloud(), chen sau normal_ground_filter() --
SNIPPETS.append((
    "new_function_mask_by_trunk_pointcloud",
    '''    mesh_out = copy.deepcopy(mesh)
    mesh_out.remove_triangles_by_mask(ground_mask)
    mesh_out.remove_unreferenced_vertices()
    return mesh_out


def project_onto_axis(pts: np.ndarray, axis: np.ndarray,
                      origin: np.ndarray) -> np.ndarray:''',
    '''    mesh_out = copy.deepcopy(mesh)
    mesh_out.remove_triangles_by_mask(ground_mask)
    mesh_out.remove_unreferenced_vertices()
    return mesh_out


def mask_by_trunk_pointcloud(mesh: o3d.geometry.TriangleMesh,
                             verts: np.ndarray,
                             pointcloud_path: str,
                             radius: float = 0.05,
                             verbose: bool = True) -> tuple:
    """
    Loai bo vertex KHONG nam gan diem nao trong trunk_pointcloud.ply -- file
    nay da duoc 05_trunk_isolation.py xac dinh dung la than cay (KDTree
    camera-anchor + RANSAC ground removal + up-axis fix, xac nhan tren du
    lieu that -- xem CLAUDE.md muc 5/12). Dung no lam moc thay vi doan lai
    tu dau bang truc trong luong co dinh: khong gia dinh than cay thang
    tuyet doi theo Y (scan that co the nghieng), va loc theo TUNG VERTEX
    (khong theo connected component) nen khong bao gio vo tinh vut ca mot
    manh mesh that su la than cay chi vi no bi dut ket noi o do.
    Tra ve (mesh, verts, True) neu ap dung duoc; (mesh, verts, False) neu
    khong tim thay / pointcloud rong (goi noi dung nen fallback sang
    gravity_axis_trim).
    """
    if not os.path.exists(pointcloud_path):
        if verbose:
            print(f"  Trunk pointcloud mask: SKIP (khong tim thay {pointcloud_path})", flush=True)
        return mesh, verts, False

    ref_pcd = o3d.io.read_point_cloud(pointcloud_path)
    ref_pts = np.asarray(ref_pcd.points, dtype=np.float64)
    if len(ref_pts) == 0:
        if verbose:
            print(f"  Trunk pointcloud mask: SKIP (pointcloud rong: {pointcloud_path})", flush=True)
        return mesh, verts, False

    tree = cKDTree(ref_pts)
    dist, _ = tree.query(verts, k=1, workers=-1)
    keep_mask = dist <= radius

    n_remove = int((~keep_mask).sum())
    if verbose:
        print(f"  Trunk pointcloud mask (r={radius*100:.1f}cm, ref={len(ref_pts):,} pts): "
              f"remove {n_remove:,} / {len(verts):,} verts "
              f"({100*n_remove/max(len(verts),1):.1f}%)", flush=True)

    if n_remove == 0:
        return mesh, verts, True

    mesh_out = copy.deepcopy(mesh)
    mesh_out.remove_vertices_by_mask(~keep_mask)
    mesh_out.remove_unreferenced_vertices()
    mesh_out.remove_degenerate_triangles()
    verts_out = np.asarray(mesh_out.vertices, dtype=np.float64)
    return mesh_out, verts_out, True


def project_onto_axis(pts: np.ndarray, axis: np.ndarray,
                      origin: np.ndarray) -> np.ndarray:''',
))

# --- 2. process_scan(): them tham so mask_radius / no_pointcloud_mask --------
SNIPPETS.append((
    "process_scan_signature",
    '''def process_scan(scan_dir: str,
                 min_frag: int       = 1500,
                 max_radius: float   = None,    # None = tu dong
                 trim_start: float   = 0.0,     # them crop tu dau ngon (m)
                 trim_end: float     = 0.0,     # them crop tu goc (m)
                 info_only: bool     = False,
                 verbose: bool       = True) -> dict:''',
    '''def process_scan(scan_dir: str,
                 min_frag: int       = 1500,
                 max_radius: float   = None,    # None = tu dong
                 trim_start: float   = 0.0,     # them crop tu dau ngon (m)
                 trim_end: float     = 0.0,     # them crop tu goc (m)
                 mask_radius: float  = 0.05,    # ban kinh (m) moc trunk_pointcloud.ply
                 no_pointcloud_mask: bool = False,  # True = ep dung fallback truc trong luong
                 info_only: bool     = False,
                 verbose: bool       = True) -> dict:''',
))

# --- 3. Step 1b: thay gravity_axis_trim-truoc-PCA bang moc trunk_pointcloud --
SNIPPETS.append((
    "step_1b_pointcloud_mask",
    '''    verts  = np.asarray(mesh.vertices, dtype=np.float64)
    N      = len(verts)

    # ── Step 1b: Gravity-axis trim (xoa blob dat TRUOC khi PCA) ──────────────
    grav_down = estimate_gravity(scan_dir)
    if grav_down is not None:
        mesh, verts = gravity_axis_trim(mesh, verts, grav_down,
                                        n_slices=80,
                                        verbose=verbose)
        N = len(verts)
        # Normal-based filter: loai bo mat phang nam ngang (dat/la/muc gan mat dat)
        # Normal cua than cay ~ vuong goc Y (|ny| ~ 0), mat phang dat ~ |ny| ~ 1
        mesh = normal_ground_filter(mesh, grav_down, normal_thr=0.75, verbose=verbose)
        mesh, _ = keep_largest_component(mesh)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        N = len(verts)
    else:
        if verbose:
            print("  Gravity trim: skip (no odometry.csv)", flush=True)''',
    '''    verts  = np.asarray(mesh.vertices, dtype=np.float64)
    N      = len(verts)

    # ── Step 1b: Loai debris/dat bang moc trunk_pointcloud.ply (TRUOC khi PCA) ──
    # Xem docstring mask_by_trunk_pointcloud(): day la nguon tham chieu dang
    # tin cay hon heuristic hinh hoc cu (truc trong luong co dinh + component),
    # vi no khong gia dinh than cay thang tuyet doi theo Y va khong the "vo
    # tinh" bo mat mot manh that su la than cay chi vi mesh bi dut ket noi o do.
    pointcloud_path = os.path.join(out_dir, "trunk_pointcloud.ply")
    if no_pointcloud_mask:
        mask_ok = False
    else:
        mesh, verts, mask_ok = mask_by_trunk_pointcloud(
            mesh, verts, pointcloud_path, radius=mask_radius, verbose=verbose)
        N = len(verts)

    if not mask_ok:
        # Fallback (scan khong co trunk_pointcloud.ply, hoac --no-pointcloud-mask):
        # dung lai heuristic truc trong luong nhu truoc. Giu keep_largest_component
        # o day vi nhanh du phong nay khong co moc nao khac de "cuu" mot manh bi
        # dut ket noi. radial_factor noi tu 2.5 -> 3.5 de bot nhay cam voi nghieng.
        if verbose:
            print("  (fallback) khong co trunk_pointcloud.ply -- dung gravity_axis_trim", flush=True)
        grav_down = estimate_gravity(scan_dir)
        mesh, verts = gravity_axis_trim(mesh, verts, grav_down, n_slices=80,
                                        radial_factor=3.5, verbose=verbose)
        mesh = normal_ground_filter(mesh, grav_down, normal_thr=0.75, verbose=verbose)
        mesh, _ = keep_largest_component(mesh)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        N = len(verts)''',
))

# --- 4. CLI: them --mask-radius / --no-pointcloud-mask + noi vao run_batch ---
SNIPPETS.append((
    "cli_flags_and_call",
    '''def main():
    p = argparse.ArgumentParser(description="Trim mesh: xoa dat/ngon/fragments")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", metavar="NAME")
    g.add_argument("--all",  action="store_true")

    p.add_argument("--info",         action="store_true",
                   help="Chi hien thi thong tin bounds, khong cat")
    p.add_argument("--min-frag",     type=int,   default=1500,
                   help="Xoa component < N tris (default 1500)")
    p.add_argument("--max-radius",   type=float, default=None,
                   help="Xoa vert xa truc hon R met (auto neu bo trong)")
    p.add_argument("--trim-start",   type=float, default=0.0,
                   help="Cat them tu dau ngon (m, default 0)")
    p.add_argument("--trim-end",     type=float, default=0.0,
                   help="Cat them tu goc/dat (m, default 0)")
    p.add_argument("--raw-data",     default=RAW_DATA_DIR)
    p.add_argument("--skip-existing",action="store_true")

    args = p.parse_args()
    run_batch(args.raw_data,
              scan_filter   = args.scan if args.scan else None,
              skip_existing = args.skip_existing,
              min_frag      = args.min_frag,
              max_radius    = args.max_radius,
              trim_start    = args.trim_start,
              trim_end      = args.trim_end,
              info_only     = args.info)''',
    '''def main():
    p = argparse.ArgumentParser(description="Trim mesh: xoa dat/ngon/fragments")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", metavar="NAME")
    g.add_argument("--all",  action="store_true")

    p.add_argument("--info",         action="store_true",
                   help="Chi hien thi thong tin bounds, khong cat")
    p.add_argument("--min-frag",     type=int,   default=1500,
                   help="Xoa component < N tris (default 1500)")
    p.add_argument("--max-radius",   type=float, default=None,
                   help="Xoa vert xa truc hon R met (auto neu bo trong)")
    p.add_argument("--trim-start",   type=float, default=0.0,
                   help="Cat them tu dau ngon (m, default 0)")
    p.add_argument("--trim-end",     type=float, default=0.0,
                   help="Cat them tu goc/dat (m, default 0)")
    p.add_argument("--mask-radius",  type=float, default=0.05,
                   help="Ban kinh (m) coi mesh vertex la gan trunk_pointcloud.ply (default 0.05)")
    p.add_argument("--no-pointcloud-mask", action="store_true",
                   help="Bo qua trunk_pointcloud.ply, dung heuristic truc trong luong cu")
    p.add_argument("--raw-data",     default=RAW_DATA_DIR)
    p.add_argument("--skip-existing",action="store_true")

    args = p.parse_args()
    run_batch(args.raw_data,
              scan_filter        = args.scan if args.scan else None,
              skip_existing      = args.skip_existing,
              min_frag           = args.min_frag,
              max_radius         = args.max_radius,
              trim_start         = args.trim_start,
              trim_end           = args.trim_end,
              mask_radius        = args.mask_radius,
              no_pointcloud_mask = args.no_pointcloud_mask,
              info_only          = args.info)''',
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
        print(f"{TARGET_NAME} (trunk_pointcloud mask fix) da duoc va truoc do roi. Khong can lam gi them.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "de tranh lam hong file. Bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_pointcloudmask")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Da sao luu ban goc: {backup}")

    applied = []
    for label, old, new in SNIPPETS:
        if new in content:
            continue  # da co san (idempotent), bo qua rieng doan nay
        content = content.replace(old, new, 1)
        applied.append(label)

    target.write_text(content, encoding="utf-8")
    print(f"Da va thanh cong {len(applied)} doan (trunk_pointcloud mask fix): {applied}")
    print("Chay lai buoc trim mesh (buoc 6) de kiem tra ket qua, vd:")
    print(f'  python {TARGET_NAME} --scan <ten_scan> --raw-data <raw_data_dir>')


if __name__ == "__main__":
    main()
