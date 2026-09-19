#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_recolor_prefer_poisson.py
===============================
Va 03_recolor_mesh.py: uu tien trunk_mesh_poisson.ply (dau ra cua
02_mesh_poisson_trunk.py) lam nguon HINH HOC, tu dong lui ve
trunk_mesh_tsdf.ply neu khong co. Mau van lay tu trunk_pointcloud.ply nhu cu,
nen phan con lai cua pipeline khong doi.

BOI CANH (xem CLAUDE.md muc 12 item 15)
---------------------------------------
trunk_mesh_tsdf.ply co lo thung lon tren dung phan than cay DA DUOC QUET, vi
TSDF cua Open3D do khoang cach doc theo tia camera: o goc nhin xien theta,
khoang cach doc tia bi keo dai 1/cos(theta), nen voxel sat be mat roi ra ngoai
dai sdf_trunc va khong bao gio duoc tich hop. Goc nhin tot nhat tren phan be
mat bi mat la 52-58 do (p50), so voi 11-22 do tren phan con nguyen.

Da thu va loai tru 11 cau hinh TSDF (sdf_trunc 0.010-0.16, voxel 0.005/0.008,
conf_thresh 0/1, them bo loc "diem bay" o mep): trong pham vi than cay so vong
bien gan nhu khong doi trong khi ban kinh mesh phinh toi 2.44x. Cac lo do DU
LIEU quyet dinh, khong phai tham so.

Poisson khop mot mat vao dam diem thay vi hoa doc tia, nen dung duoc chinh
nhung diem TSDF phai bo -- va du lieu diem trong cac lo van day du (lop vo
2.7cm trong lo so voi 2.6-5.6cm vung lanh).

DO TREN 46 SCAN, Poisson (depth 7) tot hon TSDF o moi cay:
  - lo lon nhat: 14-75 dinh, so voi 374-1383 cua TSDF
  - phinh ban kinh: 0.98-1.12x, so voi 0.95-15.3x
  - dien tich / moc hinh tru (2*pi*r_med*h) sau lam muot Taubin:
    0.89-1.52x, so voi 0.35-2.89x cua TSDF
Con so cuoi quan trong nhat: 05_detect_lichen.py tinh total_cm2 tu dien tich
tam giac mesh, va TSDF dang do THIEU rat nhieu tren mot so cay (cay_0009 chi
0.35x moc tru = mat 65% be mat; cay_0013 0.57x; cay_0010 0.75x).

--no-poisson de ep dung lai mesh TSDF nhu truoc.

CACH DUNG
---------
    python patch_recolor_prefer_poisson.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "03_recolor_mesh.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- Site 1: process_scan signature ---------------------------------------
SNIPPETS.append((
    "process_scan_signature",
    '''def process_scan(scan_dir: str, verbose: bool = True) -> dict:''',
    '''def process_scan(scan_dir: str, verbose: bool = True,
                 prefer_poisson: bool = True) -> dict:''',
))

# --- Site 2: choose the geometry source -----------------------------------
SNIPPETS.append((
    "prefer_poisson_source",
    '''    mesh_path = os.path.join(out_dir, "trunk_mesh_tsdf.ply")
    pcd_path  = os.path.join(out_dir, "trunk_pointcloud.ply")

    if not os.path.exists(mesh_path):
        result["msg"] = "trunk_mesh_tsdf.ply not found — chay 02_mesh_tsdf.py truoc"
        return result''',
    '''    # [Sep 2026] Uu tien mesh Poisson dung tu trunk_pointcloud.ply. Mesh TSDF
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
              flush=True)''',
))

# --- Site 3: run_batch signature ------------------------------------------
SNIPPETS.append((
    "run_batch_signature",
    """def run_batch(raw_data_dir, scan_filter=None, skip_existing=False):""",
    """def run_batch(raw_data_dir, scan_filter=None, skip_existing=False,
              prefer_poisson=True):""",
))

# --- Site 4: run_batch forwards the flag ----------------------------------
SNIPPETS.append((
    "run_batch_call",
    '''        r = process_scan(scan_dir, verbose=True)''',
    '''        r = process_scan(scan_dir, verbose=True, prefer_poisson=prefer_poisson)''',
))

# --- Site 5: CLI -----------------------------------------------------------
SNIPPETS.append((
    "cli_no_poisson",
    '''    p.add_argument("--raw-data",      default=RAW_DATA_DIR)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    run_batch(args.raw_data,
              scan_filter   = args.scan if args.scan else None,
              skip_existing = args.skip_existing)''',
    '''    p.add_argument("--raw-data",      default=RAW_DATA_DIR)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-poisson", action="store_true",
                   help="Dung mesh TSDF nhu truoc thay vi trunk_mesh_poisson.ply "
                        "(xem CLAUDE.md muc 12 item 15)")
    args = p.parse_args()
    run_batch(args.raw_data,
              scan_filter    = args.scan if args.scan else None,
              skip_existing  = args.skip_existing,
              prefer_poisson = not args.no_poisson)''',
))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if all(new in content for _, _, new in SNIPPETS):
        print(f"{TARGET_NAME} (prefer poisson) da duoc va truoc do roi.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_preferpoisson")
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
    print(f"Da va thanh cong {len(applied)} doan (prefer poisson): {applied}")
    print("Chay:")
    print("  python 02_mesh_poisson_trunk.py --scan <NAME>")
    print("  python 03_recolor_mesh.py --scan <NAME>")
    print("Quay ve mesh TSDF: python 03_recolor_mesh.py --scan <NAME> --no-poisson")


if __name__ == "__main__":
    main()
