#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trunk_isolation_view_anchor.py
====================================
Va 05_trunk_isolation.py: xac dinh CAY DICH bang GIAO DIEM CAC TIA NHIN cua
camera, thay cho estimate_trunk_axis() (mot trung vi (X,Z) tren 60% diem cao
nhat). Day la lan dau tien pipeline co mot co che thuc su tra loi cau hoi
"cay nao la cay dich" -- truoc do moi thu deu la bien pho.

BOI CANH (8 scan hong: cay_0032/0038/0040/0042/0044/0046/0048/0051)
-------------------------------------------------------------------
trunk_pointcloud.ply cua ca 8 ra mot khoi bet nam cheo chi cao 0.43-0.77m
(PCA lech 28-60 do khoi world Y, L1/L2 chi 1.3-3.1; cay that: 35-570).

DA DO VA LOAI TRU "loi khi quay": du lieu tho cua ca 8 hoan toan binh thuong
-- quet doc 1.88-2.20m (cay tot: 1.92-2.23m), phu phuong vi 100% voi khe ho
chi 2-8 do, ban kinh vong quay 0.27-0.42m, 3.2-7.0 trieu diem quanh quy dao,
dam may gan quy dao cao 2.77-3.84m. Nguoi quay DA di tron vong. (De so sanh:
cay_0003_1805 moi that su la loi khi quay -- quet doc chi 1.04m.)

NGUYEN NHAN THAT (tu trace tung chang):
  estimate_trunk_axis() lay MOT trung vi (X,Z) tren 60% diem cao nhat. Khi
  trong khung co tu hai cay tro len, trung vi roi vao cho tuy y. Do duoc:
    scan      truc tho lech moc   diem quanh CAY DUNG   quanh cay BI CHON
    0051           97.3cm               44,817               10,817
    0040           93.3cm               43,621               36,092
    0044           95.7cm               52,792               42,995
    0046           85.2cm               54,651               54,325
    0038           60.1cm               67,105               29,850
    0032           55.2cm               43,460               43,987
    0048           53.3cm               38,736               14,216
    0042           46.1cm               76,897               29,199
  Tren cay_0051 cay that co nhieu diem GAP 4 LAN cay bi chon ma van thua, nen
  KHONG mot luat dem diem nao cuu duoc -- ke ca max_radius_ratio hay
  keep_largest_cluster(). Moi buoc sau do bam trung thanh vao truc sai (ket
  qua cuoi chi cach truc sai 3-12cm).

MOC MOI (do tu du lieu, khong doan)
-----------------------------------
Nguoi quay CHIA CAMERA VAO CAY DICH suot buoi (CLAUDE.md muc 4), nen giao
diem binh phuong toi thieu cua moi tia nhin ngang chinh la cay do. Huong nhin
la R @ (0,0,+1) voi R = quaternion cua odometry.csv -- quy uoc nay da duoc
xac lap doc lap trong patch_slope_analysis_gravity_frame.py (R la
camera->world, khop pose_to_extrinsic() trong 02_pointcloud_and_mesh.py; tia
+Z chi vao than cay voi sai so 3-6 do).

Do tren toan bo 49 scan co odometry.csv:
  - sai so moc so voi than cay that (37 scan chay tot): 0.00-0.08m, phan lon
    duoi 0.02m. So sanh: y tuong "tam quy dao camera" da tung duoc do va LOAI
    BO vi sai 8.7-48.6cm.
  - tren 8 scan hong: moc cach ket qua cua pipeline 0.48-1.04m -- tuc moc chi
    dung cho nguoi, pipeline di lac.
  - do tin cay cua chinh moc, tren CA 49 scan: residual tia 0.018-0.058m; ty
    le tri rieng nho/lon cua ma tran 0.654-0.973 (khong bao gio suy bien);
    moc nam PHIA TRUOC 97-100% camera (xac nhan quy uoc +Z tren toan kho).
  - tren scan dang chay tot, truc trung vi cu va moc chi cach nhau 0.8cm
    (cay_0010) va 1.8cm (cay_0029), nen thay the gan nhu khong doi gi o cac
    cay da dung.

KET QUA MO PHONG TRUOC KHI VA (chay lai pipeline that, chi thay moc):
    cay_0032: 0.43m/59.5deg/L1L2 1.3  ->  2.23m/ 4.2deg/L1L2 238.6
    cay_0038: 0.51m/55.3deg/L1L2 1.9  ->  2.69m/ 7.1deg/L1L2  64.0
    cay_0040: 0.77m/43.5deg/L1L2 1.4  ->  2.12m/10.1deg/L1L2 198.6

BA CHOT AN TOAN, nguong lay tu 49 so do tren (deu de bien >=4 lan):
  max_ray_residual 0.25m (do duoc <=0.058), min_conditioning 0.15 (do duoc
  >=0.654), min_front_fraction 0.60 (do duoc >=0.97). Vi pham bat ky chot nao
  -> in canh bao va quay ve estimate_trunk_axis() cu, thay vi im lang dung
  mot moc sai.

Tat bang --no-view-anchor.

CACH DUNG
---------
    python patch_trunk_isolation_view_anchor.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_trunk_isolation.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- Site 1: the new view-ray anchor function ------------------------------
SNIPPETS.append((
    "view_anchor_function",
    '''# ---------------------------------------------------------------------------
# 2. Trunk Axis Estimation
# ---------------------------------------------------------------------------

def estimate_trunk_axis(''',
    '''# ---------------------------------------------------------------------------
# 2. Trunk Axis Estimation
# ---------------------------------------------------------------------------

def estimate_target_axis_from_views(
    dataset_dir: str,
    min_poses: int = 20,
    max_ray_residual: float = 0.25,
    min_conditioning: float = 0.15,
    min_front_fraction: float = 0.60,
    verbose: bool = True,
) -> "tuple[float, float] | None":
    """
    Xac dinh CAY DICH bang giao diem cac TIA NHIN cua camera.

    Nguoi quay chia camera vao cay dich suot buoi (CLAUDE.md muc 4), nen giao
    diem binh phuong toi thieu cua moi tia nhin ngang chinh la cay do. Day la
    co che DUY NHAT trong pipeline thuc su tra loi "cay nao la cay dich":
    estimate_trunk_axis() chi lay mot trung vi (X,Z), nen khi trong khung co
    tu hai cay tro len no roi vao cho tuy y -- do duoc lech toi 97cm tren 8
    scan, ngay ca khi cay that co nhieu diem gap 4 lan cay bi chon.

    Huong nhin la R @ (0,0,+1) voi R = quaternion cua odometry.csv (R la
    camera->world; xem pose_to_extrinsic() trong 02_pointcloud_and_mesh.py va
    patch_slope_analysis_gravity_frame.py).

    Tra ve (cx, cz), hoac None neu bat ky chot an toan nao khong dat -- khi do
    goi noi nen dung lai estimate_trunk_axis(). Nguong lay tu so do that tren
    49 scan, moi nguong de bien it nhat 4 lan (xem docstring cua
    patch_trunk_isolation_view_anchor.py).
    """
    import csv as _csv

    odo_path = os.path.join(dataset_dir, "odometry.csv")
    if not os.path.exists(odo_path):
        if verbose:
            print("  Neo huong nhin: BO QUA (khong co odometry.csv)")
        return None

    pos, quat = [], []
    try:
        with open(odo_path, "r", newline="") as f:
            for row in _csv.DictReader(f, skipinitialspace=True):
                try:
                    pos.append([float(row["x"]), float(row["z"])])
                    quat.append([float(row["qx"]), float(row["qy"]),
                                 float(row["qz"]), float(row["qw"])])
                except (KeyError, ValueError):
                    continue
    except Exception as e:
        if verbose:
            print(f"  Neo huong nhin: BO QUA (khong doc duoc odometry.csv: {e})")
        return None

    if len(pos) < min_poses:
        if verbose:
            print(f"  Neo huong nhin: BO QUA (chi co {len(pos)} pose)")
        return None

    o = np.asarray(pos, dtype=float)
    q = np.asarray(quat, dtype=float)

    # Huong nhin +Z cua tung pose, chieu xuong mat phang ngang (X, Z).
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    dx = 2.0 * (qx * qz + qy * qw)
    dz = 1.0 - 2.0 * (qx * qx + qy * qy)
    d = np.stack([dx, dz], axis=1)
    nrm = np.linalg.norm(d, axis=1)
    keep = np.isfinite(nrm) & (nrm > 1e-6)
    if keep.sum() < min_poses:
        if verbose:
            print("  Neo huong nhin: BO QUA (huong nhin gan nhu thang dung)")
        return None
    o, d = o[keep], d[keep] / nrm[keep, None]

    # Binh phuong toi thieu: diem gan nhat toi moi duong thang (o_i, d_i).
    # A = sum (I - d d^T),  b = sum (I - d d^T) o
    outer = np.einsum("ni,nj->nij", d, d)
    M = np.eye(2)[None, :, :] - outer
    A = M.sum(axis=0)
    b = np.einsum("nij,nj->i", M, o)

    ev = np.linalg.eigvalsh(A)
    cond = float(ev.min() / max(ev.max(), 1e-12))
    if not np.isfinite(cond) or cond < min_conditioning:
        if verbose:
            print(f"  Neo huong nhin: BO QUA (cac tia gan nhu song song, "
                  f"dieu kien {cond:.3f} < {min_conditioning})")
        return None

    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        if verbose:
            print("  Neo huong nhin: BO QUA (he phuong trinh suy bien)")
        return None

    perp = np.linalg.norm(np.einsum("nij,nj->ni", M, p[None, :] - o), axis=1)
    resid = float(np.median(perp))
    front = float(np.mean(np.einsum("ni,ni->n", p[None, :] - o, d) > 0))

    if resid > max_ray_residual:
        if verbose:
            print(f"  Neo huong nhin: BO QUA (cac tia khong hoi tu, "
                  f"residual {resid:.3f}m > {max_ray_residual}m)")
        return None
    if front < min_front_fraction:
        if verbose:
            print(f"  Neo huong nhin: BO QUA (moc nam phia sau camera, "
                  f"chi {front:.0%} < {min_front_fraction:.0%} nhin ve phia no)")
        return None

    if verbose:
        print(f"  Neo huong nhin camera: XZ=({p[0]:.3f}, {p[1]:.3f}) tu "
              f"{int(keep.sum())} pose [residual={resid*100:.1f}cm, "
              f"dieu kien={cond:.2f}, phia truoc={front:.0%}]")
    return float(p[0]), float(p[1])


def estimate_trunk_axis(''',
))

# --- Site 2: use the anchor in isolate_trunk -------------------------------
SNIPPETS.append((
    "isolate_trunk_use_view_anchor",
    '''    # Step 2: Ước tính trục thân cây
    cx, cz = estimate_trunk_axis(pcd, upper_fraction=0.6)''',
    '''    # Step 2: Xac dinh CAY DICH.
    # [Sep 2026] estimate_trunk_axis() chi lay MOT trung vi (X,Z) tren 60%
    # diem cao nhat, nen khi trong khung co tu hai cay tro len no roi vao cho
    # tuy y -- do duoc lech toi 97cm tren 8 scan, va tren cay_0051 cay that co
    # nhieu diem GAP 4 LAN cay bi chon ma van thua (nen khong luat dem diem
    # nao cuu duoc). Moc dang tin cay la giao diem cac TIA NHIN cua camera:
    # sai 0.00-0.08m tren 37 scan chay tot, va chi cach truc trung vi cu
    # 0.8-1.8cm o cac scan von da dung. Xem docstring
    # estimate_target_axis_from_views() + CLAUDE.md muc 14.
    cx, cz = estimate_trunk_axis(pcd, upper_fraction=0.6)
    if view_anchor:
        _anchor = estimate_target_axis_from_views(dataset_dir)
        if _anchor is not None:
            _ax, _az = _anchor
            _d = float(np.hypot(_ax - cx, _az - cz))
            if _d > 0.30:
                print(f"  CANH BAO: truc trung vi ({cx:.3f}, {cz:.3f}) cach moc "
                      f"huong nhin {_d*100:.0f}cm -- gan nhu chac chan no dang "
                      f"chi sang CAY KHAC. Dung moc huong nhin.")
            else:
                print(f"  (truc trung vi cach moc huong nhin {_d*100:.1f}cm)")
            cx, cz = _ax, _az''',
))

# --- Site 3: isolate_trunk signature ---------------------------------------
SNIPPETS.append((
    "isolate_trunk_signature_view_anchor",
    '''    seed_max_offset: float = 0.40,
    max_radius_ratio: float = 3.0,
) -> bool:''',
    '''    seed_max_offset: float = 0.40,
    max_radius_ratio: float = 3.0,
    view_anchor: bool = True,
) -> bool:''',
))

# --- Site 4: CLI -----------------------------------------------------------
SNIPPETS.append((
    "cli_no_view_anchor",
    '''    parser.add_argument(
        "--max-radius-ratio", type=float, default=3.0,''',
    '''    parser.add_argument(
        "--no-view-anchor", action="store_true",
        help="Do not locate the target tree from the camera viewing rays; fall back "
             "to the old median-of-upper-points trunk axis (see CLAUDE.md 12/14)"
    )
    parser.add_argument(
        "--max-radius-ratio", type=float, default=3.0,''',
))

SNIPPETS.append((
    "cli_call_view_anchor",
    '''            max_radius_ratio=args.max_radius_ratio,
        )''',
    '''            max_radius_ratio=args.max_radius_ratio,
            view_anchor=not args.no_view_anchor,
        )''',
))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / TARGET_NAME

    if not target.exists():
        print(f"KHONG TIM THAY: {target}")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if all(new in content for _, _, new in SNIPPETS):
        print(f"{TARGET_NAME} (view-ray target anchor) da duoc va truoc do roi.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Luu y: patch nay yeu cau patch_trunk_isolation_adaptive_crop.py, "
              "patch_trunk_isolation_adaptive_seed_gate.py va "
              "patch_trunk_isolation_slice_radius_ratio.py da duoc ap dung truoc.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_viewanchor")
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
    print(f"Da va thanh cong {len(applied)} doan (view-ray target anchor): {applied}")
    print("Chay lai buoc trunk isolation (buoc 3), vd:")
    print(f'  python {TARGET_NAME} <scan_dir>')
    print("Tat moc nay: --no-view-anchor")


if __name__ == "__main__":
    main()
