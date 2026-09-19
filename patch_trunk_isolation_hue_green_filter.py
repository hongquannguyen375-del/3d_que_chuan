#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_trunk_isolation_hue_green_filter.py
============================================
Va 05_trunk_isolation.py: them tieu chi Hue+Saturation vao
remove_green_vegetation() (dang OR voi tieu chi RGB-diff cu), de bat duoc
reu/dia y nhat mau con sot lai ma bo loc cu bo sot.

BOI CANH (do tren du lieu that, cay_0007_1805, xem CLAUDE.md muc 12):
Sau khi patch per-slice declutter xu ly xong canh cay, nguoi dung phat hien
van con cac dam mau xanh (reu/dia y) dinh tren be mat than cay trong
trunk_pointcloud.ply. Da loai 2 gia thuyet hinh hoc truoc khi ket luan day
la van de mau sac:
  1. "Doi tieu chi chon blob tu diem sang ban kinh" -- SAI: o 3 lat bat dong
     y kien giua 2 tieu chi, ben "thang theo ban kinh" luon it diem hon va
     cv (do khong tron) cao hon -- tuc la cum nhieu/thua, khong phai than
     cay that to hon.
  2. "Loc theo do tron/dac (cv)" -- SAI: cv dao dong 0.3-0.6 rai rac khap
     toan than, khong co ranh gioi ro giua lat sach va lat co dam xanh --
     vo cay that von da go ghe, do tron nen tang da cao san.
  3. remove_green_vegetation() cu chi xoa 151/90328 diem (0.2%) -- qua it
     de la mot khoi vat the tach roi (neu vay DBSCAN/declutter da bat
     duoc). Ket luan: dam xanh nam AP SAT be mat than (cung ban kinh, cung
     cum lien thong), khong co tin hieu hinh hoc nao de loc -- chi con mau
     sac la dac diem phan biet.

DO TRUC TIEP tren trunk_pointcloud.ply that: dung tieu chi Hue+Saturation
(H trong [65,150] do, S>0.25 -- CONG THUC Y HET is_vegetation trong
05_detect_lichen.py's classify_lichen_local_contrast()) de tim ung vien
"trong giong thuc vat". Ket qua: 6473/80332 diem (8.06%) khop, tap trung
thanh tung dai chieu cao lien tuc (khop voi vi tri dam xanh trong anh chup
thuc te cua nguoi dung, khong rai rac ngau nhien). Mau sac thuc te:
  - Dam xanh (nghi ngo):  H=79.2 do (trung binh), S=0.533, g-r=0.059, g-b=0.200
  - Vo cay (nen):         H=47.9 do,               S=0.237, g-r=-0.035, g-b=0.051
Hue va Saturation tach biet ro rang. NHUNG kenh g-r cua dam xanh (0.059)
chi bang 1/3 nguong hien tai (0.15) -- day chinh la nut that co chai khien
bo loc RGB-diff cu (yeu cau CA g-r>0.15 VA g-b>0.15) gan nhu khong bao gio
kich hoat, du g-b (0.200) thua suc vuot nguong. Quet nguong RGB-diff thap
hon khong giai duoc: ha xuong 0.06 chi bat duoc ~49% dam xanh voi 82% do
chinh xac; ha them nua thi bat nham vo cay ngay cang nhieu (0.02 -> chi con
58.6% chinh xac). Ket luan: RGB-diff khong phai cong thuc phu hop cho loai
mau nhat nay -- can Hue+Saturation.

FIX: them dieu kien Hue+Saturation vao remove_green_vegetation() theo kieu
OR voi dieu kien RGB-diff cu (giu nguyen dieu kien cu de khong mat kha nang
bat mau xanh manh, ro net khac). Tai su dung dung cong thuc da duoc kiem
chung o noi khac trong repo (05_detect_lichen.py), khong bay ra tieu chi
moi chua kiem chung.

CACH DUNG
---------
    python patch_trunk_isolation_hue_green_filter.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "05_trunk_isolation.py"

SNIPPETS: list[tuple[str, str, str]] = []

# --- Site 1: remove_green_vegetation() -- add Hue+Saturation OR condition --
SNIPPETS.append((
    "remove_green_vegetation_hue_sat",
    '''def remove_green_vegetation(
    pcd: o3d.geometry.PointCloud,
    green_threshold: float = 0.15,
) -> o3d.geometry.PointCloud:
    """
    Loại điểm có màu xanh lá cây mạnh (lá, rêu, cỏ).
    green_threshold: mức độ "xanh hơn đỏ và xanh dương" để lọc.
    """
    if not pcd.has_colors():
        return pcd

    colors = np.asarray(pcd.colors)  # [0,1] RGB
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]

    # Giữ điểm KHÔNG phải xanh lá đậm
    not_green = ~((g > r + green_threshold) & (g > b + green_threshold))
    kept = pcd.select_by_index(np.where(not_green)[0])
    removed = len(pcd.points) - len(kept.points)
    print(f"  Green filter: removed {removed} pts ({100*removed/max(len(pcd.points),1):.1f}%)")
    return kept''',
    '''def remove_green_vegetation(
    pcd: o3d.geometry.PointCloud,
    green_threshold: float = 0.15,
    use_hue_filter: bool = True,
    veg_hue_min: float = 65.0,
    veg_hue_max: float = 150.0,
    veg_sat_min: float = 0.25,
) -> o3d.geometry.PointCloud:
    """
    Loai diem co mau xanh la cay (la, reu, dia y nhat mau, co).
    Ket hop 2 tieu chi bang OR:
      1. RGB-diff cu (green_threshold): g > r+thr VA g > b+thr -- bat mau
         xanh manh, ro net.
      2. [Sep 2026] Hue+Saturation moi (use_hue_filter): H trong
         [veg_hue_min, veg_hue_max] do va S > veg_sat_min -- CONG THUC Y
         HET is_vegetation trong 05_detect_lichen.py's
         classify_lichen_local_contrast(). Xem CLAUDE.md muc 12: xac nhan
         tren du lieu that (cay_0007_1805) rang reu/dia y nhat mau (do anh
         sang lom dom qua tan la) co kenh g-r chi ~0.06 -- duoi xa nguong
         0.15 -- khien tieu chi (1) gan nhu khong bao gio kich hoat (chi
         151/90328 diem, 0.2%), du Hue (79 do) va Saturation (0.53) cua
         chung van tach biet ro voi vo cay (47 do, 0.24). Ha thap
         green_threshold khong giai duoc goc re (quet nguong tren du lieu
         that: 0.06 chi bat ~49% voi 82% chinh xac, thap hon nua bat nham
         vo cay ngay cang nhieu) -- vi RGB-diff khong phai phep do phu hop
         cho loai mau nhat nay.
    """
    if not pcd.has_colors():
        return pcd

    colors = np.asarray(pcd.colors)  # [0,1] RGB
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]

    # Tieu chi 1: RGB-diff cu (mau xanh manh, ro net)
    is_green_rgb = (g > r + green_threshold) & (g > b + green_threshold)

    is_green_hue = np.zeros(len(r), dtype=bool)
    if use_hue_filter:
        # Tieu chi 2: Hue+Saturation (bat reu/dia y nhat mau hon)
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        delta = maxc - minc
        sat = np.where(maxc > 1e-9, delta / np.maximum(maxc, 1e-9), 0.0)
        hue = np.zeros_like(r)
        dnz = delta > 1e-9
        mask_r = (maxc == r) & dnz
        mask_g = (maxc == g) & dnz
        mask_b = (maxc == b) & dnz
        hue[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
        hue[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
        hue[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
        is_green_hue = (hue >= veg_hue_min) & (hue <= veg_hue_max) & (sat > veg_sat_min)

    is_green = is_green_rgb | is_green_hue
    not_green = ~is_green
    kept = pcd.select_by_index(np.where(not_green)[0])
    removed = len(pcd.points) - len(kept.points)
    n_rgb = int(is_green_rgb.sum())
    n_hue_only = int((is_green_hue & ~is_green_rgb).sum())
    print(f"  Green filter: removed {removed} pts ({100*removed/max(len(pcd.points),1):.1f}%) "
          f"[rgb-diff={n_rgb}, hue-sat-only={n_hue_only}]")
    return kept''',
))

# --- Site 2: isolate_trunk() signature: add hue_green_filter flag ----------
SNIPPETS.append((
    "isolate_trunk_signature_hue",
    '''def isolate_trunk(
    dataset_dir: str,
    trunk_radius: float = 0.30,
    remove_green: bool = True,
    use_dbscan: bool = True,
    declutter_slices: bool = True,
) -> bool:''',
    '''def isolate_trunk(
    dataset_dir: str,
    trunk_radius: float = 0.30,
    remove_green: bool = True,
    use_dbscan: bool = True,
    declutter_slices: bool = True,
    hue_green_filter: bool = True,
) -> bool:''',
))

# --- Site 3: call site -- thread hue_green_filter into remove_green_vegetation ---
SNIPPETS.append((
    "call_remove_green_vegetation_with_hue_flag",
    '''    if remove_green:
        pcd = remove_green_vegetation(pcd)''',
    '''    if remove_green:
        pcd = remove_green_vegetation(pcd, use_hue_filter=hue_green_filter)''',
))

# --- Site 4: CLI flag + threading ------------------------------------------
SNIPPETS.append((
    "cli_flag_no_hue_green_filter",
    '''    parser.add_argument(
        "--no-slice-declutter", action="store_true",
        help="Disable per-slice off-axis cluster removal (branch/neighbor-object filter)"
    )
    args = parser.parse_args()

    if args.batch:
        run_batch(args.dataset, trunk_radius=args.trunk_radius)
    else:
        ok = isolate_trunk(
            args.dataset,
            trunk_radius=args.trunk_radius,
            remove_green=not args.no_green_filter,
            use_dbscan=not args.no_dbscan,
            declutter_slices=not args.no_slice_declutter,
        )
        if not ok:
            sys.exit(1)''',
    '''    parser.add_argument(
        "--no-slice-declutter", action="store_true",
        help="Disable per-slice off-axis cluster removal (branch/neighbor-object filter)"
    )
    parser.add_argument(
        "--no-hue-green-filter", action="store_true",
        help="Disable Hue+Saturation vegetation check in green filter (RGB-diff rule only)"
    )
    args = parser.parse_args()

    if args.batch:
        run_batch(args.dataset, trunk_radius=args.trunk_radius)
    else:
        ok = isolate_trunk(
            args.dataset,
            trunk_radius=args.trunk_radius,
            remove_green=not args.no_green_filter,
            use_dbscan=not args.no_dbscan,
            declutter_slices=not args.no_slice_declutter,
            hue_green_filter=not args.no_hue_green_filter,
        )
        if not ok:
            sys.exit(1)''',
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
        print(f"{TARGET_NAME} (hue+sat green filter) da duoc va truoc do roi. Khong can lam gi them.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "de tranh lam hong file. Bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_huegreenfilter")
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
    print(f"Da va thanh cong {len(applied)} doan (hue+sat green filter): {applied}")
    print("Chay lai buoc trunk isolation (buoc 3) de kiem tra ket qua, vd:")
    print(f'  python {TARGET_NAME} <scan_dir>')
    print("De tat rieng tieu chi Hue/Sat (so sanh/rollback): them --no-hue-green-filter")


if __name__ == "__main__":
    main()
