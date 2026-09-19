#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_finalize_mesh_sparse_only_ragged.py
============================================
Va 04b_finalize_mesh.py: sua trim_ragged_top() chi dung tieu chi THUA THAT
SU (slice_total < min_slice_verts) de quyet dinh cat "ngon rach", bo 2 tieu
chi density/sig-hole cu (chung do "co nhieu lo bien nho" chu khong phai
"chua scan toi").

Da xac nhan tren scan that (cay_0007_1805) qua histogram + bang per-slice
that: tieu chi sparse KHONG BAO GIO kich hoat (moi lat cat co 649-7439
vertex, xa nguong 20) -- trong khi density/sig-hole kich hoat gan nhu MOI
lat cat trong nua tren than cay (lat 7-24 / 25), khien thuat toan khong
bao gio gom du 6 lat sach lien tiep va roi vao muc kep an toan (cat mat
50% chieu dai). Ngay ca noi long 10x (sig_hole_verts 10->100, ragged_density
0.05->0.20) van chi cai thien tu 50% len 53% giu duoc.

Ly do goc re: nua tren than cay nay co mot cum rong/dia y (thay trong anh
chup thuc te) -- be mat rong/dia y go ghe hon vo cay thuong, khien TSDF
dung hinh kem min hon va sinh nhieu lo nho tu nhien hon o CHINH VUNG NAY --
day la thuoc tinh vat ly binh thuong cua vung co dia y, KHONG PHAI dau hieu
"chua quet toi". Trong khi do day la vung QUAN TRONG NHAT can do luong. Tieu
chi density/sig-hole cu vo tinh coi "co dia y" la "rach", xoa mat chinh du
lieu can thu thap.

Tieu chi sparse (mat do diem qua thap trong ca lat cat) moi la tin hieu
dung cho "camera chua quet toi" (vd nguoi dung dung quet som) -- va da xac
nhan tren du lieu that la khong bao gio kich hoat sai tren mot scan binh
thuong, dong thoi van san sang bat neu that su co vung gan nhu khong co
diem nao.

CACH DUNG
---------
    python patch_finalize_mesh_sparse_only_ragged.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_NAME = "04b_finalize_mesh.py"

SNIPPETS: list[tuple[str, str, str]] = []

SNIPPETS.append((
    "is_ragged_sparse_only",
    '''    def _is_ragged(i):
        lo, hi = edges[i], edges[i + 1]
        if slice_total[i] < min_slice_verts:
            return True   # thua
        if slice_density[i] > ragged_density:
            return True   # density cao
        # Co loop dang ke (>= sig_hole_verts) trong slice nay
        for lp, ch in loop_info:
            if lo <= ch < hi and len(lp) >= sig_hole_verts:
                return True
        return False

    if verbose:
        n_sig = sum(1 for lp, _ in loop_info if len(lp) >= sig_hole_verts)
        print(f"  Top trim: top {top_fraction*100:.0f}%  "
              f"{len(loop_info)} loops  {n_sig} significant (>={sig_hole_verts}v)",
              flush=True)''',
    '''    def _is_ragged(i):
        # [Sep 2026] CHI dung tieu chi THUA THAT SU (qua it vertex trong ca
        # lat cat) -- day moi la dau hieu dung cua "camera chua quet toi".
        # KHONG con dung density/sig-hole cua boundary loop nua: da xac nhan
        # tren du lieu that (cay_0007_1805, xem CLAUDE.md Sec12) rang vo cay
        # o vung co rong/dia y tu nhien co RAT NHIEU lo bien nho-vua rai rac
        # (218 loop, 93 loop >=10v chi trong top 65%) -- day la thuoc tinh
        # vat ly binh thuong cua be mat gan-gui/dia y (TSDF kem min hon),
        # KHONG PHAI dau hieu thieu du lieu -- va trung ngay vao vung quan
        # trong nhat can do luong. Tieu chi cu (density/sig-hole) coi gan
        # nua tren than cay la "rach" va cat mat, ngay ca khi noi long
        # nguong len 10 lan. Tieu chi sparse xac nhan khong bao gio kich
        # hoat sai tren scan binh thuong (moi lat cat that co hang tram-
        # hang nghin vertex, xa nguong min_slice_verts).
        if slice_total[i] < min_slice_verts:
            return True   # thua that su -- dau hieu dung cua "chua quet toi"
        return False

    if verbose:
        n_sig = sum(1 for lp, _ in loop_info if len(lp) >= sig_hole_verts)
        print(f"  Top trim: top {top_fraction*100:.0f}%  "
              f"{len(loop_info)} loops  {n_sig} significant (>={sig_hole_verts}v)  "
              f"[info only -- cut decision uses sparse-slice check only, see CLAUDE.md Sec12]",
              flush=True)''',
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
        print(f"{TARGET_NAME} (sparse-only ragged-top fix) da duoc va truoc do roi. Khong can lam gi them.")
        return

    missing = [label for label, old, new in SNIPPETS
               if new not in content and old not in content]
    if missing:
        print(f"KHONG KHOP: khong tim thay dung doan code can thay cho: {missing}")
        print("Co the file da bi sua tay truoc do. Dung lai, khong sua gi, "
              "de tranh lam hong file. Bao lai cho tro ly de kiem tra thu cong.")
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_sparseonlyragged")
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
    print(f"Da va thanh cong {len(applied)} doan (sparse-only ragged-top fix): {applied}")
    print("Chay lai buoc finalize mesh (buoc 7) de kiem tra ket qua, vd:")
    print(f'  python {TARGET_NAME} --scan <ten_scan> --raw-data <raw_data_dir>')


if __name__ == "__main__":
    main()
