"""
pipeline_io.py -- tien ich ghi file summary cap corpus cho toan bo pipeline.

LY DO TON TAI (Sep 2026)
------------------------
Moi script trong pipeline co mot `run_batch(raw_data_dir, scan_filter=None, ...)`
ghi mot file `<buoc>_summary.json` o thu muc Raw_data. Nhung `--scan X` cung di
qua chinh ham do, chi khac la danh sach thu muc bi loc con MOT cay -- roi van
ghi de file summary bang dung mot dong do.

Ket qua do duoc tren corpus that (D:\\Backup\\Thucdia-18May2026, 20/09/2026):

    poisson_summary.json    46 dong   <- con nguyen (lan chay --all cuoi cung)
    slope_summary.json      46 dong   <- 06_slope_analysis.py chi ghi khi --batch
    recolor_summary.json     1 dong   <- bi xoa so
    trim_summary.json        1 dong   <- bi xoa so
    finalize_summary.json    1 dong   <- bi xoa so
    lichen_summary.json      1 dong   <- bi xoa so (file chua so dia y toan corpus)

Bon file tren bi rut ve dung `cay_0004_1805` sau vai lan chay thu MOT cay. Khong
mat vinh vien (moi cay van giu `output/lichen_stats.json` rieng) nhung ban ghi
cap corpus thi mat, va do la thu duoc doc khi so sanh giua cac cay.

Module nay de trong mot file rieng thay vi nhet vao `geometry_utils.py`: module
do la quy uoc truc toa do, viec khac han. Ly do tach van y nhu ly do
`geometry_utils.py` ra doi -- de quy uoc khong am tham phan ky giua cac script
(CLAUDE.md muc 5).

CACH DUNG
---------
    from pipeline_io import merged_results, dump_summary

    dump_summary(path, {"depth": depth,
                        "results": merged_results(path, results)})

Voi `05_detect_lichen.py`, summary la mot list tran va khoa dinh danh la "scan":

    dump_summary(path, merged_results(path, rows, id_key="scan"))
"""

import json
import os

__all__ = ["merged_results", "dump_summary", "load_overrides", "save_overrides"]


def _read_json(path):
    """Doc JSON, tra ve None neu thieu file hoac file hong."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError, UnicodeDecodeError):
        # File hong thi coi nhu chua co. Khong nem loi: mat ban ghi cu con hon
        # lam hong ca lan chay dang co ket qua tot.
        return None


def _old_rows(summary_path):
    """Lay danh sach ket qua tu file summary cu, chiu ca hai hinh dang dang co.

    - list tran                  -> `lichen_summary.json`
    - {"results": [...], ...}    -> bon file con lai
    """
    data = _read_json(summary_path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    return []


def merged_results(summary_path, new_results, id_key="name", fresh=False):
    """Gop `new_results` vao ban ghi cu thay vi ghi de no.

    Dong cu co cung `id_key` bi thay bang dong moi, DUNG VI TRI CU (de file on
    dinh, de diff). Dong cu khong xuat hien trong lan chay nay duoc giu nguyen
    -- day chinh la diem cua ham: chay mot cay khong duoc xoa 45 cay kia.

    `fresh=True` bo qua ban ghi cu hoan toan (co `--fresh-summary`), dung khi
    can don cac dong da cu khong con tuong ung voi du lieu tren dia.

    Khi chay `--all` khong kem `--skip-existing`, moi cay deu co trong
    `new_results` nen gop == ghi de: hanh vi khong doi so voi truoc.
    """
    new_results = list(new_results or [])
    if fresh:
        return new_results

    old = _old_rows(summary_path)
    if not old:
        return new_results

    by_id = {}
    for r in new_results:
        if isinstance(r, dict) and r.get(id_key) is not None:
            by_id[r[id_key]] = r

    merged = []
    used = set()
    for r in old:
        rid = r.get(id_key) if isinstance(r, dict) else None
        if rid is not None and rid in by_id:
            merged.append(by_id[rid])
            used.add(rid)
        else:
            merged.append(r)

    # Cay lan dau xuat hien (va cac dong khong co dinh danh) -> noi vao cuoi.
    for r in new_results:
        rid = r.get(id_key) if isinstance(r, dict) else None
        if rid is None or rid not in used:
            merged.append(r)

    return merged


def dump_summary(path, obj):
    """Ghi JSON theo kieu atomic: ghi file tam roi os.replace.

    May nay co lich su hong du lieu o muc byte (CLAUDE.md muc 4), nen khong ghi
    thang de bi cat ngang. `os.replace` la atomic tren NTFS.
    Luon UTF-8: mac dinh cua open() tren Windows la cp1252.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_overrides(path, label=""):
    """Nap file override theo cay: {"<ten_scan>": {<tham_so>: <gia_tri>}}.

    Cung khuon voi `per_scan_overrides.json` (04_trim_mesh.py) va
    `finalize_overrides.json` (04b_finalize_mesh.py) da co tu truoc.
    """
    data = _read_json(path)
    if not isinstance(data, dict):
        return {}
    data = {k: v for k, v in data.items() if isinstance(v, dict)}
    if data:
        print(f"Per-scan overrides{label} loaded: {list(data.keys())}")
    return data


def save_overrides(path, name, values):
    """Cap nhat mot muc trong file override, giu nguyen cac muc khac.

    `values` rong hoac None -> xoa muc do (khong de lai rac trong file).
    """
    data = _read_json(path)
    if not isinstance(data, dict):
        data = {}
    if values:
        data[name] = values
    else:
        data.pop(name, None)
    if data:
        dump_summary(path, data)
    elif os.path.exists(path):
        os.remove(path)
    return data
