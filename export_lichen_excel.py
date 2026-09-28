# -*- coding: utf-8 -*-
"""Gop lichen_stats.json cua ca corpus thanh mot file Excel, kem vi tri va huong.

LUU Y VE MA HOA: file nay co chuoi tieng Viet CO DAU, khac quy uoc ASCII cua
repo (CLAUDE.md muc 5). Ly do cua quy uoc do la console Windows cp1252 khong in
duoc dau; o day moi chuoi co dau chi di vao O EXCEL, khong bao gio di qua
print(). Moi thu in ra man hinh deu la ASCII.

Nguon du lieu moi cay:
    output/lichen_stats.json      ty le dia y (mesh va diem)
    tree_gps_anchor.json          GPS luc bat dau quet, gio quet
    location.csv                  GPS tung khung, ten dia danh
    odometry.csv + location.csv   goc quay tu khung ARKit sang huong la ban
    output/trunk_mesh_final.ply   truc than, de biet nua A/B nam huong nao

Chay:
    python export_lichen_excel.py --raw-data "D:\\Backup\\Thucdia-18May2026"
"""
import argparse
import csv
import datetime
import importlib.util
import json
import math
import os
import statistics
import sys

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ply_check import ply_is_sane

DEFAULT_RAW = r"D:\Backup\Thucdia-18May2026"
TZ_HOURS = 7                 # gio Viet Nam so voi UTC
SITE_RADIUS_M = 300.0        # gom cay thanh khu vuc
FALLBACK_SIN = 0.1           # buoc 8 lui ve truc tuy tien duoi nguong nay
STAB_MIN = 0.90              # duoi muc nay thi goc la ban khong on dinh

# Con so duy nhat duoc phep bao cao (CLAUDE.md muc 12 item 22)
CORPUS_MEAN = 23.2
CORPUS_CI = (14.4, 32.0)
CORPUS_N = 10

_DL = None


def check_openpyxl():
    """Bao thieu thu vien NGAY, thay vi doc het ca corpus roi moi bao.

    openpyxl chi duoc nap trong write_excel(), tan cuoi chuong trinh. Chay
    bang python he thong (khong phai venv cua du an) thi no thieu, va loi chi
    hien ra sau khi da doc xong moi cay -- mat cong vo ich.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit("Thieu thu vien 'openpyxl'.\n"
                 "Dang chay bang: %s\n"
                 "Neu do khong phai venv cua du an thi dung venv:\n"
                 '    D:\\Lichen_project\\venv\\Scripts\\python.exe '
                 "export_lichen_excel.py ...\n"
                 "Hoac cai vao python dang dung:  pip install openpyxl"
                 % sys.executable)


def check_writable(path):
    """Bao truoc neu khong ghi duoc, thay vi doc het 49 cay roi moi do loi.

    Truong hop thuong gap nhat: file dang mo trong Excel. Excel giu khoa ghi,
    openpyxl nem PermissionError o dong save() cuoi cung -- tuc la sau khi da
    lam xong toan bo cong viec.
    """
    folder = os.path.dirname(os.path.abspath(path)) or "."
    if not os.path.isdir(folder):
        sys.exit("Khong thay thu muc de ghi: %s" % folder)
    if os.path.exists(path):
        try:
            with open(path, "r+b"):
                pass
        except (IOError, OSError):
            sys.exit("Khong ghi duoc de len:\n  %s\n"
                     "File dang mo trong Excel, hoac dang o che do chi doc.\n"
                     "Dong file do lai roi chay lai, hoac dung --out de ghi "
                     "sang ten khac." % path)
    else:
        try:
            with open(path, "wb"):
                pass
            os.remove(path)
        except (IOError, OSError) as e:
            sys.exit("Khong tao duoc file %s: %s" % (path, e))


def detect_module():
    """Nap 05_detect_lichen.py mot lan, dung lai cho moi cay."""
    global _DL
    if _DL is None:
        spec = importlib.util.spec_from_file_location(
            "detect_lichen", os.path.join(HERE, "05_detect_lichen.py"))
        _DL = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_DL)
    return _DL


# --------------------------------------------------------------------------- #
#  Hinh hoc
# --------------------------------------------------------------------------- #
def quat_to_matrix(q):
    """Quaternion (x, y, z, w) -> ma tran quay 3x3."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def frame_azimuth(vec):
    """Goc phuong vi cua mot vector trong khung ARKit (do, 0 = +X, tang ve +Z).

    Dung dung quy uoc cua annotate_lichen.sun_azimuth de hai ben so sanh duoc.
    """
    return math.degrees(math.atan2(vec[2], vec[0])) % 360.0


COMPASS_8 = [u"Bắc", u"Đông Bắc", u"Đông", u"Đông Nam",
             u"Nam", u"Tây Nam", u"Tây", u"Tây Bắc"]


def compass(bearing):
    """Do la ban -> ten huong 8 canh."""
    return COMPASS_8[int((bearing % 360) / 45.0 + 0.5) % 8]


def north_offset(scan_dir, max_samples=300):
    """Goc quay tu phuong vi khung ARKit sang huong la ban that.

    Ghep tung khung odometry voi so do la ban gan nhat theo thoi gian. Tra ve
    (goc_quay, do_on_dinh); do on dinh la do dai vector trung binh vong:
    1.0 = moi khung cho cung mot goc quay, 0 = hon loan.
    """
    od_path = os.path.join(scan_dir, "odometry.csv")
    loc_path = os.path.join(scan_dir, "location.csv")
    if not (os.path.exists(od_path) and os.path.exists(loc_path)):
        return None, 0.0
    try:
        od = np.loadtxt(od_path, delimiter=",", skiprows=1, usecols=range(9))
        rows = list(csv.DictReader(open(loc_path, encoding="utf-8-sig")))
    except Exception:
        return None, 0.0
    if len(od) < 10 or len(rows) < 10:
        return None, 0.0

    try:
        times = np.array([float(r["timestamp_unix"]) for r in rows])
        heads = np.array([float(r["heading_degrees"]) for r in rows])
    except (TypeError, ValueError, KeyError):
        return None, 0.0

    step = max(1, len(od) // max_samples)
    diffs = []
    for i in range(0, len(od), step):
        j = int(np.argmin(np.abs(times - od[i, 0])))
        if abs(times[j] - od[i, 0]) > 0.2:
            continue
        look = quat_to_matrix(od[i, 5:9]) @ np.array([0.0, 0.0, 1.0])
        if abs(look[0]) < 1e-9 and abs(look[2]) < 1e-9:
            continue
        diffs.append(math.radians((frame_azimuth(look) - heads[j]) % 360))
    if len(diffs) < 30:
        return None, 0.0
    c, s = np.cos(diffs).mean(), np.sin(diffs).mean()
    return math.degrees(math.atan2(s, c)) % 360.0, float(math.hypot(c, s))


def split_axis(scan_dir):
    """Truc ma buoc 8 dung de chia nua A / nua B.

    Lap lai dung phep tinh cua 05_detect_lichen.py (khong xap xi lai), vi cot
    phan tram hai nua trong Excel phai khop voi truc that su da sinh ra chung.

    Tra ve (vector_nua_A, goc_nghieng_than_do, dung_truc_tuy_tien).
    """
    mod = detect_module()
    mesh_path = os.path.join(scan_dir, "output", "trunk_mesh_final.ply")
    if not os.path.exists(mesh_path):
        return None, None, None

    # Kiem file TRUOC khi giao cho open3d. Mot byte hong lam open3d chet bang
    # segmentation fault o tang C -- try/except khong bat duoc, ca tien trinh
    # di theo, va 48 cay lanh con lai khong ra duoc bang. Xem ply_check.py.
    ok, why = ply_is_sane(mesh_path)
    if not ok:
        print("    BO QUA hinh hoc: %s" % why)
        return None, None, None

    verts = np.asarray(o3d.io.read_triangle_mesh(mesh_path).vertices)
    if len(verts) < 10:
        return None, None, None

    axis = mod.pca_axis(verts)
    try:
        grav = mod.estimate_gravity(scan_dir)
    except Exception:
        grav = np.array([0.0, 1.0, 0.0])
    up = grav - np.dot(grav, axis) * axis
    norm = float(np.linalg.norm(up))
    tilt = math.degrees(math.asin(min(1.0, norm)))
    if norm < FALLBACK_SIN:
        return np.array([1.0, 0.0, 0.0]), tilt, True
    return up / norm, tilt, False


# --------------------------------------------------------------------------- #
#  Doc du lieu tung cay
# --------------------------------------------------------------------------- #
def read_position(scan_dir):
    """GPS dai dien cho cay: trung vi cua 10% so do chinh xac nhat.

    Khong lay dong dau tien: luc bat dau quet GPS chua hoi tu, co ban ghi sai
    so toi 400m. Khong lay trung vi tat ca: cac so do kem lam lech ket qua.
    """
    out = {"lat": None, "lon": None, "alt": None, "acc": None,
           "place": "", "locality": ""}
    path = os.path.join(scan_dir, "location.csv")
    if not os.path.exists(path):
        return out
    try:
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    except Exception:
        return out
    if not rows:
        return out
    out["place"] = (rows[0].get("place_name") or "").strip()
    out["locality"] = (rows[0].get("locality") or "").strip()

    fixes = []
    for r in rows:
        try:
            fixes.append((float(r["gps_accuracy_m"]), float(r["latitude"]),
                          float(r["longitude"]), float(r["altitude_asl_m"])))
        except (TypeError, ValueError, KeyError):
            continue
    if not fixes:
        return out
    fixes.sort()
    best = fixes[:max(1, len(fixes) // 10)]
    out["lat"] = statistics.median(f[1] for f in best)
    out["lon"] = statistics.median(f[2] for f in best)
    out["alt"] = statistics.median(f[3] for f in best)
    out["acc"] = statistics.median(f[0] for f in best)
    return out


def find_scans(raw_root):
    """Moi thu muc con co output/lichen_stats.json deu la mot cay.

    KHONG loc theo tien to 'cay_': cac bo du lieu dat ten khac nhau -- bo
    18May dung 'cay_0007_1805', bo Jun2026 dung 'TM-tra-my-1__video_...'.
    Loc theo ket qua that su co thi dung cho moi bo, va tu dong bo qua cac
    thu muc khong phai cay nhu 'samples'.
    """
    out = []
    for name in sorted(os.listdir(raw_root)):
        d = os.path.join(raw_root, name)
        if os.path.isdir(d) and os.path.exists(
                os.path.join(d, "output", "lichen_stats.json")):
            out.append(name)
    return out


def short_name(scan):
    """Ten ngan de liet ke trong sheet khu vuc.

    'cay_0007_1805' -> '0007'. Ten kieu khac thi cat phan duoi '__video_...'
    vi no chi la dau thoi gian, khong phan biet cay.
    """
    if scan.startswith("cay_") and len(scan) >= 8:
        return scan[4:8]
    return scan.split("__video")[0].split("_video")[0]


def read_scan(raw_root, name):
    scan_dir = os.path.join(raw_root, name)
    stats_path = os.path.join(scan_dir, "output", "lichen_stats.json")
    if not os.path.exists(stats_path):
        return None

    rec = {"scan": name}
    st = json.load(open(stats_path, encoding="utf-8"))["stats"]
    ov = st.get("overall", {})
    rec["pct_mesh"] = ov.get("lichen_ratio_pct")
    rec["pct_point"] = ov.get("point_level_pct")
    rec["total_cm2"] = ov.get("total_cm2")
    rec["lichen_cm2"] = ov.get("total_lichen_cm2")
    rec["detector"] = st.get("detector", {}).get("detector", "?")
    rec["pct_a"] = st.get("upslope", {}).get("lichen_ratio_pct")
    rec["pct_b"] = st.get("downslope", {}).get("lichen_ratio_pct")

    rec.update(read_position(scan_dir))

    rec["when"] = None
    anchor_path = os.path.join(scan_dir, "tree_gps_anchor.json")
    if os.path.exists(anchor_path):
        try:
            iso = json.load(open(anchor_path))["scan_start_iso"]
            utc = datetime.datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S")
            rec["when"] = utc + datetime.timedelta(hours=TZ_HOURS)
        except Exception:
            pass

    off, stab = north_offset(scan_dir)
    rec["offset"] = off
    rec["stab"] = stab
    vec_a, tilt, arbitrary = split_axis(scan_dir)
    rec["tilt"] = tilt
    rec["arbitrary"] = arbitrary
    if vec_a is not None and off is not None:
        rec["bear_a"] = (frame_azimuth(vec_a) - off) % 360
        rec["bear_b"] = (rec["bear_a"] + 180) % 360
    else:
        rec["bear_a"] = rec["bear_b"] = None
    return rec


def assign_sites(recs):
    """Gom cay thanh khu vuc theo khoang cach GPS, dat ten tu place_name."""
    def metres(a, b):
        return math.hypot((a["lat"] - b["lat"]) * 111320.0,
                          (a["lon"] - b["lon"]) * 111320.0
                          * math.cos(math.radians(a["lat"])))

    located = [r for r in recs if r.get("lat") is not None]
    groups = []
    for r in located:
        for g in groups:
            if any(metres(r, q) < SITE_RADIUS_M for q in g):
                g.append(r)
                break
        else:
            groups.append([r])

    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if any(metres(a, b) < SITE_RADIUS_M
                       for a in groups[i] for b in groups[j]):
                    groups[i] += groups[j]
                    del groups[j]
                    merged = True
                    break
            if merged:
                break

    groups.sort(key=lambda g: -len(g))
    sites = []
    for g in groups:
        places = [r["place"] for r in g if r["place"]]
        gps_place = max(set(places), key=places.count) if places else u""
        label = gps_place or label_from_folders(g)
        for r in g:
            r["site"] = label
            r["gps_place"] = gps_place
        sites.append({"name": label, "gps_place": gps_place, "recs": g,
                      "n_named": len(places)})
    for r in recs:
        r.setdefault("site", u"Không có GPS")
        r.setdefault("gps_place", u"")
    return sites


def label_from_folders(group):
    """Ten cum khi location.csv khong co dia danh.

    Nhieu bo du lieu dat ten thu muc theo noi quet ('Tra-leng-1__video_...'),
    nen tien to chung cua ca cum thuong chinh la dia danh. Khi tien to do qua
    ngan de co nghia -- vi du cum chi gom 'TV4', 'TV5', 'Tv4' -- thi khong doan
    bua, ma neu ten mot thanh vien de nguoi doc tu nhan ra.
    """
    shorts = sorted(short_name(r["scan"]) for r in group)
    if len(shorts) > 1:
        lo, hi = shorts[0].lower(), shorts[-1].lower()
        i = 0
        while i < min(len(lo), len(hi)) and lo[i] == hi[i]:
            i += 1
        prefix = shorts[0][:i].rstrip("-_ 0123456789")
        if len(prefix) >= 3:
            return prefix
    return u"Chưa có tên — %s" % shorts[0]


# --------------------------------------------------------------------------- #
#  Ghi Excel
# --------------------------------------------------------------------------- #
COLUMNS = [
    (u"Cây", 16), (u"Khu vực", 20), (u"Vĩ độ", 12), (u"Kinh độ", 12),
    (u"Độ cao\n(m)", 9), (u"Sai số GPS\n(m)", 11),
    (u"Thời gian quét\n(giờ VN)", 16),
    (u"Địa y %\n(mesh)", 10), (u"Địa y %\n(điểm)", 10),
    (u"Tổng diện tích\n(cm²)", 13), (u"Diện tích địa y\n(cm²)", 14),
    (u"Nửa A %", 9), (u"Nửa A hướng\n(độ la bàn)", 13), (u"Nửa A", 11),
    (u"Nửa B %", 9), (u"Nửa B hướng\n(độ la bàn)", 13), (u"Nửa B", 11),
    (u"Trục chia hai nửa", 17), (u"Thân nghiêng\n(độ)", 12),
    (u"Độ ổn định\nla bàn", 11),
]


def write_excel(recs, sites, out_path, raw_data):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    head_fill = PatternFill("solid", fgColor="1F3864")
    head_font = Font(bold=True, color="FFFFFF", size=10)
    warn_fill = PatternFill("solid", fgColor="FFF2CC")
    thin = Side(style="thin", color="BFBFBF")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    centre = Alignment(horizontal="center", vertical="center", wrap_text=True)

    wb = Workbook()

    # ---- Sheet 1: bang chinh ---------------------------------------------- #
    ws = wb.active
    ws.title = u"Tỷ lệ địa y"
    for i, (name, width) in enumerate(COLUMNS, 1):
        c = ws.cell(row=1, column=i, value=name)
        c.fill, c.font, c.border, c.alignment = head_fill, head_font, box, centre
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.row_dimensions[1].height = 34
    ws.freeze_panes = "C2"

    for r, rec in enumerate(recs, 2):
        vals = [
            rec["scan"], rec.get("site", ""), rec.get("lat"), rec.get("lon"),
            rec.get("alt"), rec.get("acc"),
            rec["when"].strftime("%d/%m %H:%M") if rec.get("when") else "",
            rec.get("pct_mesh"), rec.get("pct_point"),
            rec.get("total_cm2"), rec.get("lichen_cm2"),
            rec.get("pct_a"), rec.get("bear_a"),
            compass(rec["bear_a"]) if rec.get("bear_a") is not None else "",
            rec.get("pct_b"), rec.get("bear_b"),
            compass(rec["bear_b"]) if rec.get("bear_b") is not None else "",
            (u"không đọc được mesh" if rec.get("arbitrary") is None
             else (u"tùy tiện" if rec["arbitrary"] else u"thân nghiêng")),
            rec.get("tilt"), rec.get("stab"),
        ]
        for i, v in enumerate(vals, 1):
            c = ws.cell(row=r, column=i, value=v)
            c.border = box
            if i in (3, 4):
                c.number_format = "0.00000"
            elif i in (5, 6, 8, 9, 12, 15, 19):
                c.number_format = "0.0"
            elif i in (10, 11):
                c.number_format = "#,##0"
            elif i in (13, 16):
                c.number_format = "0"
            elif i == 20:
                c.number_format = "0.00"
        if rec.get("arbitrary") is not False:
            ws.cell(row=r, column=18).fill = warn_fill
        if rec.get("stab", 1.0) < STAB_MIN:
            ws.cell(row=r, column=20).fill = warn_fill

    last = len(recs) + 3
    ws.cell(row=last, column=1, value=u"Trung bình").font = Font(bold=True)
    for col, key in ((8, "pct_mesh"), (9, "pct_point"),
                     (10, "total_cm2"), (11, "lichen_cm2")):
        vals = [r[key] for r in recs if r.get(key) is not None]
        c = ws.cell(row=last, column=col,
                    value=round(sum(vals) / len(vals), 1) if vals else None)
        c.font = Font(bold=True)
        c.number_format = "0.0" if col < 10 else "#,##0"
    note = ws.cell(row=last + 1, column=1,
                   value=u"Hai cột 'Địa y %' KHÔNG đáng tin ở mức từng cây — "
                         u"xem sheet 'Đọc trước khi dùng'.")
    note.font = Font(bold=True, color="C00000")

    # Dau nguon: de khong bao gio nham file cua bo du lieu nay voi bo khac.
    stamp = ws.cell(
        row=last + 2, column=1,
        value=u"Sinh lúc %s từ %s — %d cây."
              % (datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
                 raw_data, len(recs)))
    stamp.font = Font(italic=True, size=9, color="595959")

    # ---- Sheet 2: khu vuc -------------------------------------------------- #
    ws2 = wb.create_sheet(u"Khu vực")
    head2 = [(u"Khu vực", 22), (u"Địa danh GPS\nbáo về", 16), (u"Số cây", 8),
             (u"Vĩ độ", 12), (u"Kinh độ", 12),
             (u"Độ cao TB\n(m)", 11), (u"Bán kính khu\n(m)", 13),
             (u"Thời gian quét", 20), (u"Địa y % mesh\n(TB)", 13),
             (u"Địa y % điểm\n(TB)", 13), (u"Cây", 60)]
    for i, (h, w) in enumerate(head2, 1):
        c = ws2.cell(row=1, column=i, value=h)
        c.fill, c.font, c.border, c.alignment = head_fill, head_font, box, centre
        ws2.column_dimensions[get_column_letter(i)].width = w
    ws2.row_dimensions[1].height = 34

    for k, site in enumerate(sites, 2):
        g = site["recs"]
        lat = statistics.median(r["lat"] for r in g)
        lon = statistics.median(r["lon"] for r in g)
        radius = max(math.hypot((r["lat"] - lat) * 111320.0,
                                (r["lon"] - lon) * 111320.0
                                * math.cos(math.radians(lat))) for r in g)
        times = [r["when"] for r in g if r.get("when")]
        span = ("%s - %s" % (min(times).strftime("%d/%m %H:%M"),
                             max(times).strftime("%H:%M"))) if times else ""
        pm = [r["pct_mesh"] for r in g if r.get("pct_mesh") is not None]
        pp = [r["pct_point"] for r in g if r.get("pct_point") is not None]
        vals = [site["name"], site.get("gps_place", ""), len(g), lat, lon,
                statistics.median(r["alt"] for r in g), radius, span,
                round(sum(pm) / len(pm), 1) if pm else None,
                round(sum(pp) / len(pp), 1) if pp else None,
                ", ".join(sorted(short_name(r["scan"]) for r in g))]
        for i, v in enumerate(vals, 1):
            c = ws2.cell(row=k, column=i, value=v)
            c.border = box
            if i in (4, 5):
                c.number_format = "0.00000"
            elif i in (6, 7, 9, 10):
                c.number_format = "0.0"
        if site["n_named"] < len(g):
            ws2.cell(row=k, column=1).fill = warn_fill

    ws2.cell(row=len(sites) + 3, column=1,
             value=u"Ô tên khu vực tô vàng: một số cây trong cụm không có địa "
                   u"danh trong location.csv. Khi cả cụm đều trống, tên lấy từ "
                   u"tiền tố chung của tên thư mục. Cột 'Địa danh GPS báo về' "
                   u"là thứ điện thoại ghi lại — nó có thể thô hơn hoặc lệch "
                   u"so với tên thực địa, nên hai cột được để riêng thay vì "
                   u"gộp làm một.").font = Font(italic=True, size=9)

    # ---- Sheet 3: canh bao ------------------------------------------------- #
    ws3 = wb.create_sheet(u"Đọc trước khi dùng")
    ws3.column_dimensions["A"].width = 3
    ws3.column_dimensions["B"].width = 110

    n_arb = sum(1 for r in recs if r.get("arbitrary"))
    lines = [
        ("h", u"Ba điều phải biết trước khi trích số ra khỏi file này"),
        ("", u""),
        ("n", u"1. Cột 'Địa y %' KHÔNG dùng được ở mức từng cây."),
        ("b", u"Đo trên 10 cây được tô nhãn bằng tay: tương quan giữa số máy "
              u"tính ra và nhãn tay là r = +0.07 — tức là không có quan hệ "
              u"nào. Đoán một con số cố định cho mọi cây còn sai ít hơn: sai "
              u"số trung bình 8.8 điểm, so với 19.0 điểm của máy."),
        ("b", u"Bộ dò hiện tại giỏi việc xác định địa y NẰM ĐÂU trên thân "
              u"(F1 40.6% trên cây chưa từng thấy, so với 13.0% của hàm cũ), "
              u"không giỏi việc đếm CÓ BAO NHIÊU."),
        ("", u""),
        ("n", u"2. Con số duy nhất được phép báo cáo:"),
        ("b", u"Tỷ lệ địa y trung bình của mẫu: %.1f%%, khoảng tin cậy 95%% "
              u"%.1f–%.1f%%, n = %d cây. Lấy thẳng từ nhãn tay, không qua máy."
              % (CORPUS_MEAN, CORPUS_CI[0], CORPUS_CI[1], CORPUS_N)),
        ("b", u"Kèm theo một điều kiện: cây được chọn để quay một phần vì nhìn "
              u"thấy có địa y ở phía nắng. Con số này mô tả MẪU ĐÃ QUÉT, "
              u"không mô tả cả rừng quế."),
        ("", u""),
        ("n", u"3. Cột hướng là chỉ dấu, không phải số đo trắc địa."),
        ("b", u"Góc quay từ khung toạ độ máy quét sang hướng la bàn rất ổn "
              u"định trong từng lần quét (44/49 cây có độ ổn định >= 0.90). "
              u"Nhưng độ chính xác TUYỆT ĐỐI thì chưa kiểm chứng được. Phép "
              u"thử độc lập duy nhất có thể làm — so hướng sáng nhất đo trên "
              u"thân với vị trí mặt trời tính từ toạ độ và giờ quét — lệch "
              u"trung vị 47 độ; và không tách được lỗi đó là do la bàn hay do "
              u"'phía sáng nhất của thân' vốn không trùng hướng mặt trời (tán "
              u"che, mặt trời gần đỉnh đầu lúc giữa trưa). Đọc cột hướng với "
              u"sai số cỡ hàng chục độ."),
        ("", u""),
        ("h", u"Các cột có bẫy"),
        ("", u""),
        ("n", u"'Địa y % (mesh)' khác 'Địa y % (điểm)'"),
        ("b", u"Cột mesh tính theo diện tích tam giác; cột điểm tính theo số "
              u"điểm trong đám mây, không phụ thuộc độ mịn của mesh. Cột mesh "
              u"luôn thấp hơn một cách hệ thống, vì 8–14% diện tích mesh là "
              u"mặt cắt hai đầu thân — bị ép về 'vỏ cây' nhưng vẫn nằm ở mẫu "
              u"số. Nếu buộc phải so hai cây thì dùng cột điểm. Cả hai đều "
              u"dính cảnh báo số 1."),
        ("", u""),
        ("n", u"'Nửa A' / 'Nửa B' và cột 'Trục chia hai nửa'"),
        ("b", u"Đây là hai nửa của thân cây, mỗi nửa 180 độ, tâm nằm ở hướng "
              u"ghi trong cột bên cạnh. Trục chia do bước 8 của pipeline "
              u"chọn: khi thân nghiêng đủ nhiều (>= 5.7 độ) thì trục là hướng "
              u"nghiêng; dưới ngưỡng đó pipeline lùi về một trục TUỲ TIỆN. "
              u"Trong %d cây thì %d cây rơi vào trường hợp tuỳ tiện (ô tô "
              u"vàng)." % (len(recs), n_arb)),
        ("b", u"Cột hướng vẫn dùng được cho cả hai trường hợp: dù trục được "
              u"chọn tuỳ tiện thì hai nửa vẫn nằm ở hướng la bàn có thật — đó "
              u"chính là lý do cột hướng được thêm vào. Cái KHÔNG dùng được "
              u"là tên gọi cũ 'upslope'/'downslope' trong file JSON gốc."),
        ("", u""),
        ("n", u"'Thân nghiêng' không phải độ dốc địa hình"),
        ("b", u"Trường terrain_slope_deg trong slope_info.json đo độ nghiêng "
              u"của THÂN CÂY so với phương thẳng đứng, không đo độ dốc mặt "
              u"đất. Cây mọc thẳng bất kể đất dốc hay bằng. Trong dữ liệu thô "
              u"không có số đo độ dốc địa hình nào cả."),
        ("", u""),
        ("n", u"Toạ độ GPS xác định KHU VỰC, không xác định từng cây"),
        ("b", u"Sai số GPS tốt nhất mỗi cây là 2.5–11.5 m, trong khi cả một "
              u"khu chỉ trải rộng khoảng 30 m — các cây trong cùng khu không "
              u"phân biệt được bằng toạ độ. Giá trị trong bảng là trung vị "
              u"của 10% số đo chính xác nhất trong lần quét, không phải dòng "
              u"đầu tiên (lúc mới bật, GPS chưa hội tụ, có bản ghi sai số tới "
              u"400 m)."),
        ("", u""),
        ("h", u"Nguồn"),
        ("", u""),
        ("n", u"Bộ dữ liệu: %s" % raw_data),
        ("b", u"Sinh lúc %s, gồm %d cây. File này được GHI ĐÈ mỗi lần chạy "
              u"lại — mọi sửa tay trong file sẽ mất. Muốn giữ bản cũ thì đổi "
              u"tên nó, hoặc chạy với --out tên_khác.xlsx."
              % (datetime.datetime.now().strftime("%d/%m/%Y %H:%M"), len(recs))),
        ("", u""),
        ("b", u"Mỗi dòng lấy từ output/lichen_stats.json của cây đó, cộng với "
              u"tree_gps_anchor.json, location.csv, odometry.csv và "
              u"output/trunk_mesh_final.ply. Sinh lại bằng: "
              u"python export_lichen_excel.py"),
        ("b", u"Chi tiết cách đo, và các hướng đã thử rồi thất bại: CLAUDE.md "
              u"mục 12, các item 13, 16, 19, 20, 21, 22."),
    ]
    for row, (kind, text) in enumerate(lines, 1):
        c = ws3.cell(row=row, column=2, value=text)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if kind == "h":
            c.font = Font(bold=True, size=13, color="1F3864")
        elif kind == "n":
            c.font = Font(bold=True, size=11, color="C00000")
        else:
            c.font = Font(size=10)
            ws3.row_dimensions[row].height = max(15, 13 * (len(text) // 92 + 1))

    wb.save(out_path)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Gop lichen_stats.json toan corpus thanh mot file Excel.")
    ap.add_argument("--raw-data", default=DEFAULT_RAW,
                    help="Thu muc goc chua cac cay (mac dinh: %s)" % DEFAULT_RAW)
    ap.add_argument("--out", default=os.path.join(HERE, "ty_le_dia_y.xlsx"),
                    help="Duong dan file Excel ghi ra")
    args = ap.parse_args()

    if not os.path.isdir(args.raw_data):
        sys.exit("Khong thay thu muc: %s" % args.raw_data)
    check_openpyxl()
    check_writable(args.out)

    names = find_scans(args.raw_data)
    n_dirs = sum(1 for d in os.listdir(args.raw_data)
                 if os.path.isdir(os.path.join(args.raw_data, d)))
    print("Tim thay %d cay co ket qua (trong %d thu muc con) o %s"
          % (len(names), n_dirs, args.raw_data))
    if not names:
        sys.exit("Khong thu muc con nao co output/lichen_stats.json")

    recs, skipped = [], []
    for i, name in enumerate(names, 1):
        print("  [%2d/%2d] %s" % (i, len(names), name), flush=True)
        rec = read_scan(args.raw_data, name)
        if rec is None:
            skipped.append(name)
        else:
            recs.append(rec)
    if not recs:
        sys.exit("Khong doc duoc cay nao")
    if skipped:
        print("\nBo qua %d cay: %s" % (len(skipped), ", ".join(skipped)))

    sites = assign_sites(recs)
    write_excel(recs, sites, args.out, args.raw_data)

    n_arb = sum(1 for r in recs if r.get("arbitrary"))
    n_low = sum(1 for r in recs if r.get("stab", 1.0) < STAB_MIN)
    # In duong dan tuyet doi: --out nhan duong dan tuong doi thi file roi vao
    # thu muc dang dung, khong phai thu muc du an -- phai noi ro no o dau.
    print("\nDa ghi: %s" % os.path.abspath(args.out))
    print("  %d cay, %d khu vuc" % (len(recs), len(sites)))
    for s in sites:
        print("    %-24s %2d cay" % (s["name"], len(s["recs"])))
    print("  %d cay chia hai nua bang truc tuy tien (o to vang)" % n_arb)
    print("  %d cay co do on dinh la ban duoi %.2f (o to vang)"
          % (n_low, STAB_MIN))
    print("")
    print("  Nhac lai: cot 'Dia y phan tram' KHONG dang tin o muc tung cay.")
    print("  So duy nhat bao cao duoc: %.1f%% (KTC 95%% %.1f-%.1f%%, n=%d)."
          % (CORPUS_MEAN, CORPUS_CI[0], CORPUS_CI[1], CORPUS_N))


if __name__ == "__main__":
    main()
