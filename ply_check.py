# -*- coding: ascii -*-
"""Kiem tra file PLY nhi phan TRUOC khi giao cho open3d.

LY DO TON TAI: mot byte hong trong PLY lam open3d chet bang segmentation
fault -- o tang C, nen try/except cua Python KHONG bat duoc, ca tien trinh
di theo. Gap that tren cay_0049_1805 (bo 18May): hai byte dem so dinh cua
mat bi lat thanh 139 va 220 thay vi 3, nen open3d co doc 139 dinh cho mot
tam giac roi chay vuot vung dem. Kich thuoc file khop chinh xac tung byte
voi header, toa do deu huu han -- tuc file KHONG bi cat cut hay ghi do,
dung hai byte bi lat gia tri. Do la hong o muc byte (CLAUDE.md muc 4).

Vi khong bat duoc loi, cach duy nhat la kiem truoc. Kiem tra o day la
thuan Python, khong nap hinh hoc, nen nhanh -- chi doc header cong voi
phan ban ghi mat.

Dung nhu module:
    from ply_check import ply_is_sane
    ok, ly_do = ply_is_sane(path)

Hoac chay truc tiep de quet ca corpus:
    python ply_check.py "D:\\Backup\\Thucdia-18May2026"
    python ply_check.py "D:\\Backup\\Thucdia-18May2026" --name trunk_mesh_final.ply
"""
import argparse
import os
import struct
import sys

# Kich thuoc tung kieu du lieu PLY, tinh theo byte.
TYPE_SIZE = {
    "char": 1, "uchar": 1, "int8": 1, "uint8": 1,
    "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
    "int": 4, "uint": 4, "int32": 4, "uint32": 4,
    "float": 4, "float32": 4,
    "double": 8, "float64": 8,
}

MAX_HEADER = 65536      # header PLY that khong bao gio gan muc nay

# File nam tren duong chay that: mot file trong nhom nay hong la phai dung lai
# ban quet. Moi file .ply khac trong output/ la san pham phu khong ai doc, va
# hong o do KHONG anh huong ket qua -- gop chung vao mot danh sach "can dung
# lai" la bao dong gia, vi 63 trong 64 file hong dau tien do duoc la loai nay.
ACTIVE_FILES = {
    "pointcloud.ply",             # buoc 2 -> buoc 3
    "trunk_pointcloud.ply",       # nguon mau, mat na, nhan cho buoc 5/6/8
    "trunk_mesh_poisson.ply",     # nguon hinh hoc hien tai (buoc 4b)
    "trunk_mesh_recolored.ply",   # buoc 5 -> 6
    "trunk_mesh_trimmed.ply",     # buoc 6 -> 7
    "trunk_mesh_final.ply",       # buoc 7 -> 8, va export_lichen_excel.py doc
    "trunk_mesh_detected.ply",    # ket qua cuoi
}

# Vi sao tung file duoi day khong con quan trong -- de nguoi doc khong phai
# tra cuu CLAUDE.md moi biet co nen lo hay khong.
INERT_REASON = {
    "mesh.ply": "mesh nhanh, khong buoc nao tu 3->8 doc; nay da co --no-mesh",
    "trunk_mesh_tsdf.ply": "Poisson da thay lam nguon hinh hoc, chi de lui ve",
    "trunk_mesh_detected_CHEO.ply": "file doi chieu cua annotate_lichen",
    "trunk_labeled.ply": "mau hoa theo huong doc, khong chay vao ket qua",
}


def _parse_header(raw):
    """Doc header PLY. Tra ve (offset_du_lieu, dinh_dang, danh_sach_element).

    Moi element la dict: {"name", "count", "props"}, trong do props la danh
    sach ("scalar", kieu) hoac ("list", kieu_dem, kieu_phan_tu).
    """
    end = raw.find(b"end_header\n", 0, MAX_HEADER)
    if end < 0:
        return None, None, None
    offset = end + len(b"end_header\n")
    try:
        text = raw[:end].decode("ascii")
    except UnicodeDecodeError:
        return None, None, None

    fmt = None
    elements = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            try:
                elements.append({"name": parts[1], "count": int(parts[2]),
                                 "props": []})
            except ValueError:
                return None, None, None
        elif parts[0] == "property" and elements:
            if parts[1] == "list" and len(parts) >= 5:
                elements[-1]["props"].append(("list", parts[2], parts[3]))
            elif len(parts) >= 3:
                elements[-1]["props"].append(("scalar", parts[1]))
    return offset, fmt, elements


def ply_is_sane(path, check_lists=True):
    """Kiem tra mot file PLY nhi phan co an toan de open3d doc khong.

    Tra ve (True, "") neu khong tim thay van de, hoac (False, ly_do).

    Chi kiem duoc file 'binary_little_endian'. Voi ascii hoac big-endian thi
    tra ve True kem ly do "khong kiem" -- khong gia vo la da kiem.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except (IOError, OSError) as e:
        return False, "khong doc duoc file: %s" % e

    if not raw.startswith(b"ply"):
        return False, "khong phai file PLY (thieu chu ky 'ply')"

    offset, fmt, elements = _parse_header(raw)
    if offset is None:
        return False, "header hong hoac khong co 'end_header'"
    if fmt != "binary_little_endian":
        return True, "khong kiem (dang %s, chi kiem binary_little_endian)" % fmt
    if not elements:
        return False, "header khong khai element nao"

    pos = offset
    total = len(raw)
    for el in elements:
        fixed = 0
        has_list = False
        for p in el["props"]:
            if p[0] == "scalar":
                if p[1] not in TYPE_SIZE:
                    return True, "khong kiem (kieu la '%s')" % p[1]
                fixed += TYPE_SIZE[p[1]]
            else:
                has_list = True

        if not has_list:
            need = el["count"] * fixed
            if pos + need > total:
                return False, ("element '%s' can %d byte nhung file chi con %d"
                               % (el["name"], need, total - pos))
            pos += need
            continue

        # Element co truong danh sach (thuong la mat: 1 byte dem + N chi so).
        # Phai di tung ban ghi, vi do dai moi ban ghi phu thuoc byte dem do.
        if not check_lists:
            return True, ""
        for i in range(el["count"]):
            rec = pos
            for p in el["props"]:
                if p[0] == "scalar":
                    rec += TYPE_SIZE[p[1]]
                    continue
                cnt_t, item_t = p[1], p[2]
                if cnt_t not in TYPE_SIZE or item_t not in TYPE_SIZE:
                    return True, "khong kiem (kieu danh sach la)"
                cs = TYPE_SIZE[cnt_t]
                if rec + cs > total:
                    return False, ("het du lieu o ban ghi '%s' thu %d/%d"
                                   % (el["name"], i, el["count"]))
                n = int.from_bytes(raw[rec:rec + cs], "little")
                # Mesh tam giac tu pipeline nay luon la 3. Byte dem khac 3 la
                # dau hieu hong -- va la thu lam open3d chet.
                if el["name"] == "face" and n != 3:
                    return False, ("ban ghi 'face' thu %d/%d khai %d dinh "
                                   "(dang le 3) -- byte dem bi hong, open3d "
                                   "se doc vuot vung dem va chet"
                                   % (i, el["count"], n))
                if n > 255:
                    return False, ("ban ghi '%s' thu %d khai %d phan tu -- "
                                   "vo ly" % (el["name"], i, n))
                rec += cs + n * TYPE_SIZE[item_t]
            if rec > total:
                return False, ("ban ghi '%s' thu %d/%d vuot qua cuoi file"
                               % (el["name"], i, el["count"]))
            pos = rec

    if pos != total:
        return False, ("thua %d byte sau element cuoi (file dai hon header "
                       "khai)" % (total - pos))
    return True, ""


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Quet cac file PLY trong mot thu muc corpus, tim file "
                    "hong o muc byte truoc khi no lam open3d chet.")
    ap.add_argument("raw_data", help="Thu muc goc chua cac ban quet")
    ap.add_argument("--name", default=None,
                    help="Chi kiem mot ten file trong output/ "
                         "(mac dinh: moi file .ply)")
    args = ap.parse_args()

    if not os.path.isdir(args.raw_data):
        sys.exit("Khong thay thu muc: %s" % args.raw_data)

    n_ok = n_skip = 0
    active_bad, inert_bad = [], []
    for scan in sorted(os.listdir(args.raw_data)):
        out = os.path.join(args.raw_data, scan, "output")
        if not os.path.isdir(out):
            continue
        names = [args.name] if args.name else sorted(
            f for f in os.listdir(out) if f.lower().endswith(".ply"))
        for nm in names:
            p = os.path.join(out, nm)
            if not os.path.exists(p):
                continue
            ok, why = ply_is_sane(p)
            if ok and why:
                n_skip += 1
            elif ok:
                n_ok += 1
            elif nm in ACTIVE_FILES:
                active_bad.append((scan, nm, why))
                print("  HONG  %-22s %-26s %s" % (scan, nm, why), flush=True)
            else:
                inert_bad.append((scan, nm, why))

    print("")
    print("Ket qua: %d file lanh, %d file khong kiem duoc"
          % (n_ok, n_skip))
    print("  HONG tren duong chay that : %d" % len(active_bad))
    print("  HONG o file khong ai doc  : %d" % len(inert_bad))

    if inert_bad:
        # Tach rieng vi day la bao dong gia: lan quet dau tien bo 18May cho
        # 64 file hong ma 63 thuoc loai nay. Gop chung thi thanh "14 cay can
        # dung lai", trong khi that ra chi 1 cay.
        print("")
        print("File hong KHONG anh huong ket qua (co the xoa):")
        by_name = {}
        for _, nm, _ in inert_bad:
            by_name[nm] = by_name.get(nm, 0) + 1
        for nm in sorted(by_name):
            print("  %4d x %-28s %s"
                  % (by_name[nm], nm, INERT_REASON.get(nm, "khong co buoc nao doc")))

    if active_bad:
        print("")
        print("Cac ban quet CAN DUNG LAI:")
        for scan in sorted(set(b[0] for b in active_bad)):
            files = sorted(b[1] for b in active_bad if b[0] == scan)
            print("  %-24s %s" % (scan, ", ".join(files)))
        sys.exit(1)


if __name__ == "__main__":
    main()
