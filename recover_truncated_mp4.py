#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
recover_truncated_mp4.py -- cuu rgb.mp4 cua mot phien quet bi ngat giua chung.

TRIEU CHUNG
-----------
ffmpeg bao `moov atom not found` va buoc 1 cua pipeline chet ngay:

    [mov,mp4,m4a,3gp,3g2,mj2] moov atom not found
    Error opening input file .../rgb.mp4

File van to binh thuong (hang tram MB). Duyet cac box cap cao nhat se thay:

    ftyp    28 byte
    wide     8 byte
    mdat     0 byte     <- khai bao kich thuoc 0 = "keo den het file"
    (khong co moov)

Day KHONG phai hong o cua CLAUDE.md muc 4. Day la ban ghi hinh bi ngat:
Stray Scanner ghi bang chi muc `moov` o CUOI phien, nen het pin / thoat app
giua chung thi toan bo hinh anh van con trong `mdat` ma khong co gi mo ta no.
Cung ly do do, `camera_matrix.csv` co the cung thieu -- xem ghi chu cuoi file.

CACH CUU
--------
Du lieu anh trong `mdat` la cac NAL don vi HEVC co tien to do dai 4 byte. Thu
duy nhat mat di la cau hinh bo ma (VPS/SPS/PPS), von nam trong box `hvcC`
thuoc `moov`. Lay no tu mot scan LANH cung may / cung app / cung thiet lap,
roi:

    1. doc VPS/SPS/PPS tu file tham chieu
    2. duyet `mdat` hong, doi tien to do dai -> ma khoi dau Annex-B
    3. ghep header vao dau -> luong .h265 giai ma duoc

Da cuu thanh cong `cay_0006_1805` (2,812 khung) va `cay_0022_1805` (1,724
khung) bang `cay_0007_1805` lam khuon. Ca hai doc het file khong dut dong bo
mot lan nao, va so khung ra khop voi so anh depth (2,850 va 1,728) -- day la
phep kiem doc lap cho thay viec phan tich luong la dung.

DUNG
----
    python recover_truncated_mp4.py <scan_hong> --ref <scan_lanh> [--extract]

`--extract` giai luon ra <scan_hong>/rgb/%06d.jpg theo dung quy uoc ten ma
01_extract_rgb_frames.py dung, roi chay tiep pipeline tu buoc 2:

    python run_full_pipeline.py <scan_hong> --from-step 2
"""

import argparse
import os
import struct
import subprocess
import sys

START_CODE = b"\x00\x00\x00\x01"


def iter_boxes(path):
    """Duyet cac box cap cao nhat cua mot file mp4."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        off = 0
        while off < size:
            f.seek(off)
            hdr = f.read(8)
            if len(hdr) < 8:
                return
            n = struct.unpack(">I", hdr[:4])[0]
            typ = hdr[4:8].decode("latin1")
            if n == 1:
                n = struct.unpack(">Q", f.read(8))[0]
            yield typ, n, off
            if n < 8:
                return
            off += n


def find_moov(path):
    for typ, n, off in iter_boxes(path):
        if typ == "moov":
            with open(path, "rb") as f:
                f.seek(off)
                return f.read(n)
    return None


def codec_header(ref_path):
    """Tra ve (header Annex-B, so byte cua truong do dai) tu scan tham chieu."""
    moov = find_moov(ref_path)
    if moov is None:
        raise SystemExit(f"File tham chieu cung khong co moov: {ref_path}")
    i = moov.find(b"hvcC")
    if i < 0:
        raise SystemExit("Chua ho tro: file tham chieu khong phai HEVC (khong co hvcC)")
    n = struct.unpack(">I", moov[i - 4:i])[0]
    cfg = moov[i + 4:i - 4 + n]
    length_size = (cfg[21] & 3) + 1

    p = 22
    n_arrays = cfg[p]
    p += 1
    out = b""
    for _ in range(n_arrays):
        cnt = struct.unpack(">H", cfg[p + 1:p + 3])[0]
        p += 3
        for _ in range(cnt):
            ln = struct.unpack(">H", cfg[p:p + 2])[0]
            p += 2
            out += START_CODE + cfg[p:p + ln]
            p += ln
    return out, length_size


def recover(broken_path, ref_path, out_path):
    hdr, length_size = codec_header(ref_path)

    mdat_off = None
    for typ, n, off in iter_boxes(broken_path):
        if typ == "mdat":
            mdat_off = off + 8
            break
    if mdat_off is None:
        raise SystemExit("Khong tim thay mdat -- file nay hong kieu khac")

    data = open(broken_path, "rb").read()
    out = bytearray(hdr)
    off, n_nal, n_frames = mdat_off, 0, 0
    while off + length_size <= len(data):
        ln = int.from_bytes(data[off:off + length_size], "big")
        if ln <= 0 or off + length_size + ln > len(data):
            break
        nal = data[off + length_size:off + length_size + ln]
        out += START_CODE + nal
        n_nal += 1
        if (nal[0] >> 1) & 0x3F < 32:      # NAL anh (VCL)
            n_frames += 1
        off += length_size + ln

    read = off - mdat_off
    total = len(data) - mdat_off
    open(out_path, "wb").write(out)
    print(f"  {n_nal:,} NAL, {n_frames:,} khung hinh")
    print(f"  doc {read:,}/{total:,} byte" +
          ("  (het file, dong bo tot)" if read == total else "  *** MAT DONG BO ***"))
    print(f"  -> {out_path}  ({len(out) / 2**20:.0f} MB)")
    return n_frames


def extract(h265_path, scan_dir):
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"
    rgb_dir = os.path.join(scan_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-f", "hevc", "-i", h265_path,
           "-q:v", "2", "-start_number", "0",
           os.path.join(rgb_dir, "%06d.jpg")]
    subprocess.run(cmd, check=True)
    n_rgb = len([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
    dep = os.path.join(scan_dir, "depth")
    n_dep = len(os.listdir(dep)) if os.path.isdir(dep) else 0
    print(f"  giai ra {n_rgb:,} khung rgb  (co {n_dep:,} anh depth)")
    if n_dep and abs(n_rgb - n_dep) > 0.05 * n_dep:
        print(f"  *** CANH BAO: lech qua 5% so voi so anh depth ***")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scan", help="Thu muc scan co rgb.mp4 bi cat cut")
    p.add_argument("--ref", required=True,
                   help="Thu muc scan LANH cung may/app de lay VPS/SPS/PPS")
    p.add_argument("--out", default=None, help="File .h265 ghi ra")
    p.add_argument("--extract", action="store_true",
                   help="Giai luon ra <scan>/rgb/%%06d.jpg")
    a = p.parse_args()

    broken = os.path.join(a.scan, "rgb.mp4")
    ref = os.path.join(a.ref, "rgb.mp4")
    for f in (broken, ref):
        if not os.path.exists(f):
            raise SystemExit(f"Khong thay {f}")
    out = a.out or os.path.join(a.scan, "rgb_recovered.h265")

    print(f"Cuu {broken}\n  khuon: {ref}")
    recover(broken, ref, out)
    if a.extract:
        extract(out, a.scan)


if __name__ == "__main__":
    main()
