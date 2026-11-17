"""从 LIDC-IDRI 医生轮廓生成 mask，并与 LUNA16 的「直径圆」对比。

数据来源：pylidc 的注释库 data/pylidc.sqlite（无需下载 DICOM、无需装 pylidc）
运行环境：D:\\Anaconda3\\python.exe（需要 SimpleITK / pandas / numpy / PIL）

用法：
    D:\\Anaconda3\\python.exe code/lidc_mask.py --seriesuid <UID>
    D:\\Anaconda3\\python.exe code/lidc_mask.py --seriesuid <UID> --dump 3
"""
import argparse
import os
import sqlite3
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image, ImageDraw

ROOT = r"D:\AAA\project2-lung-seg"
DB = os.path.join(ROOT, "data", "pylidc.sqlite")
LUNA = r"E:\luna16"
OUT = os.path.join(ROOT, "data", "lidc_masks")


def load_scan(seriesuid):
    """返回 (scan_id, sorted_zvals)；zvals 的排序号 = 切片序号。"""
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT id FROM scans WHERE series_instance_uid=?", (seriesuid,)).fetchone()
    if not row:
        conn.close()
        raise SystemExit(f"未在 LIDC 中找到 seriesuid: {seriesuid}")
    sid = row[0]
    zs = [v for (v,) in conn.execute("SELECT val FROM zvals WHERE scan_id=? ORDER BY val", (sid,))]
    conn.close()
    return sid, zs


def contour_rows(scan_id):
    """取出该扫描所有医生的轮廓行。"""
    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT a.id AS ann_id, a._nodule_id, a.malignancy, c.inclusion,
               c.image_z_position, c.coords
        FROM annotations a JOIN contours c ON c.annotation_id = a.id
        WHERE a.scan_id = ?
        ORDER BY a.id
    """, (scan_id,)).fetchall()
    conn.close()
    return rows


def build_mask(rows, zs, shape):
    """把轮廓栅格化成 3D mask。返回（mask, 医生数说明）。"""
    H, W = shape[1], shape[2]
    mask = np.zeros(shape, dtype=np.uint8)
    for ann_id, nodule_id, malign, inclusion, zpos, coords in rows:
        if not inclusion:          # 0 = 排除（医生画的外圈不算）
            continue
        if zpos is None:
            continue
        # image_z_position → 切片序号（用最近的 zval）
        idx = int(np.argmin(np.abs(np.array(zs) - zpos)))
        pts = []
        for line in coords.strip().splitlines():
            if "," not in line:
                continue
            x, y = line.split(",")[:2]
            pts.append((float(x), float(y)))
        if len(pts) < 3:
            continue
        img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(img).polygon(pts, outline=1, fill=1)
        layer = np.array(img, dtype=np.uint8)
        mask[idx] = np.maximum(mask[idx], layer)
    return mask


def luna_nodules(seriesuid):
    df = pd.read_csv(os.path.join(LUNA, "annotations.csv"))
    return df[df["seriesuid"] == seriesuid].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seriesuid", required=True)
    ap.add_argument("--dump", type=int, default=0, help="把前 N 个医生轮廓的 mask 存成 npy")
    args = ap.parse_args()

    img = sitk.ReadImage(os.path.join(LUNA, args.seriesuid + ".mhd"))
    shape = (img.GetSize()[2], img.GetSize()[1], img.GetSize()[0])   # (z, y, x)
    sx, sy, sz = img.GetSpacing()
    print(f"影像 shape={shape}  spacing=({sx:.2f},{sy:.2f},{sz:.2f})")

    sid, zs = load_scan(args.seriesuid)
    rows = contour_rows(sid)
    ann_ids = sorted({r[0] for r in rows})
    print(f"LIDC 标注 {len(ann_ids)} 个，轮廓 {len(rows)} 行")

    luna = luna_nodules(args.seriesuid)
    print(f"LUNA 在此例标注 {len(luna)} 个结节\n")

    os.makedirs(OUT, exist_ok=True)
    print("%-6s %-10s %6s %10s %8s %10s" % ("ann", "nodule", "层数", "体积mm3", "质心距", "最近LUNA"))
    for k, ann_id in enumerate(ann_ids, 1):
        sub = [r for r in rows if r[0] == ann_id]
        mask = build_mask(sub, zs, shape)
        vox = int(mask.sum())
        if vox == 0:
            print("%-6d %-10s %6s %10s" % (ann_id, sub[0][1], "-", "0（无内轮廓）"))
            continue
        vol = vox * sx * sy * sz
        zz, yy, xx = np.nonzero(mask)
        cx, cy, cz = xx.mean(), yy.mean(), zz.mean()
        # 体素 → 世界坐标
        wx, wy, wz = img.TransformContinuousIndexToPhysicalPoint((float(cx), float(cy), float(cz)))
        best, bd = None, 1e9
        for _, n in luna.iterrows():
            d = ((wx - n["coordX"]) ** 2 + (wy - n["coordY"]) ** 2 + (wz - n["coordZ"]) ** 2) ** 0.5
            if d < bd:
                best, bd = n, d
        print("%-6d %-10s %6d %10.1f %8.2f %10s" % (
            ann_id, str(sub[0][1])[:9], int(mask.sum(axis=(1, 2)).astype(bool).sum()), vol, bd,
            f"{best['diameter_mm']:.1f}mm" if best is not None else "-"))
        if args.dump and k <= args.dump:
            np.save(os.path.join(OUT, f"ann{ann_id}.npy"), mask)
            print("      → 已存", os.path.join(OUT, f"ann{ann_id}.npy"))


if __name__ == "__main__":
    main()
