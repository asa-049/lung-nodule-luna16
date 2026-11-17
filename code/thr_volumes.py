# 用 numpy 直接按 HU 阈值算体积（绕开 Slicer 界面，参数写死、可复现）
# 目标结节：CT 658609808059 的小结节，等效直径 4.34 mm，第 57 层，中心 (x=408, y=337)
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

CT  = r"E:\luna16\1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"
SD  = "658609808059"
D_MM = 4.343199739
HALF_XY, HALF_Z = 10, 3          # 以中心为心，20×20×(2*3+1) 的窗口

im = sitk.ReadImage(CT)
a = sitk.GetArrayFromImage(im).astype(np.float32)
sx, sy, sz = im.GetSpacing()
vv = sx * sy * sz

ann = pd.read_csv(os.path.join(os.path.dirname(CT), "annotations.csv"))
row = ann[(ann.seriesuid.str.endswith(SD)) & (ann.diameter_mm < 5)].iloc[0]
cx, cy, cz = im.TransformPhysicalPointToIndex((row.coordX, row.coordY, row.coordZ))
print(f"中心体素 (x={cx}, y={cy}, z={cz})   spacing {sx:.3f}/{sy:.3f}/{sz:.3f}  体素 {vv:.4f} mm3")

sub = a[cz - HALF_Z:cz + HALF_Z + 1, cy - HALF_XY:cy + HALF_XY + 1, cx - HALF_XY:cx + HALF_XY + 1]
print(f"窗口 {sub.shape}（z,y,x），HU {sub.min():.0f} ~ {sub.max():.0f}\n")

R = D_MM / 2
sphere_v = 4 / 3 * np.pi * R ** 3
print(f"名义球（直径 {D_MM:.2f} mm）体积 {sphere_v:.1f} mm3\n")

print(f"{'阈值区间':>14} | {'体素':>5} | {'体积mm3':>8} | {'等效直径':>8} | {'层数':>4} | {'meanHU':>7} | {'最大岛体素':>8}")
print("-" * 78)
rows = []
for lo in (-400, -300, -200, -100):
    m = (sub >= lo) & (sub <= 250)
    if m.sum() == 0:
        print(f"{lo:>6}~250     | 0")
        continue
    lab, n = ndimage.label(m)
    if n > 1:                                     # 等价于 Slicer 的 Keep largest island
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    vox = int(m.sum())
    vol = vox * vv
    eqd = (6 * vol / np.pi) ** (1 / 3)
    layers = int(m.any(axis=(1, 2)).sum())
    hu = sub[m]
    print(f"{lo:>6}~250     | {vox:5d} | {vol:8.1f} | {eqd:7.2f} | {layers:4d} | {hu.mean():7.1f} | "
          f"{vox if n <= 1 else int(max(ndimage.sum(m, lab, range(1, n + 1)))):8d}")
    rows.append(dict(lo=lo, hi=250, vox=vox, vol=round(vol, 1), eqd=round(eqd, 2),
                     layers=layers, meanHU=round(float(hu.mean()), 1), maxHU=int(hu.max())))

df = pd.DataFrame(rows)
out = r"D:\AAA\project2-lung-seg\figs\threshold_volumes.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print("\nCSV →", out)
