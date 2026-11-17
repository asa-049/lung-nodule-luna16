# 为第三个结节（10.5 mm）做导航：算层号、中心坐标、像素直径，并画预览图
import os
import glob, sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

LUNA = r"E:\luna16"
SD   = (sys.argv[1] if len(sys.argv) > 1 else "478441")
OUT  = rf"D:\AAA\project2-lung-seg\figs\nav_{SD}.png"

cand = [p for p in glob.glob(os.path.join(LUNA, "*.mhd")) if os.path.basename(p).endswith(SD + ".mhd")]
mhd = cand[0]
img = sitk.ReadImage(mhd)
arr = sitk.GetArrayFromImage(img)
sx, sy, sz = img.GetSpacing()

ann = pd.read_csv(os.path.join(LUNA, "annotations.csv"))
sub = ann[ann["seriesuid"].str.endswith(SD)].sort_values("diameter_mm")

print(f"CT: {os.path.basename(mhd)}")
print(f"体积 {arr.shape}（z,y,x）  spacing {sx:.3f}/{sy:.3f}/{sz:.3f} mm")
print(f"结节数 {len(sub)}\n")

for _, r in sub.iterrows():
    x, y, z = img.TransformPhysicalPointToIndex((r.coordX, r.coordY, r.coordZ))
    d_px = r.diameter_mm / sx
    vol = 4 / 3 * np.pi * (r.diameter_mm / 2) ** 3
    print(f"直径 {r.diameter_mm:.2f} mm → 第 {z} 层 / 共 {arr.shape[0]} 层，中心 (x={x}, y={y})，"
          f"直径 {d_px:.1f} px，半径 {d_px / 2:.1f} px；名义球体积 {vol:.0f} mm3")

r = sub.iloc[-1]
x, y, z = img.TransformPhysicalPointToIndex((r.coordX, r.coordY, r.coordZ))
d_px = r.diameter_mm / sx


def win(a, level=-600, width=1500):
    lo, hi = level - width / 2, level + width / 2
    return (np.clip(a, lo, hi) - lo) / (hi - lo)


S = 40
y0, y1, x0, x1 = y - S, y + S, x - S, x + S
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(win(arr[z, y0:y1, x0:x1]), cmap="gray")
ax.add_patch(patches.Circle((x - x0, y - y0), d_px / 2, fill=False, edgecolor="r", lw=1.8))
ax.set_title(f"layer {z}/{arr.shape[0]}  ·  {r.diameter_mm:.1f} mm ~= {d_px:.0f} px  (red = 直径圆)")
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
plt.close(fig)
print("\n图 →", OUT)
