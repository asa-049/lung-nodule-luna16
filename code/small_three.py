# 小结节三个版本的对照：手工粗笔刷 / 手工调窗后 / 阈值法，再加直径圆
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

CT  = r"E:\luna16\1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"
SEG = r"D:\AAA\project2-lung-seg\slicer\Segmentation.seg.nrrd"
OUT = r"D:\AAA\project2-lung-seg\figs\small_three_versions.png"
D_MM = 4.343199739

im = sitk.ReadImage(CT)
a = sitk.GetArrayFromImage(im)
sx, sy, sz = im.GetSpacing()
vv = sx * sy * sz
seg = sitk.GetArrayFromImage(sitk.ReadImage(SEG))

names = {1: "13.6mm", 2: "4.3mm 手工v2", 3: "4.3mm 阈值法"}
masks = {}
for v in sorted(np.unique(seg)):
    if v == 0:
        continue
    m = seg == v
    zz, yy, xx = np.nonzero(m)
    vol = m.sum() * vv
    eq = (6 * vol / np.pi) ** (1 / 3)
    zs = np.where(m.any(axis=(1, 2)))[0]
    print(f"[{v}] {names.get(v, '?'):14s} vox {int(m.sum()):5d}  vol {vol:8.1f} mm3  eqD {eq:5.2f}  "
          f"layers {len(zs):2d} ({zs.min()}-{zs.max()})  centroid ({xx.mean():.0f},{yy.mean():.0f},{zz.mean():.0f})  "
          f"meanHU {a[m].mean():6.0f}")
    masks[v] = m

print("\n--- 与名义值比 ---")
R = D_MM / 2
sv = 4 / 3 * np.pi * R ** 3
print(f"名义球 直径 {D_MM:.2f} mm -> 体积 {sv:.1f} mm3")


def dice(x, y):
    s = x.sum() + y.sum()
    return 2 * (x & y).sum() / s if s else 0.0


small = [v for v in masks if v != 1]
print("\n--- 小版本之间 3D Dice ---")
for i in range(len(small)):
    for j in range(i + 1, len(small)):
        vi, vj = small[i], small[j]
        print(f"{names[vi]} vs {names[vj]}: Dice {dice(masks[vi], masks[vj]):.3f}")

# 直径圆（第 57 层附近）
ann = pd.read_csv(os.path.join(os.path.dirname(CT), "annotations.csv"))
row = ann[(ann.seriesuid.str.endswith("658609808059")) & (ann.diameter_mm < 5)].iloc[0]
cx, cy, cz = im.TransformPhysicalPointToIndex((row.coordX, row.coordY, row.coordZ))
r_px = R / sx
circ = np.zeros_like(masks[2])
yy, xx = np.ogrid[:circ.shape[0], :circ.shape[1]]
circ[((xx - cx) ** 2 + (yy - cy) ** 2) <= r_px ** 2] = True
print("\n--- 各版本 vs 直径圆（2D, 最大层）---")
for v in small:
    z = np.where(masks[v].any(axis=(1, 2)))[0]
    z = z[int(np.argmax([masks[v][k].sum() for k in z]))]
    s = masks[v][z].sum() + circ.sum()
    print(f"{names[v]}: 层 {z}, 2D Dice {2 * (masks[v][z] & circ).sum() / s:.3f}")

zmax = int(cz)
S = 30
y0, y1, x0, x1 = cy - S, cy + S, cx - S, cx + S


def win(v, level=-600, width=1500):
    lo, hi = level - width / 2, level + width / 2
    return (np.clip(v, lo, hi) - lo) / (hi - lo)


fig, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ["直径圆（合成 GT）", "手工 v2（调窗重涂）", "阈值法", "三者叠加"]
colors = {2: "#55ff88", 3: "#ffcc44"}
for i, ax in enumerate(axes):
    ax.imshow(win(a[zmax, y0:y1, x0:x1]), cmap="gray")
    ax.set_title(titles[i])
    ax.axis("off")

axes[0].add_patch(patches.Circle((cx - x0, cy - y0), r_px, fill=False, edgecolor="#ff5555", lw=2))
axes[1].contour(masks[2][zmax, y0:y1, x0:x1], levels=[.5], colors=[colors[2]], linewidths=2)
axes[2].contour(masks[3][zmax, y0:y1, x0:x1], levels=[.5], colors=[colors[3]], linewidths=2)
axes[3].contour(masks[2][zmax, y0:y1, x0:x1], levels=[.5], colors=[colors[2]], linewidths=2)
axes[3].contour(masks[3][zmax, y0:y1, x0:x1], levels=[.5], colors=[colors[3]], linewidths=2)
axes[3].add_patch(patches.Circle((cx - x0, cy - y0), r_px, fill=False, edgecolor="#ff5555", lw=2, ls="--"))

fig.tight_layout()
fig.savefig(OUT, dpi=150)
plt.close(fig)
print("\n图 →", OUT)
