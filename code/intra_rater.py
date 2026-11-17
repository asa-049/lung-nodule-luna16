# intra-rater 一致性：同一颗结节、同一人、隔 8 天两次勾画
# 输入：slicer\ 下两次导出的二值 labelmap
# 输出：体积/层数/Dice/每层面积对比 + figs/intra_rater.png
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"D:\AAA\project2-lung-seg"
CT   = r"E:\luna16\1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"
R1   = os.path.join(ROOT, "slicer", "Segmentation-nodule_13.6mm-label.nrrd")
R2   = os.path.join(ROOT, "slicer", "Segmentation-nod_808059_d13.6_hand_r2-label.nrrd")
OUT  = os.path.join(ROOT, "figs", "intra_rater.png")

im = sitk.ReadImage(CT)
a = sitk.GetArrayFromImage(im)
sx, sy, sz = im.GetSpacing()
vv = sx * sy * sz

m1 = sitk.GetArrayFromImage(sitk.ReadImage(R1)) > 0
m2 = sitk.GetArrayFromImage(sitk.ReadImage(R2)) > 0


def dice(x, y):
    s = x.sum() + y.sum()
    return 2 * (x & y).sum() / s if s else 0.0


def stats(m):
    zs = np.where(m.any(axis=(1, 2)))[0]
    areas = {int(z): m[z].sum() * sx * sy for z in zs}
    return dict(vox=int(m.sum()), vol=m.sum() * vv, n_layers=len(zs),
                zmin=int(zs.min()), zmax=int(zs.max()), areas=areas)


s1, s2 = stats(m1), stats(m2)
inter = int((m1 & m2).sum())
union = int((m1 | m2).sum())

print(f"体素间距 {sx:.3f}/{sy:.3f}/{sz:.3f}  单体素 {vv:.4f} mm3\n")
print(f"{'':10s}{'体素':>7s}{'体积mm3':>10s}{'层数':>5s}{'层范围':>10s}")
print(f"{'第1次(11/9)':10s}{s1['vox']:7d}{s1['vol']:10.1f}{s1['n_layers']:5d}{str((s1['zmin'], s1['zmax'])):>12s}")
print(f"{'第2次(11/17)':10s}{s2['vox']:7d}{s2['vol']:10.1f}{s2['n_layers']:5d}{str((s2['zmin'], s2['zmax'])):>12s}")

d = dice(m1, m2)
dv = (s2["vol"] - s1["vol"]) / s1["vol"] * 100
print(f"\n交集 {inter} 体素 | 并集 {union} 体素")
print(f"3D Dice      = {d:.3f}")
print(f"体积差        = {dv:+.1f}%  （|差| = {abs(dv):.1f}%）")
print(f"包含关系      : m1⊂m2? {bool((m1 & ~m2).sum()==0)}   m2⊂m1? {bool((m2 & ~m1).sum()==0)}")

print("\n--- 每层面积 (mm2) ---")
print(f"{'层':>4s}{'第1次':>9s}{'第2次':>9s}{'差':>9s}")
zs = sorted(set(s1["areas"]) | set(s2["areas"]))
for z in zs:
    a1 = s1["areas"].get(z, 0.0)
    a2 = s2["areas"].get(z, 0.0)
    print(f"{z:4d}{a1:9.1f}{a2:9.1f}{a2-a1:+9.1f}")

# 图：最大层 3 面板
zmax = max(s1["zmax"], s2["zmax"])
zshow = int(np.argmax([m1[z].sum() + m2[z].sum() for z in range(a.shape[0])]))


def win(v, level=-600, width=1500):
    lo, hi = level - width / 2, level + width / 2
    return (np.clip(v, lo, hi) - lo) / (hi - lo)


zz, yy, xx = np.nonzero(m1 | m2)
cy, cx = int(yy.mean()), int(xx.mean())
S = 45
y0, y1, x0, x1 = cy - S, cy + S, cx - S, cx + S

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for ax in axes:
    ax.imshow(win(a[zshow, y0:y1, x0:x1]), cmap="gray")
    ax.axis("off")

axes[0].contour(m1[zshow, y0:y1, x0:x1], levels=[.5], colors=["#55ff88"], linewidths=2)
axes[0].set_title(f"第1次 (11/9) · layer {zshow}")
axes[1].contour(m2[zshow, y0:y1, x0:x1], levels=[.5], colors=["#ffcc44"], linewidths=2)
axes[1].set_title(f"第2次 (11/17) · layer {zshow}")
axes[2].contour(m1[zshow, y0:y1, x0:x1], levels=[.5], colors=["#55ff88"], linewidths=2)
axes[2].contour(m2[zshow, y0:y1, x0:x1], levels=[.5], colors=["#ffcc44"], linewidths=2)
axes[2].set_title(f"叠加 · 3D Dice = {d:.3f}")

fig.tight_layout()
fig.savefig(OUT, dpi=150)
plt.close(fig)
print("\n图 →", OUT)
