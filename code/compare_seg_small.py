# 小结节（4.3 mm）人工勾画 vs 直径圆：从 .seg.nrrd 的并集里减掉大结节，单独取出小结节
import os
import sys
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

LAB   = sys.argv[1] if len(sys.argv) > 1 else r"D:\AAA\project2-lung-seg\slicer\Segmentation.seg.nrrd"
LAB13 = r"D:\AAA\project2-lung-seg\slicer\Segmentation-nodule_13.6mm-label.nrrd"
CT    = r"E:\luna16\1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"
SD    = "658609808059"
D_MM  = 4.343199739
OUT   = r"D:\AAA\project2-lung-seg\figs\seg_small_vs_circle.png"

ct_img = sitk.ReadImage(CT)
arr = sitk.GetArrayFromImage(ct_img)
sx, sy, sz = ct_img.GetSpacing()

union = sitk.GetArrayFromImage(sitk.ReadImage(LAB)) > 0
big = sitk.GetArrayFromImage(sitk.ReadImage(LAB13)) > 0
lab = union & ~big
print(f"并集 {int(union.sum())}，大结节 {int(big.sum())}，相减得小结节 {int(lab.sum())}")

vox = int(lab.sum())
vv = sx * sy * sz
vol = vox * vv
eq_d = (6 * vol / np.pi) ** (1 / 3)
R = D_MM / 2
sphere_v = 4 / 3 * np.pi * R ** 3
sphere_a = np.pi * R ** 2

layers = np.where(lab.any(axis=(1, 2)))[0]
areas = np.array([lab[z].sum() * sx * sy for z in layers])
zmax = int(layers[int(np.argmax(areas))])

print(f"\n半径像素 R = {R / sx:.2f} px（直径 {D_MM / sx:.1f} px）")
print(f"勾画：体素 {vox}，体积 {vol:.1f} mm3 = {vol / 1000:.4f} cm3，等效球直径 {eq_d:.2f} mm")
print(f"覆盖 {len(layers)} 层（{layers.min()}~{layers.max()}），最大层 {zmax}，面积 {areas.max():.1f} mm2")
print(f"\n--- 对比 ---")
print(f"名义球（直径 {D_MM:.2f} mm）：面积 {sphere_a:.1f} mm2，体积 {sphere_v:.1f} mm3")
print(f"人工勾画：体积 {vol:.1f} mm3")
print(f"差异：{vol - sphere_v:+.1f} mm3（{vol / sphere_v:.2f} 倍）")

roi = arr[lab]
print(f"\n勾画内 HU：min {roi.min()}  max {roi.max()}  mean {roi.mean():.1f}")

ann = pd.read_csv(os.path.join(os.path.dirname(CT), "annotations.csv"))
row = ann[(ann.seriesuid.str.endswith(SD)) & (ann.diameter_mm < 5)].iloc[0]
cx, cy, cz = ct_img.TransformPhysicalPointToIndex((row.coordX, row.coordY, row.coordZ))
r_px = R / sx

circ = np.zeros_like(lab[zmax])
yy, xx = np.ogrid[:circ.shape[0], :circ.shape[1]]
circ[((xx - cx) ** 2 + (yy - cy) ** 2) <= r_px ** 2] = True
d2 = 2 * (lab[zmax] & circ).sum() / (lab[zmax].sum() + circ.sum())
print(f"最大层 2D Dice（勾画 vs 直径圆）= {d2:.3f}")


def win(a, level=-600, width=1500):
    lo, hi = level - width / 2, level + width / 2
    return (np.clip(a, lo, hi) - lo) / (hi - lo)


S = 36
y0, y1, x0, x1 = cy - S, cy + S, cx - S, cx + S

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax in axes:
    ax.imshow(win(arr[zmax, y0:y1, x0:x1]), cmap="gray")
    ax.axis("off")

axes[0].add_patch(patches.Circle((cx - x0, cy - y0), r_px, fill=False, edgecolor="#ff5555", lw=1.5))
axes[0].set_title(f"直径圆（合成 GT）· layer {zmax}\n直径 {D_MM / sx:.1f} px")

axes[1].contour(lab[zmax, y0:y1, x0:x1], levels=[0.5], colors=["#55ff88"], linewidths=1.5)
axes[1].set_title(f"人工勾画 · layer {zmax}\n面积 {areas.max():.1f} mm2")

axes[2].contour(lab[zmax, y0:y1, x0:x1], levels=[0.5], colors=["#55ff88"], linewidths=1.5)
axes[2].add_patch(patches.Circle((cx - x0, cy - y0), r_px, fill=False, edgecolor="#ff5555", lw=1.5, ls="--"))
axes[2].set_title(f"叠加 · 2D Dice = {d2:.3f}")

fig.tight_layout()
fig.savefig(OUT, dpi=150)
plt.close(fig)
print("\n图 →", OUT)
