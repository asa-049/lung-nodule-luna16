# 人工勾画 vs 直径圆：算体积、等效直径、每层面积，并出一张对比图
import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

LAB = r"D:\AAA\project2-lung-seg\slicer\Segmentation-nodule_13.6mm-label.nrrd"
CT  = r"E:\luna16\1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"
OUT = r"D:\AAA\project2-lung-seg\figs\seg_vs_circle.png"

lab_img = sitk.ReadImage(LAB)
ct_img = sitk.ReadImage(CT)
arr = sitk.GetArrayFromImage(ct_img)                  # (z, y, x)
sx, sy, sz = ct_img.GetSpacing()

if lab_img.GetSize() != ct_img.GetSize():
    lab_img = sitk.Resample(lab_img, ct_img, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0.0)
    print("[!] label 尺寸与 CT 不同，已重采样到 CT 几何")

lab = sitk.GetArrayFromImage(lab_img) > 0
print(f"CT   {arr.shape}  spacing {sx:.3f}/{sy:.3f}/{sz:.3f}")
print(f"label {lab.shape}  前景层数 {int((lab.any(axis=(1, 2))).sum())}")

vox = int(lab.sum())
vox_mm3 = sx * sy * sz
vol = vox * vox_mm3
eq_d = (6 * vol / np.pi) ** (1 / 3)
print(f"\n体素 {vox}   单个体素 {vox_mm3:.4f} mm3   体积 {vol:.1f} mm3 = {vol/1000:.3f} cm3")
print(f"等效球直径 {eq_d:.2f} mm")

# 每层面积
layers = np.where(lab.any(axis=(1, 2)))[0]
areas = np.array([lab[z].sum() * sx * sy for z in layers])
imax = int(np.argmax(areas))
zmax = int(layers[imax])
print(f"覆盖层 {layers.min()}~{layers.max()}（共 {len(layers)} 层）")
print(f"最大层 {zmax}，面积 {areas[imax]:.1f} mm2")

R_mm = 13.59647134 / 2
circle_mm2 = np.pi * R_mm ** 2
sphere_mm3 = 4 / 3 * np.pi * R_mm ** 3
print(f"\n--- 对比 ---")
print(f"等效球（直径 13.60 mm）: 面积 {circle_mm2:.1f} mm2, 体积 {sphere_mm3:.1f} mm3")
print(f"人工勾画                : 体积 {vol:.1f} mm3")
print(f"差异                    : {vol - sphere_mm3:+.1f} mm3 ({(vol/sphere_mm3 - 1)*100:+.1f}%)")

# 质心
zz, yy, xx = np.nonzero(lab)
print(f"质心 z={zz.mean():.1f} y={yy.mean():.1f} x={xx.mean():.1f}")
print(f"包围盒 z[{zz.min()}-{zz.max()}] y[{yy.min()}-{yy.max()}] x[{xx.min()}-{xx.max()}]")


def win(a, level=-600, width=1500):
    lo, hi = level - width / 2, level + width / 2
    return (np.clip(a, lo, hi) - lo) / (hi - lo)


# 直径圆的圆心来自 annotations（世界坐标→体素）
import pandas as pd
ann = pd.read_csv(os.path.join(os.path.dirname(CT), "annotations.csv"))
row = ann[(ann.seriesuid.str.endswith("658609808059")) &
          (ann.diameter_mm > 13)].iloc[0]
cx, cy, cz = ct_img.TransformPhysicalPointToIndex((row.coordX, row.coordY, row.coordZ))
r_px = R_mm / sx

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax in axes:
    ax.imshow(win(arr[zmax]), cmap="gray")
    ax.axis("off")

axes[0].add_patch(patches.Circle((cx, cy), r_px, fill=False, edgecolor="#ff5555", lw=1.8))
axes[0].set_title(f"直径圆（合成 GT）- layer {zmax}\nr={r_px:.1f}px")

axes[1].contour(lab[zmax], levels=[0.5], colors=["#55ff88"], linewidths=1.8)
axes[1].set_title(f"人工勾画 - layer {zmax}\narea={areas[imax]:.1f} mm2")

axes[2].contour(lab[zmax], levels=[0.5], colors=["#55ff88"], linewidths=1.8)
axes[2].add_patch(patches.Circle((cx, cy), r_px, fill=False, edgecolor="#ff5555", lw=1.8, ls="--"))
axes[2].set_title("叠加：绿=人工勾画  红=直径圆")

fig.tight_layout()
fig.savefig(OUT, dpi=130)
plt.close(fig)
print("\n图 →", OUT)
