# 为勾画做导航：算出目标结节在第几层、中心像素、直径多少像素，并画一张带红圈的图
# 输出：终端 4 行 + D:\AAA\project2-lung-seg\figs\nav_<series>.png
import glob, os
import numpy as np, pandas as pd, SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

LUNA = r"E:\luna16"
SD   = "658609808059"          # 该 CT 有 2 个结节：4.3 mm 和 13.6 mm
OUT  = rf"D:\AAA\project2-lung-seg\figs\nav_{SD}.png"

# 文件名是完整 UID，所以按后缀匹配
cand = [p for p in glob.glob(os.path.join(LUNA, "*.mhd")) if os.path.basename(p).endswith(SD + ".mhd")]
mhd = cand[0] if cand else os.path.join(LUNA, SD + ".mhd")
if not os.path.exists(mhd):
    found = sorted(glob.glob(os.path.join(LUNA, "*.mhd")))
    print(f"没找到 {mhd}")
    print(f"E:\\luna16 里现有 .mhd 数量：{len(found)}")
    if found:
        print("前 3 个：", [os.path.basename(p) for p in found[:3]])
    raise SystemExit(1)

img = sitk.ReadImage(mhd)
arr = sitk.GetArrayFromImage(img)               # (z, y, x)
sx, sy, sz = img.GetSpacing()
print(f"CT: {SD}")
print(f"体积 {arr.shape}（z,y,x）  spacing {sx:.3f}/{sy:.3f}/{sz:.3f} mm")

ann = pd.read_csv(os.path.join(LUNA, "annotations.csv"))
sub = ann[ann["seriesuid"].str.endswith(SD)].sort_values("diameter_mm")
print(f"该 CT 的结节数：{len(sub)}")


def win(a, level, width):
    lo, hi = level - width / 2, level + width / 2
    return (np.clip(a, lo, hi) - lo) / (hi - lo)


print("\n--- 导航信息（拿去 Slicer / ITK-SNAP 里用）---")
for _, r in sub.iterrows():
    x, y, z = img.TransformPhysicalPointToIndex((r.coordX, r.coordY, r.coordZ))
    d_px = r.diameter_mm / sx
    print(f"直径 {r.diameter_mm:5.1f} mm → 第 {z} 层 / 共 {arr.shape[0]} 层，"
          f"中心 (x={x}, y={y})，直径 {d_px:.1f} px，半径 {d_px / 2:.1f} px")

# 画最大的那个
r = sub.iloc[-1]
x, y, z = img.TransformPhysicalPointToIndex((r.coordX, r.coordY, r.coordZ))
d_px = r.diameter_mm / sx
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(win(arr[z], -600, 1500), cmap="gray")
ax.add_patch(patches.Circle((x, y), d_px / 2, fill=False, edgecolor="r", lw=1.5))
ax.set_title(f"layer {z} / {arr.shape[0]}  ·  {r.diameter_mm:.1f} mm ~= {d_px:.0f} px (red = synthetic circle)")
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT, dpi=140)
plt.close(fig)
print("\n图 →", OUT)
