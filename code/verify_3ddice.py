# 复核 README 里的两个 3D Dice：理想球体 0.766 / 训练用 3 层圆盘 0.465
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

CT  = r"E:\luna16\1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"
LAB = r"D:\AAA\project2-lung-seg\slicer\Segmentation-nodule_13.6mm-label.nrrd"

ct = sitk.ReadImage(CT)
arr = sitk.GetArrayFromImage(ct)
sx, sy, sz = ct.GetSpacing()

lab_img = sitk.ReadImage(LAB)
if lab_img.GetSize() != ct.GetSize():
    lab_img = sitk.Resample(lab_img, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0)
man = sitk.GetArrayFromImage(lab_img) > 0
print(f"CT {arr.shape} spacing {sx:.3f}/{sy:.3f}/{sz:.3f}")
print(f"人工勾画 {int(man.sum())} 体素, 体积 {man.sum()*sx*sy*sz:.1f} mm3")

ann = pd.read_csv(os.path.join(os.path.dirname(CT), "annotations.csv"))
row = ann[(ann.seriesuid.str.endswith("658609808059")) & (ann.diameter_mm > 13)].iloc[0]
cx, cy, cz = ct.TransformPhysicalPointToIndex((row.coordX, row.coordY, row.coordZ))
D = float(row.diameter_mm)
R = D / 2
print(f"LUNA16 中心 (x={cx},y={cy},z={cz}) 直径 {D:.2f} mm, 半径 {R:.2f} mm")

zz, yy, xx = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]
dxy2 = ((xx - cx) * sx) ** 2 + ((yy - cy) * sy) ** 2
d2 = dxy2 + ((zz - cz) * sz) ** 2

sphere = d2 <= R ** 2
disk3 = (np.abs(zz - cz) <= 1) & (dxy2 <= R ** 2)


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else 0.0


for name, m in [("理想球体", sphere), ("3层圆盘 cz±1", disk3)]:
    print(f"{name:14s} 体素 {int(m.sum()):5d}  体积 {m.sum()*sx*sy*sz:7.1f} mm3  "
          f"与人工勾画 3D Dice = {dice(m, man):.3f}")

# 人工勾画实际跨层
layers = np.where(man.any(axis=(1, 2)))[0]
print(f"\n人工勾画覆盖层 {layers.min()}~{layers.max()}（{len(layers)} 层）")
print(f"训练取 3 层 = {cz-1}~{cz+1}，占人工层数 {3/len(layers)*100:.0f}%")

# 每层面积（mm2）
areas = np.array([man[z].sum() * sx * sy for z in layers])
print(f"每层面积 mm2: {[round(float(a),1) for a in areas]}")

