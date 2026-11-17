# 体积-阈值曲线：同一个结节，边界阈值每挪一点，体积变多少
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

CT = r"E:\luna16\1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"
SD = "658609808059"
D_MM = 4.343199739
HALF_XY, HALF_Z = 10, 3
OUT = r"D:\AAA\project2-lung-seg\figs\volume_vs_threshold.png"

im = sitk.ReadImage(CT)
a = sitk.GetArrayFromImage(im).astype(np.float32)
sx, sy, sz = im.GetSpacing()
vv = sx * sy * sz

ann = pd.read_csv(os.path.join(os.path.dirname(CT), "annotations.csv"))
row = ann[(ann.seriesuid.str.endswith(SD)) & (ann.diameter_mm < 5)].iloc[0]
cx, cy, cz = im.TransformPhysicalPointToIndex((row.coordX, row.coordY, row.coordZ))
sub = a[cz - HALF_Z:cz + HALF_Z + 1, cy - HALF_XY:cy + HALF_XY + 1, cx - HALF_XY:cx + HALF_XY + 1]

ths = np.arange(-500, -39, 10)
vols = []
for lo in ths:
    m = (sub >= lo) & (sub <= 250)
    if m.sum() == 0:
        vols.append(0.0)
        continue
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    vols.append(m.sum() * vv)
vols = np.array(vols)

R = D_MM / 2
sphere = 4 / 3 * np.pi * R ** 3

# 反推：体积等于名义球时的阈值
idx = np.argmin(np.abs(vols - sphere))
th_at_sphere = ths[idx]
print(f"名义球 {sphere:.1f} mm3 对应的阈值 ≈ {th_at_sphere} HU（该阈值下体积 {vols[idx]:.1f}）")

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(ths, vols, lw=2, color="#2b7bba")
ax.axhline(sphere, ls="--", color="#ff5555", lw=1.5)
ax.text(-495, sphere + 4, f"名义球（等效直径 4.34 mm）= {sphere:.1f} mm3", color="#ff5555", fontsize=9)
ax.axvline(th_at_sphere, ls=":", color="#ff5555", lw=1)
ax.annotate(f"≈ {th_at_sphere} HU", (th_at_sphere, sphere), textcoords="offset points",
            xytext=(6, -14), fontsize=9, color="#ff5555")

for v, name, c in [(92.2, "手勾 v1", "#55aa55"), (66.6, "手勾 v2", "#33aa33"), (10.2, "阈值 -100（Slicer）", "#999999")]:
    ax.axhline(v, ls="-.", color=c, lw=1, alpha=.8)
    ax.text(-495, v + 2, f"{name} = {v} mm3", color=c, fontsize=8)

ax.set_xlabel("HU 阈值下界（上界固定 250）")
ax.set_ylabel("体积 mm3")
ax.set_title("同一颗 4.3 mm 结节：体积完全由“边界阈值”决定")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
plt.close(fig)
print("图 →", OUT)
