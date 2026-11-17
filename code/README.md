# code/ · 分析脚本

> 这些脚本产出 README 里的全部数字。2026-11-12 从 `_tmp\` 归位，路径已改到项目内，**已实跑验证**。
> 环境：`D:\Anaconda3\python.exe`（MONAI 1.6 / SimpleITK / pandas / scipy / matplotlib 都在这里）

## 换机器时要改的 3 个常量

每个脚本头部都有，改这三行就能跑：

```python
LUNA = r"E:\luna16"                                   # LUNA16 的 mhd/raw 目录
ROOT = r"D:\AAA\project2-lung-seg"                    # 本项目目录
CT   = r"E:\luna16\...808059.mhd"                     # 具体那例 CT
```

数据不在这台机器上时，`E:\luna16` 换成你的 LUNA16 目录即可；`slicer\` 和 `figs\` 是项目内相对位置。

## 脚本清单

| 脚本 | 做什么 | 输出 | 关键数字 |
|---|---|---|---|
| `nav_nodule.py` | 定位 CT `808059` 的两颗结节（层号、中心像素、像素直径） | 终端 + `figs/nav_658609808059.png` | 13.6 mm→第 68 层；4.3 mm→第 57 层 |
| `nav_nodule3.py` | 定位 `478441` 的 9.26 mm 结节 | 终端 + `figs/nav_478441.png` | 第 161 层，中心 (176, 274) |
| `compare_seg.py` | 人工勾画 vs 直径球（体积、质心、包围盒） | `figs/seg_vs_circle.png` | 1628.8 vs 1316.1 mm³（**+23.8%**） |
| `verify_3ddice.py` | **三维 Dice**：人工勾画 vs 理想球 / vs 训练用的 3 层圆盘 | 终端 | **0.766 / 0.465** |
| `thr_volumes.py` | 同一小结节按不同 HU 阈值算体积 | `figs/threshold_volumes.csv` | −400→−100：**126.5→10.9 mm³** |
| `volume_curve.py` | 体积–阈值曲线，反推等效直径对应的阈值 | `figs/volume_vs_threshold.png` | 42.9 mm³ ≈ **−230 HU** |
| `compare_seg_small.py` | 小结节勾画 vs 直径圆（含 2D Dice） | `figs/seg_small_vs_circle.png` | 小结节 2D Dice 0.559 |
| `small_three.py` | 多个小结节版本并排对比 | `figs/small_three_versions.png` | 39.2 / 66.6 mm³ 等 |

## 运行

```powershell
cd "D:\AAA\project2-lung-seg\code"
D:\Anaconda3\python.exe verify_3ddice.py
```

## 2026-11-12 实跑验证结果（与 README 一致）

```
verify_3ddice.py   理想球 3539 体素 / 1332.5 mm³ → Dice 0.766
                   3 层圆盘 1467 体素 / 552.3 mm³ → Dice 0.465
                   人工勾画跨 64~72 共 9 层，训练取 3 层 = 33%
compare_seg.py     人工 1628.8 mm³，偏差 +312.7 mm³（+23.8%）
thr_volumes.py     126.5 / 74.2 / 35.8 / 10.9 mm³（−400→−100）
```

## 注意

- **SimpleITK 读不了中文路径**（`fopen failed`）→ 所以项目目录已改名为英文 `project2-lung-seg`，别再改回中文。
- 部分脚本依赖 `slicer\` 里的 labelmap。如果那批文件被覆盖，先重跑 `nav_*.py` 确认坐标，再从 Slicer 重新导出。
- `compare_seg_small.py` 默认读 `Segmentation.seg.nrrd`，但该文件里的段可能已被覆盖；**要复现小结节数字，先确认段还在**。
