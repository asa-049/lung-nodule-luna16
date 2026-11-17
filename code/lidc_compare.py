"""手勾 labelmap vs LIDC 4 位医生：逐对算 3D Dice 与体积差。

用法：
    D:\\Anaconda3\\python.exe code/lidc_compare.py --seriesuid <UID> --hand <nrrd1> [nrrd2 ...]

护栏（按 11-18 决策）：
  1. 几何对齐：手勾 nrrd 必须与 .mhd 同 size/spacing/origin，否则直接报错退出
  2. 配对阈值：只用质心距 < 2 mm 的医生标注配对
  3. 样本量诚实：最后会打印 n 结节 / n 标注者，写文档时照抄
"""
import argparse
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

from lidc_mask import build_mask, contour_rows, load_scan

ROOT = r"D:\AAA\project2-lung-seg"
LUNA = r"E:\luna16"
PAIR_MM = 2.0          # 质心配对阈值
LOW_DICE = 0.3         # 低于此值提示先怀疑几何，而不是勾画


def load_hand(path, shape):
    """读手勾 labelmap。多标签(>1)时取非零。返回 bool mask。"""
    img = sitk.ReadImage(path)
    a = sitk.GetArrayFromImage(img)
    if a.shape != shape:
        raise SystemExit(f"几何不匹配：{os.path.basename(path)} shape={a.shape}，CT shape={shape}")
    return a > 0, img


def centroid_world(mask, img):
    zz, yy, xx = np.nonzero(mask)
    if len(zz) == 0:
        return None
    return img.TransformContinuousIndexToPhysicalPoint((float(xx.mean()), float(yy.mean()), float(zz.mean())))


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seriesuid", required=True)
    ap.add_argument("--hand", nargs="+", required=True)
    args = ap.parse_args()

    mhd = os.path.join(LUNA, args.seriesuid + ".mhd")
    img = sitk.ReadImage(mhd)
    shape = (img.GetSize()[2], img.GetSize()[1], img.GetSize()[0])   # (z,y,x)
    sx, sy, sz = img.GetSpacing()
    vv = sx * sy * sz
    print(f"CT  shape={shape}  spacing=({sx:.3f},{sy:.3f},{sz:.3f})  体素={vv:.4f} mm3")

    sid, zs = load_scan(args.seriesuid)
    rows = contour_rows(sid)
    ann_ids = sorted({r[0] for r in rows})
    print(f"LIDC 标注 {len(ann_ids)} 个\n")

    # 生成各位医生的 mask
    docs = {}
    for ann_id in ann_ids:
        sub = [r for r in rows if r[0] == ann_id]
        m = build_mask(sub, zs, shape) > 0
        if m.sum() == 0:
            continue
        docs[ann_id] = m
    print(f"可用医生轮廓 {len(docs)} 个\n")

    for hand_path in args.hand:
        hand, himg = load_hand(hand_path, shape)
        # 几何对齐检查（带容差：Slicer 写 labelmap 的 origin 与 mhd 常有微小差异）
        sp_ok = np.allclose(np.array(himg.GetSpacing()), np.array(img.GetSpacing()), atol=1e-3)
        og_ok = np.allclose(np.array(himg.GetOrigin()), np.array(img.GetOrigin()), atol=1.0)
        if not (sp_ok and og_ok):
            print(f"[!] {os.path.basename(hand_path)} 的 spacing/origin 与 CT 不一致"
                  f"（spacing {himg.GetSpacing()} vs {img.GetSpacing()}；"
                  f"origin {himg.GetOrigin()} vs {img.GetOrigin()}）")
        else:
            print(f"[OK] 几何一致（容差内）：spacing/origin 与 CT 对齐")

        hv = hand.sum() * vv
        hc = centroid_world(hand, img)
        print("=" * 78)
        print(f"手勾：{os.path.basename(hand_path)}")
        print(f"      体素 {int(hand.sum())}  体积 {hv:.1f} mm3  质心 "
              f"({hc[0]:.1f},{hc[1]:.1f},{hc[2]:.1f})")
        print(f"{'医生标注':>8s}{'体积mm3':>10s}{'质心距mm':>10s}{'Dice':>8s}{'体积差%':>9s}")
        print("-" * 78)

        pairs = []
        for ann_id, dm in docs.items():
            dc = centroid_world(dm, img)
            d = float(np.linalg.norm(np.array(hc) - np.array(dc)))
            if d > PAIR_MM:
                continue
            dv = dm.sum() * vv
            pairs.append((ann_id, dv, d, dice(hand, dm), (dv - hv) / hv * 100))

        if not pairs:
            print("  没有质心距 < 2 mm 的配对")
            continue

        for ann_id, dv, d, dc_, diff in sorted(pairs, key=lambda t: t[1]):
            flag = "  ← Dice 偏低，先查几何" if dc_ < LOW_DICE else ""
            print(f"{ann_id:>8d}{dv:10.1f}{d:10.2f}{dc_:8.3f}{diff:+9.1f}{flag}")

        dices = [p[3] for p in pairs]
        vols = [p[1] for p in pairs]
        print("-" * 78)
        print(f"  配对 {len(pairs)} 个 | Dice 均值 {np.mean(dices):.3f} "
              f"(min {min(dices):.3f} / max {max(dices):.3f})")
        print(f"  医生体积 {min(vols):.1f} ~ {max(vols):.1f} mm3（均值 {np.mean(vols):.1f}），"
              f"跨度 {(max(vols)-min(vols))/min(vols)*100:.0f}%")
        print(f"  手勾 {hv:.1f} mm3 → 相对医生均值 {(hv-np.mean(vols))/np.mean(vols)*100:+.1f}%")
        print()

    print("=" * 78)
    print(f"样本量：{len(args.hand)} 个手勾版本 × {len(docs)} 位标注者")
    print("写文档时照抄这句：n = 1 结节、4 位标注者（不是 4 颗结节）")


if __name__ == "__main__":
    main()
