# patchssim.py
import os
import cv2
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure

def ssim(img1, img2):
    metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    return metric(img1, img2)

def compute_patch_ssim_from_matches(
    ref_img,
    gen_img,
    kp1,
    kp2,
    filtered_matches,
    patch_size=32,
    stride=None,
    save_dir=None,
    save_only_matched=True,
):
    """
    基于 filtered 匹配点进行空间对齐的小块 SSIM 计算，并可选地将 patch 对保存下来。

    参数
    ----
    ref_img : np.ndarray
        参考图，灰度或 BGR，要求与 gen_img 尺寸一致。
    gen_img : np.ndarray
        生成图，灰度或 BGR，要求与 ref_img 尺寸一致。
    kp1 : list[cv2.KeyPoint]
        参考图上的特征点（与 filtered_matches.queryIdx 对应）。
    kp2 : list[cv2.KeyPoint]
        生成图上的特征点（与 filtered_matches.trainIdx 对应）。
    filtered_matches : list[cv2.DMatch]
        已经过滤后的匹配结果（例如 flann_match_and_filter 返回的 filtered）。
    patch_size : int
        小块边长（正方形 patch）。
    stride : int or None
        小块滑动步长，默认为 patch_size（即不重叠格子）。
    save_dir : str or None
        若不为 None，则在该目录下保存 patch 图像：
        save_dir/ref  下保存参考图 patch，
        save_dir/gen  下保存生成图 patch。
    save_only_matched : bool
        若为 True，则仅保存“包含匹配点”的 patch 对；
        若为 False，则保存所有网格上的 patch 对。

    返回
    ----
    mean_ssim : float
        所有有效 patch 的 SSIM 均值。
    ssim_list : list[float]
        每个有效 patch 的 SSIM 值列表（顺序按网格遍历）。
    """
    if stride is None:
        stride = patch_size

    # ---- 确保是灰度图 ----
    if ref_img.ndim == 3:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_img.copy()

    if gen_img.ndim == 3:
        gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
    else:
        gen_gray = gen_img.copy()

    if ref_gray.shape != gen_gray.shape:
        raise ValueError(
            f"ref_img shape {ref_gray.shape} != gen_img shape {gen_gray.shape}, "
            "请先在外部保证两张图大小一致。"
        )

    h, w = ref_gray.shape[:2]
    half = patch_size // 2

    # ---- 先把匹配点的坐标取出来，方便后面快速过滤 ----
    # 每项为 (x1, y1, x2, y2)
    match_pairs = []
    for m in filtered_matches:
        p1 = kp1[m.queryIdx].pt  # (x, y) in ref
        p2 = kp2[m.trainIdx].pt  # (x, y) in gen
        match_pairs.append((p1[0], p1[1], p2[0], p2[1]))

    ssim_vals = []

    # 如果需要保存 patch，先建好目录
    if save_dir is not None:
        ref_patch_dir = os.path.join(save_dir, "ref")
        gen_patch_dir = os.path.join(save_dir, "gen")
        os.makedirs(ref_patch_dir, exist_ok=True)
        os.makedirs(gen_patch_dir, exist_ok=True)
        patch_idx = 0  # 仅统计实际保存的 patch 对数量
    else:
        patch_idx = None  # 占位

    # ---- 遍历参考图上的 patch 网格 ----
    for y0 in range(0, h - patch_size + 1, stride):
        for x0 in range(0, w - patch_size + 1, stride):
            # ref patch 的中心
            cx = x0 + half
            cy = y0 + half

            # 找到落在当前 patch 中的匹配点（以 ref 坐标判断）
            local_pairs = [
                p for p in match_pairs
                if (x0 <= p[0] < x0 + patch_size) and (y0 <= p[1] < y0 + patch_size)
            ]

            # 默认情况：用同一空间位置对齐
            x1_gen = x0
            y1_gen = y0

            if local_pairs:
                # 用局部匹配点的空间关系估计平移：dx = x2 - x1, dy = y2 - y1
                dxs = [p[2] - p[0] for p in local_pairs]
                dys = [p[3] - p[1] for p in local_pairs]

                # 用中位数，抗 outlier
                dx = float(np.median(dxs))
                dy = float(np.median(dys))

                # 以中心点为基准进行平移
                cx2 = cx + dx
                cy2 = cy + dy

                # 换成生成图上 patch 左上角坐标
                x1_gen = int(round(cx2 - half))
                y1_gen = int(round(cy2 - half))

                # 如果越界，则退回到“同一位置对齐”的策略
                if (x1_gen < 0 or y1_gen < 0 or
                    x1_gen + patch_size > w or
                    y1_gen + patch_size > h):
                    x1_gen = x0
                    y1_gen = y0

            # 取出两个 patch
            ref_patch = ref_gray[y0:y0 + patch_size, x0:x0 + patch_size]
            gen_patch = gen_gray[y1_gen:y1_gen + patch_size,
                                 x1_gen:x1_gen + patch_size]

            # 理论上尺寸应该刚好是 patch_size×patch_size，再防一下越界
            if ref_patch.shape != (patch_size, patch_size) or \
               gen_patch.shape != (patch_size, patch_size):
                continue

            # 计算 SSIM
            val = ssim(ref_patch, gen_patch)
            ssim_vals.append(float(val))

            # ---- 保存 patch 对（可选）----
            if save_dir is not None:
                # save_only_matched=True 时，只保存包含匹配点的 patch
                if (not save_only_matched) or local_pairs:
                    ref_name = f"patch_{patch_idx:05d}_ref_x{x0}_y{y0}.png"
                    gen_name = f"patch_{patch_idx:05d}_gen_x{x1_gen}_y{y1_gen}.png"

                    cv2.imwrite(os.path.join(ref_patch_dir, ref_name), ref_patch)
                    cv2.imwrite(os.path.join(gen_patch_dir, gen_name), gen_patch)

                    patch_idx += 1

    if not ssim_vals:
        # 如果一个 patch 都没成功计算（极端情况），给个 NaN
        return float("nan"), []

    mean_ssim = float(np.mean(ssim_vals))
    return mean_ssim, ssim_vals
