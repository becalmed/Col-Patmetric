import cv2
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from skimage.color import rgb2lab, deltaE_ciede2000
from .visualization import _visualize_palette_matching,_visualize_palette_proportion

def _to_lab_bgr(img_bgr):
    """BGR -> LAB，返回 float64，范围与 skimage 一致（L≈0-100，a/b≈-128~127）"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = rgb2lab(img_rgb)
    return lab

def _masked_lab_pixels(img_bgr, mask):
    """提取 mask 内的 Lab 像素，返回 N x 3"""
    assert img_bgr.shape[:2] == mask.shape[:2], "mask 与图像尺寸不一致"
    lab = _to_lab_bgr(img_bgr)
    idx = mask > 0
    if idx.sum() == 0:
        return np.empty((0, 3), dtype=np.float64)
    return lab[idx].reshape(-1, 3)

def _per_channel_wasserstein(lab1, lab2, bins=64):
    """
    LAB 每通道 1D 直方图 + Wasserstein 距离。
    若 scipy 不可用，回退到归一化 L1 直方图距离。
    返回 dict: {'L': dL, 'a': da, 'b': db, 'mean': avg}
    """
    res = {}
    channels = ['L', 'a', 'b']
    # 经验范围：L[0,100]，a/b[-128,127]
    ranges = [(0, 100), (-128, 127), (-128, 127)]
    vals1 = [lab1[:, i] if lab1.size else np.array([]) for i in range(3)]
    vals2 = [lab2[:, i] if lab2.size else np.array([]) for i in range(3)]

    dists = []
    for i, (lo, hi) in enumerate(ranges):
        h1, edges = np.histogram(vals1[i], bins=bins, range=(lo, hi), density=True)
        h2, _     = np.histogram(vals2[i], bins=bins, range=(lo, hi), density=True)
        centers = 0.5*(edges[:-1] + edges[1:])
        # 用连续位置 + 概率权重的 Wasserstein-1
        d = wasserstein_distance(centers, centers, u_weights=h1, v_weights=h2)
        dists.append(d)
        res[channels[i]] = float(d)

    res['mean'] = float(np.mean(dists))
    return res

def _pairwise_deltaE(c1, c2):
    """
    计算两组 Lab 中心的两两 ΔE2000，返回 (n1,n2) 矩阵
    c1: (n1,3), c2: (n2,3)
    """
    if c1.size == 0 or c2.size == 0:
        return np.zeros((len(c1), len(c2)), dtype=np.float64)
    n1, n2 = len(c1), len(c2)
    D = np.empty((n1, n2), dtype=np.float64)
    for i in range(n1):
        a = np.repeat(c1[i][None, :], n2, axis=0)
        D[i, :] = deltaE_ciede2000(a, c2)
    return D

def _kmeans_centers_weights(pixels, K,attempts=2, seed=42):
    if len(pixels) == 0:
        return np.zeros((0,3)), np.zeros((0,))
    # 子采样，避免超大 N
    N = len(pixels)
    take = min(N, 20000)
    sel = pixels[np.random.RandomState(seed).choice(N, take, replace=False)]
    data = sel.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    compactness, labels, centers = cv2.kmeans(
        data, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.float64)
    # 权重按全体（非仅子样）再聚类一次标签不现实；近似用子样分布
    weights = np.bincount(labels.flatten(), minlength=K).astype(np.float64)
    weights = weights / (weights.sum() + 1e-12)
    # 剔除极小簇，避免噪声主导
    keep = weights > (1.0 / (K * 20))
    return centers[keep], weights[keep]

def _palette_deltaE_with_details(img_ref_bgr, img_gen_bgr, mask_ref, mask_gen, K=6):
    L1 = _masked_lab_pixels(img_ref_bgr, mask_ref)
    L2 = _masked_lab_pixels(img_gen_bgr, mask_gen)
    c1, w1 = _kmeans_centers_weights(L1, K=K)
    c2, w2 = _kmeans_centers_weights(L2, K=K)
    if len(c1) == 0 or len(c2) == 0:
        return {
            "palette_deltaE": None, "c1": c1, "w1": w1, "c2": c2, "w2": w2,
            "assign": (np.array([], dtype=int), np.array([], dtype=int)),
            "pair_de": np.array([]), "D": np.zeros((len(c1), len(c2)))
        }
    D = _pairwise_deltaE(c1, c2)
    W = 0.5 * (w1[:, None] + w2[None, :])
    C = D * W
    rids, cids = linear_sum_assignment(C)
    pair_de = D[rids, cids]
    palette_deltaE = float((pair_de * w1[rids]).sum() / (w1[rids].sum() + 1e-12))
    return {
        "palette_deltaE": palette_deltaE, "c1": c1, "w1": w1, "c2": c2, "w2": w2,
        "assign": (rids, cids), "pair_de": pair_de, "D": D
    }


def _proj_axis_lab(centers):
    """
    生成将 Lab 映射到1D上的投影轴：用 (a,b) 的主方向 + 轻微引入 L。
    这样排序会更接近“色相-色度”的顺序，同时兼顾亮度。
    """
    if len(centers) == 0:
        return np.array([0.2, 0.6, 0.2])  # 兜底
    A = centers[:, 1:3]  # a,b
    A = A - A.mean(0, keepdims=True)
    # PCA一主成分
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    ab_dir = vh[0]  # shape (2,)
    # 把L也引入一点点，避免全是同色不同亮度时排序退化
    axis = np.array([0.2, ab_dir[0]*0.8, ab_dir[1]*0.8])
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return axis

def _palette_deltaE_proportion(c1, w1, c2, w2):
    """
    按“颜色占比”进行配对：将两个调色盘按同一投影轴排序，
    用累计权重做分位对齐，对每一小段计算 ΔE2000 并按质量加权平均。
    返回加权平均 ΔE2000。
    """
    if len(c1) == 0 or len(c2) == 0:
        return None

    # 归一化权重
    w1 = w1 / (w1.sum() + 1e-12)
    w2 = w2 / (w2.sum() + 1e-12)

    # 选择共同的排序轴（用 c1∪c2 拟合）
    allc = np.concatenate([c1, c2], axis=0)
    axis = _proj_axis_lab(allc)

    # 沿轴投影并排序
    s1 = (c1 @ axis)
    s2 = (c2 @ axis)
    o1 = np.argsort(s1)  # 上方色带顺序
    o2 = np.argsort(s2)  # 下方色带顺序

    c1, w1 = c1[o1], w1[o1]
    c2, w2 = c2[o2], w2[o2]

    i, j = 0, 0
    r1, r2 = w1.copy(), w2.copy()   # 剩余质量
    num = 0.0  # 加权ΔE的分子
    den = 0.0  # 加权ΔE的分母（总配对质量）

    while i < len(c1) and j < len(c2):
        # 当前可配的最小质量
        m = r1[i] if r1[i] <= r2[j] else r2[j]
        if m <= 0:
            if r1[i] <= 0: i += 1
            if r2[j] <= 0: j += 1
            continue
        # 计算当前小段的 ΔE
        de = float(deltaE_ciede2000(c1[i][None, :], c2[j][None, :])[0])
        num += de * m
        den += m
        # 扣减剩余质量
        r1[i] -= m
        r2[j] -= m
        # 谁耗尽就前进
        if r1[i] <= 1e-12: i += 1
        if r2[j] <= 1e-12: j += 1

    return num / (den + 1e-12)

def _palette_details_common(img_ref_bgr, img_gen_bgr, mask_ref, mask_gen, K=6):
    """抽取KMeans中心与权重，供两种匹配策略使用。"""
    L1 = _masked_lab_pixels(img_ref_bgr, mask_ref)
    L2 = _masked_lab_pixels(img_gen_bgr, mask_gen)
    c1, w1 = _kmeans_centers_weights(L1, K=K)
    c2, w2 = _kmeans_centers_weights(L2, K=K)
    return c1, w1, c2, w2

def robust_cloth_color_diff(img_ref_bgr, img_gen_bgr, mask_ref, mask_gen,
                            bins=64, K=6, return_palette=False,
                            palette_vis_path=None,palette_match="hungarian"):
    """
    return:
    {
      'hist_wasserstein': {...},
      'palette_deltaE': float | None,
      'mean_color_deltaE': float,
      # 当 return_palette=True 时，额外返回：
      'palette_details': {...},
      'palette_vis_path': 'xxx.png' | None
    }
    """
    L1 = _masked_lab_pixels(img_ref_bgr, mask_ref)
    L2 = _masked_lab_pixels(img_gen_bgr, mask_gen)

    # 1) 分布漂移
    hist_dist = _per_channel_wasserstein(L1, L2, bins=bins)

    # 2) 主色调差 + 可视化细节
    palette_details = _palette_deltaE_with_details(img_ref_bgr, img_gen_bgr, mask_ref, mask_gen, K=K)
    palette_de = palette_details["palette_deltaE"]

    # 3) 全局均色差
    mean1 = L1.mean(axis=0) if len(L1) else np.zeros(3)
    mean2 = L2.mean(axis=0) if len(L2) else np.zeros(3)
    mean_color_de = float(deltaE_ciede2000(mean1.reshape(1,3), mean2.reshape(1,3))[0])

    out = {
        "hist_wasserstein": hist_dist,
        "palette_deltaE": palette_de,
        "mean_color_deltaE": mean_color_de
    }
    if return_palette:
        vis_path = None
        if palette_vis_path is not None:
            # 只有当 matplotlib 存在并且我们确实有聚类结果时才出图
            if plt is not None and len(palette_details["c1"]) and len(palette_details["c2"]):
                vis_path = _visualize_palette_matching(palette_details, save_path=palette_vis_path)
        out["palette_details"] = palette_details
        out["palette_vis_path"] = vis_path
    return out

if __name__ == "__main__":
    ref = cv2.imread("ref.jpg")
    gen = cv2.imread("gen.jpg")
    mask_ref = cv2.imread("ref_mask.png", cv2.IMREAD_GRAYSCALE)
    mask_gen = cv2.imread("gen_mask.png", cv2.IMREAD_GRAYSCALE)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_OPEN, k)
    mask_gen = cv2.morphologyEx(mask_gen, cv2.MORPH_OPEN, k)

    scores = robust_cloth_color_diff(ref, gen, mask_ref, mask_gen, bins=64, K=6)
    print(scores)
