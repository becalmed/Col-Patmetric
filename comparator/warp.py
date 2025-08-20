import numpy as np
import cv2
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree
from .boundary import extract_boundary_points

# ---- 1) 局部降采样（半径抑制 + 远点优先，避免同patch“内卷”） ----
def local_thin(P_src, P_tgt, center, k, radius):
    """
    在 center 周围从 (P_src->P_tgt) 选择最多 k 个、彼此至少 radius 的代表点。
    返回: 选中的索引数组 idx
    """
    if len(P_src) <= k:
        return np.arange(len(P_src))

    # 先按距离排序，远点优先可降低极近点内卷
    d = np.linalg.norm(P_src - center[None, :], axis=1)
    order = np.argsort(d)  # 近->远
    order = order[::-1]    # 远->近

    picked = []
    pts = P_src.copy()
    used = np.zeros(len(P_src), dtype=bool)
    tree = None

    for idx in order:
        if used[idx]:
            continue
        if len(picked) == 0:
            picked.append(idx)
            used[idx] = True
            if len(picked) >= k:
                break
            tree = cKDTree(P_src[picked])
            continue

        # 距已选点的最近距离要 >= radius
        if tree is not None:
            dist, _ = tree.query(P_src[idx], k=1)
            if dist < radius:
                continue
        picked.append(idx)
        used[idx] = True
        tree = cKDTree(P_src[picked])
        if len(picked) >= k:
            break

    if len(picked) == 0:
        picked = [int(np.argmin(d))]
    return np.array(picked, dtype=int)

# ---- 2) 鲁棒函数（Huber/Tukey），用于 IRLS ----
def huber_weights(r, c):
    """ r: 残差范数 (k,), c: 门限 """
    w = np.ones_like(r, dtype=float)
    mask = r > c
    w[mask] = c / (r[mask] + 1e-12)
    return w

def tukey_biweight(r, c):
    """ Tukey biweight (更强抑制离群) """
    u = r / (c + 1e-12)
    w = (1 - u**2)**2
    w[np.abs(u) >= 1] = 0.0
    return np.clip(w, 0.0, 1.0)

# ---- 3) alpha 的 KNN 平滑，避免权重抖动/局部过度集中 ----
def smooth_alpha(points, alpha, knn=16, step=0.5):
    tree = cKDTree(points)
    k = min(knn, len(points))
    d, idx = tree.query(points, k=k)
    if k == 1:
        return alpha
    neigh = alpha[idx]      # (N,k)
    mean_a = neigh.mean(axis=1)
    return (1 - step) * alpha + step * mean_a

# ---- 4) 位移夹逼：基于邻域自适应阈值，限制极端聚集 ----
def clamp_displacements(points, warped, knn=16, sigma=3.0):
    """
    对每个点的位移幅度做邻域自适应上限: median + sigma * MAD
    """
    disp = warped - points
    mag  = np.linalg.norm(disp, axis=1)  # (N,)
    tree = cKDTree(points)
    k = min(knn, len(points))
    _, idx = tree.query(points, k=k)
    if k == 1:
        return warped
    mags_nei = mag[idx]  # (N,k)
    med = np.median(mags_nei, axis=1)
    mad = np.median(np.abs(mags_nei - med[:, None]), axis=1) + 1e-9
    cap = med + sigma * mad
    scale = np.minimum(1.0, cap / (mag + 1e-12))
    disp_clamped = disp * scale[:, None]
    return points + disp_clamped

# ---- 5) 幽灵锚点：在掩膜内部自动撒点，作为 all_src->all_dst 的静止约束 ----
def augment_with_ghost_anchors(all_src, all_dst, ref_mask_gray, num=200, mindist=20):
    """
    在 ref_mask_gray(255=内) 内部撒 num 个锚点，目标=原位置，用于均匀化/稳定稀疏区。
    mindist: 与已有控制点/锚点的最小间距（像素）
    """
    if ref_mask_gray.ndim == 3:
        ref_mask_gray = cv2.cvtColor(ref_mask_gray, cv2.COLOR_BGR2GRAY)
    _, binm = cv2.threshold(ref_mask_gray, 127, 255, cv2.THRESH_BINARY)

    h, w = binm.shape[:2]
    ys, xs = np.nonzero(binm > 0)
    if len(xs) == 0:
        return all_src, all_dst

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    # 先在内部做一个较均匀的采样（简单用步进+随机挑选）
    step = max(4, int(mindist // 2))
    grid_x = np.arange(0, w, step)
    grid_y = np.arange(0, h, step)
    gyy, gxx = np.meshgrid(grid_y, grid_x, indexing="ij")
    grid = np.stack([gxx.ravel(), gyy.ravel()], axis=1).astype(np.float32)
    inside = binm[grid[:,1].astype(int), grid[:,0].astype(int)] > 0
    cand = grid[inside]

    if len(cand) == 0:
        return all_src, all_dst

    # 与已有控制点拉开距离
    base = all_src if len(all_src) > 0 else np.empty((0,2), dtype=np.float32)
    selected = []
    if len(base) > 0:
        tree = cKDTree(base)
    else:
        tree = None

    rng = np.random.default_rng(2025)
    order = rng.permutation(len(cand))
    for idx in order:
        p = cand[idx]
        ok = True
        if tree is not None:
            d, _ = tree.query(p, k=1)
            if d < mindist:
                ok = False
        if ok:
            selected.append(p)
            base = np.vstack([base, p[None, :]]) if len(base) else p[None, :]
            tree = cKDTree(base)
        if len(selected) >= num:
            break

    if len(selected) == 0:
        return all_src, all_dst

    ghosts = np.array(selected, dtype=np.float32)
    all_src2 = np.vstack([all_src, ghosts]) if len(all_src) else ghosts
    all_dst2 = np.vstack([all_dst, ghosts]) if len(all_dst) else ghosts
    return all_src2, all_dst2

def build_tps_with_constraints(filtered_src_pts, filtered_dst_pts,
                               ref_mask_gray, gen_mask_gray,
                               corner_config='adaptive',
                               method='corner_based',
                               n_boundary=30, smooth=0.1):
    """构建 TPS/RBF；把过滤后的匹配点与边界对齐点一起作为约束。"""
    if ref_mask_gray.ndim == 3:
        ref_mask_gray = cv2.cvtColor(ref_mask_gray, cv2.COLOR_BGR2GRAY)
    if gen_mask_gray.ndim == 3:
        gen_mask_gray = cv2.cvtColor(gen_mask_gray, cv2.COLOR_BGR2GRAY)
    _, ref_bin = cv2.threshold(ref_mask_gray, 127, 255, cv2.THRESH_BINARY)
    _, gen_bin = cv2.threshold(gen_mask_gray, 127, 255, cv2.THRESH_BINARY)

    b1, b2 = extract_boundary_points(ref_bin, gen_bin, num_points=n_boundary,
                                     method=method, corner_config=corner_config)

    if filtered_src_pts.size == 0 or filtered_dst_pts.size == 0:
        # 只有边界约束（兜底）
        all_src = b1
        all_dst = b2
    else:
        if len(b1) > 0 and len(b2) > 0:
            all_src = np.vstack([filtered_src_pts, b1])
            all_dst = np.vstack([filtered_dst_pts, b2])
        else:
            all_src = filtered_src_pts
            all_dst = filtered_dst_pts

    # RBF thin-plate spline
    rbf_x = Rbf(all_src[:, 0], all_src[:, 1], all_dst[:, 0], function='thin_plate', smooth=smooth)
    rbf_y = Rbf(all_src[:, 0], all_src[:, 1], all_dst[:, 1], function='thin_plate', smooth=smooth)
    return rbf_x, rbf_y, b1, b2

def map_points(rbf_x, rbf_y, pts_xy):
    """批量映射点坐标 (N,2) -> (N,2)"""
    xs = rbf_x(pts_xy[:, 0], pts_xy[:, 1])
    ys = rbf_y(pts_xy[:, 0], pts_xy[:, 1])
    return np.column_stack([xs, ys])

def apply_rbf_warp(points, rbf_x, rbf_y):
    """
    points: (N,2) 需要变形的查询点
    返回: (N,2) 经TPS/RBF的全局变形结果
    """
    px = points[:, 0]
    py = points[:, 1]
    gx = rbf_x(px, py)
    gy = rbf_y(px, py)
    return np.stack([gx, gy], axis=1)

def _mls_affine_single(v, Ps, Pt, w):
    """
    对单点 v ∈ R^2 用加权仿射MLS拟合：返回 v' 的绝对位置
    Ps: (k,2) 源控制点
    Pt: (k,2) 目标控制点
    w : (k,)  权重
    """
    wsum = np.sum(w)
    if wsum < 1e-12:
        return v.copy()

    x_bar = (w[:, None] * Ps).sum(axis=0) / wsum
    y_bar = (w[:, None] * Pt).sum(axis=0) / wsum
    X = Ps - x_bar
    Y = Pt - y_bar

    WX = w[:, None] * X
    M = X.T @ WX               # (2,2) = X^T W X
    N = X.T @ (w[:, None] * Y) # (2,2) = X^T W Y

    # 稳定正则
    M_reg = M + 1e-8 * np.eye(2)
    A = N @ np.linalg.inv(M_reg)
    t = y_bar - A @ x_bar
    return A @ v + t

def local_mls_points(points, all_src, all_dst, k=12, sigma=None, weight='gaussian'):
    """
    对 points 用局部MLS计算“绝对目标位置”
    all_src/all_dst: 作为局部拟合的控制对（建议= 过滤匹配点 + 边界点）
    """
    if len(all_src) == 0:
        return points.copy()

    tree = cKDTree(all_src)
    k = min(k, len(all_src))
    # 自动估计核宽
    if sigma is None:
        dd, _ = tree.query(points, k=min(8, len(all_src)))
        base = np.median(dd[:, -1]) if dd.ndim == 2 else np.median(dd)
        sigma = max(float(base), 1e-6)

    out = np.zeros_like(points, dtype=float)
    for i, v in enumerate(points):
        dists, idxs = tree.query(v, k=k)
        if k == 1:
            dists = np.array([dists]); idxs = np.array([idxs])
        Ps = all_src[idxs]
        Pt = all_dst[idxs]
        if weight == 'gaussian':
            w = np.exp(-0.5 * (dists / sigma) ** 2)
        else:
            w = 1.0 / (dists**2 + 1e-8)
        w = np.clip(w, 1e-6, None)
        out[i] = _mls_affine_single(v, Ps, Pt, w)
    return out

def _mls_affine_single_irls(v, Ps, Pt, w_init, iters=3, robust='huber', c=None):
    """
    IRLS 加权仿射 MLS: 对单点 v 求 v'。
    w_init: 距离权重（高斯/反距离）
    robust: 'huber' or 'tukey'
    c: 鲁棒门限(像素)，默认取 Ps->Pt 残差的中位尺度
    """
    w = w_init.copy()
    for _ in range(max(1, iters)):
        # 标准加权MLS解
        wsum = np.sum(w)
        if wsum < 1e-12:
            return v.copy()
        x_bar = (w[:, None] * Ps).sum(axis=0) / wsum
        y_bar = (w[:, None] * Pt).sum(axis=0) / wsum
        X = Ps - x_bar
        Y = Pt - y_bar
        WX = w[:, None] * X
        M = X.T @ WX
        N = X.T @ (w[:, None] * Y)
        M_reg = M + 1e-8 * np.eye(2)
        A = N @ np.linalg.inv(M_reg)
        t = y_bar - A @ x_bar
        v_pred = (A @ v + t)

        # 计算控制点的拟合残差，用于鲁棒重加权
        pred_Ps = (A @ Ps.T).T + t  # (k,2)
        res = np.linalg.norm(pred_Ps - Pt, axis=1)  # (k,)

        if c is None:
            med = np.median(res)
            mad = np.median(np.abs(res - med)) + 1e-6
            c_local = med + 2.5 * mad
        else:
            c_local = float(c)

        if robust == 'tukey':
            rw = tukey_biweight(res, c_local)
        else:
            rw = huber_weights(res, c_local)

        # 总权重 = 距离权重 * 鲁棒权重
        w = np.clip(w_init * rw, 1e-6, None)
    return v_pred

def local_mls_points_v2(points, all_src, all_dst, k=12, sigma=None,
                        weight='gaussian', do_thin=True, thin_radius=6.0,
                        irls_iters=3, robust='huber', robust_c=None):
    """
    升级版局部MLS：
      - 邻域降采样(local_thin)：避免同patch内卷
      - IRLS鲁棒：抑制离群匹配点
    """
    if len(all_src) == 0:
        return points.copy()

    N = len(points)
    out = np.zeros_like(points, dtype=float)
    tree = cKDTree(all_src)

    # 自动 sigma
    if sigma is None:
        dd, _ = tree.query(points, k=min(8, len(all_src)))
        base = np.median(dd[:, -1]) if dd.ndim == 2 else np.median(dd)
        sigma = max(float(base), 1e-6)

    k = min(k, len(all_src))
    for i, v in enumerate(points):
        dists, idxs = tree.query(v, k=k)
        if k == 1:
            dists = np.array([dists]); idxs = np.array([idxs])

        Ps = all_src[idxs]
        Pt = all_dst[idxs]

        # 局部降采
        if do_thin and len(Ps) > 3:
            sel = local_thin(Ps, Pt, v, k=min(k, len(Ps)), radius=thin_radius)
            Ps = Ps[sel]; Pt = Pt[sel]
            # 重新计算与 v 的距离（更准确）
            dists = np.linalg.norm(Ps - v[None, :], axis=1)

        # 距离权重
        if weight == 'gaussian':
            w_init = np.exp(-0.5 * (dists / sigma) ** 2)
        else:
            w_init = 1.0 / (dists**2 + 1e-8)
        w_init = np.clip(w_init, 1e-6, None)

        out[i] = _mls_affine_single_irls(
            v, Ps, Pt, w_init,
            iters=irls_iters, robust=robust, c=robust_c
        )
    return out

def estimate_density_alpha(points, all_src, k=32, sigma=None, p_lo=0.05, p_hi=0.95):
    """
    基于控制点密度的 alpha ∈ [0,1]：密集区 alpha→1（更信MLS），稀疏区 alpha→0（更信TPS）
    """
    if len(all_src) == 0:
        return np.zeros(len(points), dtype=float)
    tree = cKDTree(all_src)
    k = min(k, len(all_src))
    dists, _ = tree.query(points, k=k)
    if k == 1:
        dists = dists[:, None]
    if sigma is None:
        # 控制点空间的典型间距
        dd_ctrl, _ = tree.query(all_src, k=min(8, len(all_src)))
        base = np.median(dd_ctrl[:, -1]) if dd_ctrl.ndim == 2 else np.median(dd_ctrl)
        sigma = max(float(base), 1e-6)

    density = np.exp(-0.5 * (dists / sigma) ** 2).sum(axis=1)
    lo = np.quantile(density, p_lo)
    hi = np.quantile(density, p_hi)
    hi = max(hi, lo + 1e-9)
    alpha = (density - lo) / (hi - lo)
    return np.clip(alpha, 0.0, 1.0)

def make_all_constraints(filtered_src_pts, filtered_dst_pts,
                         ref_mask_gray, gen_mask_gray,
                         corner_config='adaptive', method='corner_based', n_boundary=30):
    if ref_mask_gray.ndim == 3:
        ref_mask_gray = cv2.cvtColor(ref_mask_gray, cv2.COLOR_BGR2GRAY)
    if gen_mask_gray.ndim == 3:
        gen_mask_gray = cv2.cvtColor(gen_mask_gray, cv2.COLOR_BGR2GRAY)
    _, ref_bin = cv2.threshold(ref_mask_gray, 127, 255, cv2.THRESH_BINARY)
    _, gen_bin = cv2.threshold(gen_mask_gray, 127, 255, cv2.THRESH_BINARY)

    b1, b2 = extract_boundary_points(ref_bin, gen_bin, num_points=n_boundary,
                                     method=method, corner_config=corner_config)

    if filtered_src_pts.size == 0 or filtered_dst_pts.size == 0:
        all_src = b1
        all_dst = b2
    else:
        if len(b1) > 0 and len(b2) > 0:
            all_src = np.vstack([filtered_src_pts, b1])
            all_dst = np.vstack([filtered_dst_pts, b2])
        else:
            all_src = filtered_src_pts
            all_dst = filtered_dst_pts
    return all_src, all_dst


def uniformized_warp_points(
    query_points,                 # (N,2) 网格顶点/像素坐标
    rbf_x, rbf_y,                 # 由 build_tps_with_constraints 返回
    all_src, all_dst,             # 约束点对（特征点+边界点）
    k_local=12,
    sigma_local=None,
    alpha_k=32,
    alpha_sigma=None,
    feather=False, feather_step=0.3, feather_iters=1,  # 可选羽化（简单邻域平均，点云无拓扑时不启用）
    neighbors_for_feather=8
):
    """
    返回：points_out (N,2)
    过程：TPS 全局 -> MLS 局部 -> 密度自适应融合
    """
    # 全局 TPS
    pts_global = apply_rbf_warp(query_points, rbf_x, rbf_y)
    # 局部 MLS
    pts_local  = local_mls_points(query_points, all_src, all_dst, k=k_local, sigma=sigma_local, weight='gaussian')
    # 密度权重
    alpha = estimate_density_alpha(query_points, all_src, k=alpha_k, sigma=alpha_sigma)

    pts_out = (alpha[:, None] * pts_local) + ((1.0 - alpha)[:, None] * pts_global)

    # 简易羽化（无网格拓扑时，用点邻域平均一两次平滑；若有网格结构，建议换成Laplacian）
    if feather:
        tree = cKDTree(pts_out)
        for _ in range(max(0, int(feather_iters))):
            dists, idxs = tree.query(pts_out, k=min(neighbors_for_feather, len(pts_out)))
            if neighbors_for_feather == 1:
                idxs = idxs[:, None]
            neigh = pts_out[idxs]         # (N,k,2)
            meann = neigh.mean(axis=1)    # (N,2)
            pts_out = (1.0 - feather_step) * pts_out + feather_step * meann
    return pts_out, alpha

def uniformized_warp_points_v2(
    query_points,
    rbf_x, rbf_y,
    all_src, all_dst,
    k_local=12,
    sigma_local=None,
    alpha_k=32, alpha_sigma=None,
    alpha_smooth_knn=16, alpha_smooth_step=0.5,
    do_thin=True, thin_radius=6.0,
    irls_iters=3, robust='huber', robust_c=None,
    do_clamp=True, clamp_knn=16, clamp_sigma=3.0
):
    """
    TPS(全局) + MLS(IRLS,降采) 融合 + alpha平滑 + 位移夹逼
    """
    # 全局
    pts_global = apply_rbf_warp(query_points, rbf_x, rbf_y)

    # 局部（升级版）
    pts_local  = local_mls_points_v2(
        query_points, all_src, all_dst,
        k=k_local, sigma=sigma_local,
        weight='gaussian',
        do_thin=do_thin, thin_radius=thin_radius,
        irls_iters=irls_iters, robust=robust, robust_c=robust_c
    )

    # 密度 alpha
    alpha = estimate_density_alpha(query_points, all_src, k=alpha_k, sigma=alpha_sigma)
    if alpha_smooth_knn > 1 and alpha_smooth_step > 0:
        alpha = smooth_alpha(query_points, alpha, knn=alpha_smooth_knn, step=alpha_smooth_step)

    pts = (alpha[:, None] * pts_local) + ((1.0 - alpha)[:, None] * pts_global)

    # 位移夹逼（抑制局部聚集/爆炸）
    if do_clamp:
        pts = clamp_displacements(query_points, pts, knn=clamp_knn, sigma=clamp_sigma)
    return pts, alpha