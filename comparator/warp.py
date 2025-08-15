import numpy as np
import cv2
from scipy.interpolate import Rbf
from .boundary import extract_boundary_points

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
