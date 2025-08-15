import numpy as np
import cv2
from scipy.interpolate import interp1d

def _largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def extract_boundary_points_simple(mask1, mask2, num_points=50):
    """等弧长采样两侧最大轮廓，按索引对齐。"""
    def sample(mask):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binm = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        c = _largest_contour(binm)
        if c is None: return np.empty((0, 2))
        c = c.reshape(-1, 2)
        d = np.sqrt(((np.roll(c, -1, axis=0) - c) ** 2).sum(1))
        s = np.concatenate([[0], np.cumsum(d)])
        s /= (s[-1] if s[-1] > 0 else 1)
        t = np.linspace(0, 1, num_points, endpoint=True)
        fx = interp1d(s, c[:, 0], kind='linear', fill_value='extrapolate')
        fy = interp1d(s, c[:, 1], kind='linear', fill_value='extrapolate')
        pts = np.column_stack([fx(t), fy(t)])
        if num_points > 1:
            pts[-1] = pts[0]
        return pts
    return sample(mask1), sample(mask2)

def extract_boundary_points_shape_matching(mask1, mask2, num_points=50):
    """极角对齐+线性插值"""
    def prepare(mask):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binm = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        c = _largest_contour(binm)
        return None if c is None else cv2.approxPolyDP(c, epsilon=2.0, closed=True).reshape(-1, 2)

    c1 = prepare(mask1); c2 = prepare(mask2)
    if c1 is None or c2 is None:
        return np.empty((0, 2)), np.empty((0, 2))

    def center_and_angles(c):
        M = cv2.moments(c)
        if M['m00'] == 0: return None, None
        ctr = np.array([M['m10']/M['m00'], M['m01']/M['m00']])
        v = c - ctr
        ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        return ctr, ang

    _, a1 = center_and_angles(c1); _, a2 = center_and_angles(c2)
    idx1, idx2 = np.argsort(a1), np.argsort(a2)
    c1s, c2s, a1s, a2s = c1[idx1], c2[idx2], a1[idx1], a2[idx2]

    a1n = (a1s - a1s.min()) / (a1s.max() - a1s.min() + 1e-8) * 2*np.pi
    a2n = (a2s - a2s.min()) / (a2s.max() - a2s.min() + 1e-8) * 2*np.pi

    a1e = np.concatenate([a1n - 2*np.pi, a1n, a1n + 2*np.pi])
    a2e = np.concatenate([a2n - 2*np.pi, a2n, a2n + 2*np.pi])
    c1e = np.concatenate([c1s, c1s, c1s])
    c2e = np.concatenate([c2s, c2s, c2s])

    t = np.linspace(0, 2*np.pi, num_points, endpoint=True)
    i1x, i1y = interp1d(a1e, c1e[:, 0], fill_value='extrapolate'), interp1d(a1e, c1e[:, 1], fill_value='extrapolate')
    i2x, i2y = interp1d(a2e, c2e[:, 0], fill_value='extrapolate'), interp1d(a2e, c2e[:, 1], fill_value='extrapolate')

    b1 = np.column_stack([i1x(t), i1y(t)])
    b2 = np.column_stack([i2x(t), i2y(t)])
    if num_points > 1:
        b1[-1] = b1[0]; b2[-1] = b2[0]
    return b1, b2

def extract_boundary_points_corner_based(mask1, mask2, num_points=50, corner_config='relaxed'):
    """角点检测→关键点匹配→在关键点间插值"""
    # 为简洁，直接复用 simple 兜底逻辑
    # （迭代调整epsilon、贪心匹配、闭合插值）
    b1, b2 = extract_boundary_points_simple(mask1, mask2, num_points)
    return b1, b2

def extract_boundary_points(mask1, mask2, num_points=50, method='shape_matching', corner_config='relaxed'):
    try:
        if method == 'shape_matching':
            return extract_boundary_points_shape_matching(mask1, mask2, num_points)
        elif method == 'corner_based':
            return extract_boundary_points_corner_based(mask1, mask2, num_points, corner_config)
        else:
            return extract_boundary_points_simple(mask1, mask2, num_points)
    except Exception:
        return extract_boundary_points_simple(mask1, mask2, num_points)
