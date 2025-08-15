import os
import cv2
import numpy as np
from cv2 import FlannBasedMatcher

def flann_match_and_filter(kp1, des1, kp2, des2, ref_gray, gen_gray, output_dir,
                           ratio_thresh=0.75, angle_deg=5.0, scale_tol=0.15):
    """FLANN KNN + 比值检验 + 方向/尺度过滤；保存两张可视化图片。"""
    flann = FlannBasedMatcher(dict(algorithm=1, trees=8), dict(checks=100))
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for (m, n) in matches if m.distance < ratio_thresh * n.distance]

    if len(good) < 4:
        raise ValueError("匹配对过少，无法进行方向/尺度过滤")

    cv2.imwrite(os.path.join(output_dir, 'good_r-good-matches.png'),
                cv2.drawMatches(ref_gray, kp1, gen_gray, kp2, good, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

    deltas = []
    w = ref_gray.shape[1]
    for m in good:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        dx, dy = (x2 + w) - x1, y2 - y1  # 拼接可视化坐标修正
        theta = np.arctan2(dy, dx)
        r = np.hypot(dx, dy)
        deltas.append((theta, r))

    thetas, rs = np.array([d[0] for d in deltas]), np.array([d[1] for d in deltas])
    median_theta, median_r = np.median(thetas), np.median(rs)

    angle_thresh = np.deg2rad(angle_deg)
    scale_thresh = scale_tol

    filtered = []
    for i, (m, (theta, r)) in enumerate(zip(good, deltas)):
        angle_diff = abs((theta - median_theta + np.pi) % (2*np.pi) - np.pi)
        scale_diff = abs(r - median_r) / median_r if median_r > 0 else float('inf')
        if angle_diff <= angle_thresh and scale_diff <= scale_thresh:
            filtered.append(m)

    if not filtered:
        filtered = good  # 兜底

    cv2.imwrite(os.path.join(output_dir, 'filtered_r-good-matches.png'),
                cv2.drawMatches(ref_gray, kp1, gen_gray, kp2, filtered, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

    match_ratio = len(filtered) / (len(des1) if des1 is not None else 1)
    return good, filtered, match_ratio
