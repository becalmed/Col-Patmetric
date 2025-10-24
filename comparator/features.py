# comparator/features.py
import cv2
import numpy as np

# ========== 策略注册表 ==========
# 说明：
# - rank 越小越严格（strict）/ 越大越宽松（loose）
# - kaze: OpenCV KAZE_create 参数
# - post: 额外后处理（边界距离、响应值分位截断），按需要可省略
STRATEGY_REGISTRY = [
    {
        "name": "ultra_strict",
        "rank": 0,
        "kaze": dict(extended=False, upright=False, threshold=1.5e-3, nOctaves=4, nOctaveLayers=2, diffusivity=2),
        "post": dict(boundary_dist=16, response_quantile=None)
    },
    {
        "name": "strict",
        "rank": 1,
        "kaze": dict(extended=False, upright=False, threshold=1.2e-3, nOctaves=4, nOctaveLayers=2, diffusivity=2),
        "post": dict(boundary_dist=14, response_quantile=None)
    },
    {
        "name": "fast",
        "rank": 2,
        "kaze": dict(extended=False, upright=False, threshold=8e-4, nOctaves=4, nOctaveLayers=3, diffusivity=2),
        "post": dict(boundary_dist=14, response_quantile=None)
    },
    {
        "name": "balanced",
        "rank": 3,
        "kaze": dict(extended=False, upright=False, threshold=4e-4, nOctaves=4, nOctaveLayers=4, diffusivity=2),
        "post": dict(boundary_dist=14, response_quantile=None)  
    },
    {
        "name": "dense",
        "rank": 4,
        "kaze": dict(extended=False, upright=False, threshold=1e-4, nOctaves=5, nOctaveLayers=5, diffusivity=2),
        "post": dict(boundary_dist=14, response_quantile=None)
    },
]

def get_strategy_names():
    return [s["name"] for s in sorted(STRATEGY_REGISTRY, key=lambda x: x["rank"])]


# —— 工具：保证 mask 是二值
def _ensure_binary_mask(mask):
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

# —— 工具：边界 + 响应值过滤
def _filter_keypoints(mask_bin, keypoints, descriptors, boundary_dist=8, response_q=None):
    if not keypoints:
        return [], None
    kept = list(range(len(keypoints)))

    if mask_bin is not None:
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary_mask = np.zeros_like(mask_bin)
        cv2.drawContours(boundary_mask, contours, -1, 255, thickness=2)
        dist_map = cv2.distanceTransform(255 - boundary_mask, cv2.DIST_L2, 5)

        kept2 = []
        for i in kept:
            x, y = int(round(keypoints[i].pt[0])), int(round(keypoints[i].pt[1]))
            if 0 <= x < dist_map.shape[1] and 0 <= y < dist_map.shape[0]:
                if dist_map[y, x] > boundary_dist and (mask_bin[y, x] > 0):
                    kept2.append(i)
        kept = kept2

    if response_q is not None and kept:
        resp = np.array([keypoints[i].response for i in kept], dtype=np.float32)
        thr = np.quantile(resp, response_q)
        kept = [i for i in kept if keypoints[i].response >= thr]

    if descriptors is not None and kept:
        des_out = np.stack([descriptors[i] for i in kept], axis=0)
    else:
        des_out = None
    kp_out = [keypoints[i] for i in kept]
    return kp_out, des_out

# —— 工具：按响应值 Top-K 截断（用于 n>high）
def _topk_by_response(kp, des, k):
    if not kp:
        return [], None
    idx = np.argsort([-p.response for p in kp])[:k]
    kp2 = [kp[i] for i in idx]
    des2 = (des[idx] if (des is not None and len(des) == len(kp)) else None)
    return kp2, des2

# —— 跑一次指定策略
def _run_kaze_once(gray, mask, *, strategy, save_path=None):
    mask_bin = _ensure_binary_mask(mask)
    k = cv2.KAZE_create(**strategy["kaze"])
    kp, des = k.detectAndCompute(gray, mask=mask_bin)

    post = strategy.get("post", {}) or {}
    bd = int(post.get("boundary_dist", 8))
    rq = post.get("response_quantile", None)
    kp, des = _filter_keypoints(mask_bin, kp, des, boundary_dist=bd, response_q=rq)

    if save_path:
        vis = cv2.drawKeypoints(gray, kp, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(save_path, vis)

    info = {
        "config_name": strategy["name"],
        "rank": strategy["rank"],
        "keypoint_count": len(kp),
        "kaze_params": strategy["kaze"],
        "post": post
    }
    return kp, des, info, k, mask_bin

def detect_kaze(
    gray, mask=None, save_path=None,
    strategy_name: str | None = None,
    target_range=(300, 1500),
    max_steps=6,
    save_each_step_dir: str | None = None,
    enforce_range: bool = True, 
):
    """
    返回: (kp, des, info)
    info['tried'] 记录每一步；info['fallback'] 记录兜底动作
    """
    reg_sorted = sorted(STRATEGY_REGISTRY, key=lambda s: s["rank"])
    name2idx = {s["name"]: i for i, s in enumerate(reg_sorted)}
    if strategy_name:
        strategy = next((s for s in reg_sorted if s["name"] == strategy_name), None)
        if strategy is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        kp, des, info, kaze_obj, mask_bin = _run_kaze_once(gray, mask, strategy=strategy, save_path=save_path)
        # 直接进兜底（如果需要）
        if enforce_range:
            kp, des, info = _enforce_range(gray, mask_bin, kp, des, info, kaze_obj, target_range, save_path)
        return kp, des, info
    # 自适应：从 strict 出发
    if "strict" not in name2idx:
        raise ValueError("Strategy 'strict' must exist for auto mode.")
    idx = name2idx["strict"]
    tried = []
    kp = des = kaze_obj = mask_bin = None
    info = {}
    low, high = target_range
    
    for step in range(max_steps):
        strategy = reg_sorted[idx]
        sp = (f"{save_each_step_dir}/step_{step:02d}_{strategy['name']}.png" if save_each_step_dir else None)
        kp, des, info, kaze_obj, mask_bin = _run_kaze_once(gray, mask, strategy=strategy, save_path=sp)
        tried.append({"name": strategy["name"], "rank": strategy["rank"], "count": len(kp)})

        n = len(kp)
        if low <= n and n <= high:
            info["tried"] = tried
            # 命中即返；也可额外保存最终图
            if save_path and sp and save_path != sp:
                vis = cv2.drawKeypoints(gray, kp, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
                cv2.imwrite(save_path, vis)
            return kp, des, info

        if n > high and idx > 0:
            idx -= 1     # 更严格
        elif n < low and idx < len(reg_sorted) - 1:
            idx += 1     # 更宽松
        else:
            break        # 到边界了，跳出渐进

    # 渐进走到头了，还不在范围：兜底
    info["tried"] = tried
    if enforce_range:
        kp, des, info = _enforce_range(gray, mask_bin, kp, des, info, kaze_obj, target_range, save_path)
    return kp, des, info

# —— 兜底核心：软性收敛 + 硬性补齐 + Top‑K 截断
def _enforce_range(gray, mask_bin, kp, des, info, kaze_obj, target_range, save_path):
    low, high = target_range
    n = len(kp)
    info.setdefault("fallback", [])

    # A) 点太多：直接按响应值 Top‑K 到 high
    if n > high:
        kp2, des2 = _topk_by_response(kp, des, high)
        info["fallback"].append(f"trim_topk_to_{high}")
        _maybe_save(gray, kp2, save_path)
        info["keypoint_count"] = len(kp2)
        return kp2, des2, info

    # B4: 兜底失败（极端情况）——返回已有点
    info["fallback"].append("fallback_return_as_is")
    info["keypoint_count"] = len(kp)
    return kp, des, info

    # 已经在范围内
    return kp, des, info

def _maybe_save(gray, kp, save_path):
    if save_path:
        vis = cv2.drawKeypoints(gray, kp, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(save_path, vis)