import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

from skimage.color import lab2rgb,deltaE_ciede2000 
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

def create_patch_comparison_visualization(patch_pairs, output_dir, max_patches_per_image=50):
    if not patch_pairs:
        print("没有有效的patch对进行可视化")
        return

    patch_pairs_sorted = sorted(patch_pairs, key=lambda x: x['diff_euclid'], reverse=True)
    diffs = [p['diff_euclid'] for p in patch_pairs_sorted]
    avg_diff, max_diff, min_diff = np.mean(diffs), np.max(diffs), np.min(diffs)

    num_batches = (len(patch_pairs_sorted) + max_patches_per_image - 1) // max_patches_per_image
    for b in range(num_batches):
        st, ed = b * max_patches_per_image, min((b + 1) * max_patches_per_image, len(patch_pairs_sorted))
        batch = patch_pairs_sorted[st:ed]
        _create_single(batch, output_dir, b, avg_diff, max_diff, min_diff)

    _create_summary(patch_pairs_sorted, output_dir)
    print(f"生成了 {num_batches} 个patch对比可视化文件和1个统计汇总文件")

def _create_single(patch_pairs, output_dir, batch_idx, avg_diff, max_diff, min_diff):
    n = len(patch_pairs)
    if n == 0: return
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    patch_display = 100
    spacing = 15
    header_h = 50
    title_h = 40
    img_w = cols * (patch_display * 2 + spacing * 3) + spacing
    img_h = title_h + rows * (patch_display + header_h + spacing * 2) + spacing

    canvas = np.ones((img_h, img_w, 3), np.uint8) * 255
    cv2.putText(canvas, f"Patch Comparison Batch {batch_idx+1} - {n} patches", (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    for i, p in enumerate(patch_pairs):
        r = i // cols; c = i % cols
        x0 = c * (patch_display * 2 + spacing * 3) + spacing
        y0 = title_h + r * (patch_display + header_h + spacing * 2) + spacing

        ref_patch = cv2.resize(p['ref_patch'].astype(np.uint8), (patch_display, patch_display))
        gen_patch = cv2.resize(p['gen_patch'].astype(np.uint8), (patch_display, patch_display))

        rx, ry = x0, y0 + header_h
        gx, gy = x0 + patch_display + spacing, y0 + header_h
        canvas[ry:ry+patch_display, rx:rx+patch_display] = ref_patch
        canvas[gy:gy+patch_display, gx:gx+patch_display] = gen_patch

        cv2.putText(canvas, f"#{p['index']}", (x0, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        pos = f"({p['ref_center'][0]},{p['ref_center'][1]}) -> ({p['gen_center'][0]},{p['gen_center'][1]})"
        cv2.putText(canvas, pos, (x0, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        cv2.putText(canvas, f"Diff: {p['diff_euclid']:.1f}", (x0, y0 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        cv2.putText(canvas, f"R:{p['diff_r']:.1f} G:{p['diff_g']:.1f} B:{p['diff_b']:.1f}",
                    (x0, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)

        level = min(255, int(p['diff_euclid'] * 10))
        col = (0, 255 - level, level)
        cv2.rectangle(canvas, (rx-2, ry-2), (rx+patch_display+2, ry+patch_display+2), col, 3)
        cv2.rectangle(canvas, (gx-2, gy-2), (gx+patch_display+2, gy+patch_display+2), col, 3)
        cv2.putText(canvas, "REF", (rx+5, ry+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(canvas, "GEN", (gx+5, gy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    fn = f'patch_comparison_batch_{batch_idx:02d}.png'
    cv2.imwrite(os.path.join(output_dir, fn), canvas)

def _create_summary(patch_pairs, output_dir):
    diffs = [p['diff_euclid'] for p in patch_pairs]
    h, w = 400, 600
    img = np.ones((h, w, 3), np.uint8) * 255
    cv2.putText(img, "Patch Color Difference Statistics", (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    stats = [
        f"Total Patches: {len(patch_pairs)}",
        f"Average Difference: {np.mean(diffs):.2f}",
        f"Max Difference: {np.max(diffs):.2f}",
        f"Min Difference: {np.min(diffs):.2f}",
        f"Std Deviation: {np.std(diffs):.2f}",
        f"Median: {np.median(diffs):.2f}"
    ]
    for i, t in enumerate(stats):
        cv2.putText(img, t, (50, 80 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

    y0, hh, ww, x0 = 250, 100, 500, 50
    cv2.rectangle(img, (x0, y0), (x0 + ww, y0 + hh), (200, 200, 200), 1)
    bins = 20
    hist, _ = np.histogram(diffs, bins=bins)
    m = max(1, hist.max())
    bw = ww // bins
    for i, cnt in enumerate(hist):
        bh = int((cnt / m) * hh)
        x1 = x0 + i * bw
        y1 = y0 + hh - bh
        cv2.rectangle(img, (x1, y1), (x1 + bw - 1, y0 + hh), (100, 150, 200), -1)

    cv2.imwrite(os.path.join(output_dir, 'patch_statistics_summary.png'), img)

def _visualize_palette_matching(details, save_path="palette_deltaE_vis.png",
                                title_prefix="Palette Matching (ΔE2000)"):
    if plt is None:
        return None  # 没有 matplotlib 就跳过可视化
    c1, w1 = details["c1"], details["w1"]
    c2, w2 = details["c2"], details["w2"]
    rids, cids = details["assign"]
    pair_de = details["pair_de"]
    pal_de = details["palette_deltaE"]

    def lab_to_rgb01(Lab):
        rgb = lab2rgb(Lab.reshape(1, 1, 3)).reshape(3,)
        return np.clip(rgb, 0, 1)

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y_top, y_bot, h = 0.75, 0.25, 0.18

    def blocks(weights):
        if len(weights) == 0:
            return []
        w = weights / (weights.sum() + 1e-12)
        x_left, x_right = 0.05, 0.95
        span = x_right - x_left
        widths = w * span
        x_starts = np.cumsum(np.concatenate([[x_left], widths[:-1]]))
        x_ends = x_starts + widths
        return list(zip(x_starts, x_ends))

    top_blocks = blocks(w1)
    bot_blocks = blocks(w2)

    for i in range(len(c1)):
        x0, x1 = top_blocks[i]
        rgb = lab_to_rgb01(c1[i])
        ax.add_patch(plt.Rectangle((x0, y_top - h/2), x1 - x0, h))
        ax.patches[-1].set_facecolor(rgb)
        ax.text((x0 + x1)/2, y_top + h/2 + 0.03, f"{w1[i]*100:.1f}%", ha="center", va="bottom", fontsize=9)

    for j in range(len(c2)):
        x0, x1 = bot_blocks[j]
        rgb = lab_to_rgb01(c2[j])
        ax.add_patch(plt.Rectangle((x0, y_bot - h/2), x1 - x0, h))
        ax.patches[-1].set_facecolor(rgb)
        ax.text((x0 + x1)/2, y_bot - h/2 - 0.05, f"{w2[j]*100:.1f}%", ha="center", va="top", fontsize=9)

    for k in range(len(rids)):
        i, j = int(rids[k]), int(cids[k])
        xt = (top_blocks[i][0] + top_blocks[i][1]) / 2
        xb = (bot_blocks[j][0] + bot_blocks[j][1]) / 2
        ax.plot([xt, xb], [y_top - h/2, y_bot + h/2])  # 默认样式
        ax.text((xt + xb)/2, 0.5, f"{pair_de[k]:.2f}", ha="center", va="center", fontsize=9)

    title = f"{title_prefix} | weighted mean ΔE={pal_de:.2f}" if pal_de is not None else f"{title_prefix}"
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def _visualize_palette_proportion(c1, w1, c2, w2, save_path="palette_prop_vis.png",
                                  title="Palette Matching (Proportion)"):
    import matplotlib.pyplot as plt
    from skimage.color import lab2rgb

    def lab_to_rgb01(Lab):
        rgb = lab2rgb(Lab.reshape(1,1,3)).reshape(3,)
        return np.clip(rgb, 0, 1)

    # 排序（使用和 proportion 匹配相同的轴）
    axis = _proj_axis_lab(np.concatenate([c1, c2], axis=0) if len(c1)+len(c2)>0 else np.zeros((1,3)))
    s1 = (c1 @ axis) if len(c1) else np.array([])
    s2 = (c2 @ axis) if len(c2) else np.array([])
    o1, o2 = np.argsort(s1), np.argsort(s2)
    c1, w1 = c1[o1], w1[o1] / (w1.sum() + 1e-12)
    c2, w2 = c2[o2], w2[o2] / (w2.sum() + 1e-12)

    # 累积区间
    x0, span = 0.05, 0.90
    edges1 = np.concatenate([[0], np.cumsum(w1)]) * span + x0
    edges2 = np.concatenate([[0], np.cumsum(w2)]) * span + x0

    # 画布
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y_top, y_bot, h = 0.75, 0.25, 0.18

    # 上条
    for i in range(len(c1)):
        ax.add_patch(plt.Rectangle((edges1[i], y_top - h/2), edges1[i+1]-edges1[i], h,
                                   facecolor=lab_to_rgb01(c1[i])))
    # 下条
    for j in range(len(c2)):
        ax.add_patch(plt.Rectangle((edges2[j], y_bot - h/2), edges2[j+1]-edges2[j], h,
                                   facecolor=lab_to_rgb01(c2[j])))

    # 按分位匹配画连线
    i, j = 0, 0
    left1, left2 = edges1[0], edges2[0]
    right1, right2 = edges1[1], edges2[1]
    while i < len(c1) and j < len(c2):
        # 区间交集
        x_left = max(left1, left2)
        x_right = min(right1, right2)
        if x_right > x_left:
            xm = (x_left + x_right)/2
            # ΔE
            de = float(deltaE_ciede2000(c1[i][None,:], c2[j][None,:])[0])
            # 连线
            ax.plot([xm, xm], [y_bot + h/2, y_top - h/2], color='k')
            ax.text(xm, 0.5, f"{de:.2f}", ha="center", va="center", fontsize=8)
        # 推进区间
        if right1 < right2:
            i += 1
            left1, right1 = edges1[i], edges1[i+1] if i+1 <= len(c1) else 1.0
        elif right2 < right1:
            j += 1
            left2, right2 = edges2[j], edges2[j+1] if j+1 <= len(c2) else 1.0
        else:
            i += 1; j += 1
            if i < len(c1): left1, right1 = edges1[i], edges1[i+1]
            if j < len(c2): left2, right2 = edges2[j], edges2[j+1]

    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

