import os
import cv2
import numpy as np

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
