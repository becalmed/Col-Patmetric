import os
import cv2
import numpy as np
from PIL import Image

from .segmentation import auto_crop_interest_area_separate, resize_to_larger
from .features import detect_kaze
from .matching import flann_match_and_filter
from .warp import build_tps_with_constraints, map_points
from .visualization import create_patch_comparison_visualization

class PatchColorComparatorBase:
    def __init__(self, patch_size=32, server_addr='172.16.2.47:8080',
                 corner_config='adaptive', region='upper'):
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.server_addr = server_addr
        self.corner_config = corner_config
        self.region = region  # 默认 upper

    # ==== 小工具 ====
    def _clear_output_dir(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            return
        # 默认清空
        for fn in os.listdir(output_dir):
            fp = os.path.join(output_dir, fn)
            if os.path.isfile(fp) or os.path.islink(fp):
                try: os.unlink(fp)
                except: pass
            elif os.path.isdir(fp):
                import shutil; shutil.rmtree(fp, ignore_errors=True)

    def mask_grid_sample(self, mask_gray, step=10):
        h, w = mask_gray.shape
        return np.array([(x, y) for y in range(0, h, step)
                         for x in range(0, w, step) if mask_gray[y, x] > 0])

    # 为兼容旧用法暴露 detect_kaze（内部走 features.detect_kaze）
    def detect_kaze(self, gray, mask=None, save_path=None, strategy_name=None):
        return detect_kaze(gray, mask=mask, save_path=save_path, strategy_name=strategy_name, filter_boundary=True)

    # ==== 主流程 ====
    def compare(self, ref_path, gen_path, output_dir='.', clear_output=True):
        if clear_output:
            self._clear_output_dir(output_dir)

        result = {}

        # 读图
        ref_color = cv2.imread(ref_path); gen_color = cv2.imread(gen_path)
        if ref_color is None or gen_color is None:
            raise FileNotFoundError("输入图像不存在或读取失败")
        ref_gray_full = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        gen_gray_full = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)

        # 分别裁剪 ROI（默认 upper）
        (bbox_ref, bbox_gen), ref_color_crop, gen_color_crop, ref_mask_crop, gen_mask_crop = \
            auto_crop_interest_area_separate(ref_color, gen_color, self.server_addr, self.region)
        (_, _), ref_gray_crop, gen_gray_crop, ref_mask_gray_crop, gen_mask_gray_crop = \
            auto_crop_interest_area_separate(ref_gray_full, gen_gray_full, self.server_addr, self.region)

        # 转灰度
        if ref_gray_crop.ndim == 3: ref_gray_crop = cv2.cvtColor(ref_gray_crop, cv2.COLOR_BGR2GRAY)
        if gen_gray_crop.ndim == 3: gen_gray_crop = cv2.cvtColor(gen_gray_crop, cv2.COLOR_BGR2GRAY)

        # 统一大小
        ref_color_crop, gen_color_crop, ref_gray_crop, gen_gray_crop, ref_mask_gray_crop, gen_mask_gray_crop = \
            resize_to_larger(ref_color_crop, gen_color_crop, ref_gray_crop, gen_gray_crop, ref_mask_gray_crop, gen_mask_gray_crop)

        # 保存裁剪图
        Image.fromarray(cv2.cvtColor(ref_color_crop, cv2.COLOR_BGR2RGB)).save(os.path.join(output_dir, 'reference_cropped.png'))
        Image.fromarray(cv2.cvtColor(gen_color_crop, cv2.COLOR_BGR2RGB)).save(os.path.join(output_dir, 'generated_cropped.png'))
        cv2.imwrite(os.path.join(output_dir, 'reference_cropped_gray.png'), ref_gray_crop)
        cv2.imwrite(os.path.join(output_dir, 'generated_cropped_gray.png'), gen_gray_crop)

        # ---- KAZE + 策略一致性 ----
        print("在参考图像上进行自适应KAZE检测以确定最佳策略...")
        kp1, des1, ref_info = detect_kaze(
            ref_gray_crop, mask=ref_mask_gray_crop,
            save_path=os.path.join(output_dir, 'reference_keypoints.png'),
            strategy_name="dense",  # 自适应选择
            target_range=(300, 500),
            max_steps=8,
            save_each_step_dir=os.path.join(output_dir, "ref_kaze_steps")
        )
        if "fallback" in ref_info and ref_info["fallback"]==["fallback_return_as_is"]:
            print(f"点数未到达下限，直接输出,策略: {ref_info['config_name']}")
            result.update({
            'total_keypoints': len(des1) if des1 is not None else 0,
            'good_matches': 0,
            'filtered_matches': 0,
            'match_ratio': 0,
            'mean_diff_r': 0,
            'mean_diff_g': 0,
            'mean_diff_b': 0,
            'mean_euclid_diff': 0
        })
            return result
        print(f"✅ 参考图像策略: {ref_info['config_name']} | 特征点: {ref_info['keypoint_count']}")
        kp2, des2, gen_info = detect_kaze(
            gen_gray_crop, mask=gen_mask_gray_crop,
            target_range=(300, 500),
            save_path=os.path.join(output_dir, 'generated_keypoints.png'),
            strategy_name=ref_info['config_name']  # 保持策略一致
        )
        print(f"🔄 生成图像沿用策略: {gen_info['config_name']} | 特征点: {gen_info['keypoint_count']}")
        result.update({
            'ref_strategy_info': ref_info,
            'gen_strategy_info': gen_info,
            'strategy_consistency': True
        })

        # ---- 匹配 + 过滤 ----
        good, filtered, match_ratio = flann_match_and_filter(
            kp1, des1, kp2, des2, ref_gray_crop, gen_gray_crop, output_dir,
            ratio_thresh=0.75, angle_deg=5.0, scale_tol=0.15
        )

        # 取过滤后的匹配坐标
        filtered_src_pts = np.array([kp1[m.queryIdx].pt for m in filtered], dtype=np.float32) if filtered else np.empty((0, 2))
        filtered_dst_pts = np.array([kp2[m.trainIdx].pt for m in filtered], dtype=np.float32) if filtered else np.empty((0, 2))

        # ---- TPS/RBF （含边界约束）----
        try:
            rbf_x, rbf_y, b1, b2 = build_tps_with_constraints(
                filtered_src_pts, filtered_dst_pts,
                ref_mask_gray_crop, gen_mask_gray_crop,
                corner_config=self.corner_config, method='corner_based', n_boundary=30, smooth=0.1
            )
            use_tps = True
        except Exception as e:
            print(f"TPS 构建失败，改用直接对应: {e}")
            rbf_x = rbf_y = None
            use_tps = False

        # ---- 采样 + 色差 ----
        points = self.mask_grid_sample(ref_gray_crop, step=self.patch_size)
        h_ref, w_ref = ref_color_crop.shape[:2]
        h_gen, w_gen = gen_color_crop.shape[:2]

        diffs_r, diffs_g, diffs_b, diffs_euclid = [], [], [], []
        ref_vis, gen_vis = ref_color_crop.copy(), gen_color_crop.copy()
        patch_pairs = []

        for i, (x1, y1) in enumerate(points):
            if use_tps:
                x2f, y2f = rbf_x(x1, y1), rbf_y(x1, y1)
            else:
                x2f, y2f = x1, y1

            x1, y1 = int(round(x1)), int(round(y1))
            x2, y2 = int(round(float(x2f))), int(round(float(y2f)))

            x1a, x1b = max(0, x1 - self.half), min(w_ref, x1 + self.half)
            y1a, y1b = max(0, y1 - self.half), min(h_ref, y1 + self.half)
            x2a, x2b = max(0, x2 - self.half), min(w_gen, x2 + self.half)
            y2a, y2b = max(0, y2 - self.half), min(h_gen, y2 + self.half)

            cv2.rectangle(ref_vis, (x1a, y1a), (x1b, y1b), (0, 0, 255), 2)
            cv2.rectangle(gen_vis, (x2a, y2a), (x2b, y2b), (255, 0, 0), 2)

            patch_ref = ref_color_crop[y1a:y1b, x1a:x1b].astype(np.float32)
            patch_gen = gen_color_crop[y2a:y2b, x2a:x2b].astype(np.float32)
            h, w = min(patch_ref.shape[0], patch_gen.shape[0]), min(patch_ref.shape[1], patch_gen.shape[1])
            if h == 0 or w == 0: continue
            patch_ref = patch_ref[:h, :w]; patch_gen = patch_gen[:h, :w]
            N = h * w

            # 通道差：先求和后做差，最后除以像素数
            diff_r = abs(patch_ref[:, :, 2].sum() - patch_gen[:, :, 2].sum()) / N
            diff_g = abs(patch_ref[:, :, 1].sum() - patch_gen[:, :, 1].sum()) / N
            diff_b = abs(patch_ref[:, :, 0].sum() - patch_gen[:, :, 0].sum()) / N
            ref_eu = np.sqrt((patch_ref ** 2).sum(axis=2))
            gen_eu = np.sqrt((patch_gen ** 2).sum(axis=2))
            diff_euclid = abs(ref_eu.sum() - gen_eu.sum()) / N

            diffs_r.append(diff_r); diffs_g.append(diff_g); diffs_b.append(diff_b); diffs_euclid.append(diff_euclid)
            patch_pairs.append({
                'index': i,
                'ref_center': (x1, y1),
                'gen_center': (x2, y2),
                'ref_bbox': (x1a, y1a, x1b, y1b),
                'gen_bbox': (x2a, y2a, x2b, y2b),
                'ref_patch': patch_ref.copy(),
                'gen_patch': patch_gen.copy(),
                'diff_r': diff_r, 'diff_g': diff_g, 'diff_b': diff_b, 'diff_euclid': diff_euclid
            })

        # 可视化
        cv2.imwrite(os.path.join(output_dir, 'ref_patch_vis.png'), ref_vis)
        cv2.imwrite(os.path.join(output_dir, 'gen_patch_vis.png'), gen_vis)
        cv2.imwrite(os.path.join(output_dir, 'joint_patch_lines.png'), np.concatenate([ref_vis, gen_vis], axis=1))
        create_patch_comparison_visualization(patch_pairs, output_dir)

        # 汇总
        result.update({
            'total_keypoints': len(des1) if des1 is not None else 0,
            'good_matches': len(good),
            'filtered_matches': len(filtered),
            'match_ratio': match_ratio,
            'mean_diff_r': np.mean(diffs_r) if diffs_r else None,
            'mean_diff_g': np.mean(diffs_g) if diffs_g else None,
            'mean_diff_b': np.mean(diffs_b) if diffs_b else None,
            'mean_euclid_diff': np.mean(diffs_euclid) if diffs_euclid else None
        })

        print("\n====== 完整对比结果 ======")
        print(f"总特征点数: {result['total_keypoints']}")
        print(f"原始good matches: {result['good_matches']}")
        print(f"过滤后matches: {result['filtered_matches']}")
        print(f"匹配比例: {result['match_ratio']:.2%}")
        print("各通道平均色差 (越低越相似):")
        print(f"  R 平均差: {result['mean_diff_r']:.2f}")
        print(f"  G 平均差: {result['mean_diff_g']:.2f}")
        print(f"  B 平均差: {result['mean_diff_b']:.2f}")
        print(f"  Euclidean 色差: {result['mean_euclid_diff']:.2f}")

        return result
