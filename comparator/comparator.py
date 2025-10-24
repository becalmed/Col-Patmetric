import os
import cv2
import numpy as np
from PIL import Image

from .segmentation import auto_crop_interest_area_separate, resize_to_larger, crop_interest_area_separate
from .features import detect_kaze
from .matching import flann_match_and_filter
from .cie import robust_cloth_color_diff
from .visualization import create_patch_comparison_visualization
from .warp import build_tps_with_constraints, map_points, make_all_constraints, uniformized_warp_points, augment_with_ghost_anchors, uniformized_warp_points_v2
class PatchColorComparatorBase:
    def __init__(self, patch_size=32, server_addr='172.16.2.47:8080',
                 corner_config='adaptive', region='upper'):
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.server_addr = server_addr
        self.corner_config = corner_config
        self.region = region 

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
            crop_interest_area_separate(ref_gray_full, gen_gray_full, self.server_addr, self.region, ref_color, gen_color)

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
            target_range=(300, 800),
            max_steps=8,
            save_each_step_dir=os.path.join(output_dir, "ref_kaze_steps")
        )
        if "fallback" in ref_info and ref_info["fallback"]==["fallback_return_as_is"]:
            print(f"点数未到达下限，直接输出,策略: {ref_info['config_name']}, 特征点: {ref_info['keypoint_count']}")
        print(f"✅ 参考图像策略: {ref_info['config_name']} | 特征点: {ref_info['keypoint_count']}")
        kp2, des2, gen_info = detect_kaze(
            gen_gray_crop, mask=gen_mask_gray_crop,
            target_range=(300, 900),
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

        score = robust_cloth_color_diff(ref_color_crop, gen_color_crop,ref_mask_gray_crop,gen_mask_gray_crop, return_palette=True, palette_vis_path=os.path.join(output_dir, 'palette_vis.png'))



        
        # 汇总
        result.update({
            'total_keypoints': len(des1) if des1 is not None else 0,
            'good_matches': len(good),
            'filtered_matches': len(filtered),
            'match_ratio': match_ratio,
            'score' : score
        })

        print("\n====== 完整对比结果 ======")
        print(f"总特征点数: {result['total_keypoints']}")
        print(f"原始good matches: {result['good_matches']}")
        print(f"过滤后matches: {result['filtered_matches']}")
        print(f"匹配比例: {result['match_ratio']:.2%}")
        print("分数详情：")
        hist = result['score']['hist_wasserstein']
        print(f"  Wasserstein(L,a,b,mean): "
            f"{hist['L']:.2f}, {hist['a']:.2f}, {hist['b']:.2f}, mean={hist['mean']:.2f}")
        print(f"  Palette ΔE2000: {result['score']['palette_deltaE']:.2f}")
        print(f"  Mean Color ΔE2000: {result['score']['mean_color_deltaE']:.2f}")
        spatten=match_ratio
        spal=1-min(1,score['palette_deltaE']/20)
        smean=1-min(1,score['mean_color_deltaE']/20)
        b1=0.7  
        b2=0.3
        scol=b1 * spal + b2 * smean
        sfinal=(spatten+scol)/2
        print(f"Spattern: {spatten:.4f}")
        print(f"Scolor: {scol:.4f}")
        print(f"Sfinal = {sfinal:.4f}")
        

        return result
