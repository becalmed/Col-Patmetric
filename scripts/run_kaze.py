import os
import sys
import cv2
import numpy as np

# 让脚本能直接运行而找到 comparator 包
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from comparator.features import detect_kaze

if __name__ == "__main__":
    # ==== 配置部分 ====
    img_path = "/home/fangjingwu/data/dataset/test_dataset/upper/paired_image/100036.jpg"   # 输入图像
    mask_path = None                       # 可选：对应的mask（灰度图）
    output_dir = "./kaze_test_output"      # 输出目录
    strategy_name = None                   # None=自适应; 也可写 'fast'/'balanced'/'dense'
    filter_boundary = True                 # 是否做“贴边特征点过滤”

    os.makedirs(output_dir, exist_ok=True)

    # ==== 读入图像 ====
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"图像文件不存在: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # ==== KAZE 检测 ====
    kp, des, info = detect_kaze(
        gray,
        mask=mask,
        save_path=os.path.join(output_dir, "keypoints.png"),
        strategy_name=strategy_name,
        filter_boundary=filter_boundary
    )

    print(f"检测完成: 策略 = {info['config_name']}")
    print(f"特征点数量: {info['keypoint_count']}")
    if des is not None:
        print(f"描述子形状: {des.shape}")
    else:
        print("无描述子（可能是没有检测到特征点）")

    print(f"可视化结果已保存至: {os.path.join(output_dir, 'keypoints.png')}")
