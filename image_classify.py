import os
import cv2
import numpy as np
from PIL import Image
import shutil
from typing import Optional, Tuple

from .comparator.segmentation import auto_crop_interest_area_separate, resize_to_larger, crop_interest_area_separate
from .comparator.features import detect_kaze

def kaze(self, ref_path):
    # 读图
    ref_color = cv2.imread(ref_path)
    if ref_color is None:
        raise FileNotFoundError("输入图像不存在或读取失败")
    ref_gray_full = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

    # 分别裁剪 ROI（默认 upper）
    (bbox_ref, bbox_gen), ref_color_crop, ref_color_crop, ref_mask_crop, ref_mask_crop = \
        auto_crop_interest_area_separate(ref_color, ref_color, self.server_addr, self.region)
    (_, _), ref_gray_crop, ref_gray_crop, ref_mask_gray_crop, ref_mask_gray_crop = \
        crop_interest_area_separate(ref_gray_full, ref_gray_full, self.server_addr, self.region, ref_color, ref_color)

    # 转灰度
    if ref_gray_crop.ndim == 3: ref_gray_crop = cv2.cvtColor(ref_gray_crop, cv2.COLOR_BGR2GRAY)

    # 统一大小
    ref_color_crop, gen_color_crop, ref_gray_crop, gen_gray_crop, ref_mask_gray_crop, gen_mask_gray_crop = \
        resize_to_larger(ref_color_crop, gen_color_crop, ref_gray_crop, gen_gray_crop, ref_mask_gray_crop, gen_mask_gray_crop)
    
    # KAZE 检测
    kp1, des1, ref_info = detect_kaze(
        ref_gray_crop, mask=ref_mask_gray_crop,
        max_steps=8
    )

    return kp1

REF_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def categorize(result: float) -> str:
    if result < 300:
        return "easy"
    elif result > 1400:
        return "hard"
    else:
        return "medium"

def ensure_output_dirs(base: str) -> dict:
    dirs = {
        "easy_img":  os.path.join(base, "paired_image", "easy"),
        "med_img":   os.path.join(base, "paired_image", "medium"),
        "hard_img":  os.path.join(base, "paired_image", "hard"),
        "easy_cloth": os.path.join(base, "paired_cloth", "easycloth"),
        "med_cloth":  os.path.join(base, "paired_cloth", "mediumcloth"),
        "hard_cloth": os.path.join(base, "paired_cloth", "hardcloth"),
        "unpaired":   os.path.join(base, "unpaired"),
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs

def safe_copy(src: str, dst_dir: str) -> bool:
    try:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
        return True
    except Exception as e:
        print(f"[拷贝失败] {src} -> {dst_dir}: {e}")
        return False

def safe_move(src: str, dst_dir: str) -> bool:
    try:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, os.path.join(dst_dir, os.path.basename(src)))
        return True
    except Exception as e:
        print(f"[移动失败] {src} -> {dst_dir}: {e}")
        return False

def to_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def main():
    ref_dir   = "/path/to/ref_dir"
    cloth_dir = "/path/to/cloth_dir"
    outputdir = "/path/to/save"

    dirs = ensure_output_dirs(outputdir)

    # 统计
    stat = {"easy": 0, "medium": 0, "hard": 0, "unpaired": 0, "failed": 0}
    missing_cloth = []
    failed_files  = []

    ref_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(REF_EXTS)]
    ref_files.sort()

    for fname in ref_files:
        ref_path = os.path.join(ref_dir, fname)
        cloth_src = os.path.join(cloth_dir, fname)

        # 先跑 kaze，得到分类门槛
        try:
            result_raw = kaze(ref_path, output_dir=outputdir, clear_output=True)
            result_val = to_float(result_raw)
            if result_val is None:
                raise ValueError(f"kaze 返回值无效: {result_raw}")
            print(f"[成功] {fname} -> result={result_val:.4f}")
        except Exception as e:
            print(f"[失败] {fname}: {e}")
            stat["failed"] += 1
            failed_files.append(fname)
            continue

        bucket = categorize(result_val)

        # 如果 cloth 缺失：把 ref 移动到 unpaired（不再放 paired_image）
        if not os.path.isfile(cloth_src):
            print(f"[提示] 未找到同名 cloth 文件: {cloth_src}，移动参考图到 unpaired/")
            moved = safe_move(ref_path, dirs["unpaired"])
            if moved:
                stat["unpaired"] += 1
                missing_cloth.append(fname)
            # 如果移动失败，ref 仍留在原目录，继续下一个文件
            continue

        # cloth 存在：按 bucket 分别放到 paired_image 与 paired_cloth
        if bucket == "easy":
            img_dir, cloth_dir_out = dirs["easy_img"], dirs["easy_cloth"]
        elif bucket == "hard":
            img_dir, cloth_dir_out = dirs["hard_img"], dirs["hard_cloth"]
        else:
            img_dir, cloth_dir_out = dirs["med_img"], dirs["med_cloth"]

        # 拷贝参考图与 cloth
        ok_img = safe_copy(ref_path, img_dir)
        ok_cloth = safe_copy(cloth_src, cloth_dir_out)

        if ok_img:
            stat[bucket] += 1

        if not ok_cloth:
            # cloth 拷贝失败则尝试把 ref 移到 unpaired
            print(f"[警告] cloth 拷贝失败，回退：移动参考图到 unpaired/")
            safe_move(os.path.join(img_dir, fname), dirs["unpaired"])
            stat[bucket] -= 1
            stat["unpaired"] += 1

    # 汇总
    print("\n=== 处理完成 ===")
    print(f"paired_image easy   : {stat['easy']}")
    print(f"paired_image medium : {stat['medium']}")
    print(f"paired_image hard   : {stat['hard']}")
    print(f"unpaired            : {stat['unpaired']}")
    print(f"失败                : {stat['failed']}")

    if missing_cloth:
        print("\n以下文件缺少同名 cloth，已移到 unpaired/：")
        for f in missing_cloth:
            print("  -", f)

    if failed_files:
        print("\n以下文件处理失败（kaze 异常或返回值无效）：")
        for f in failed_files:
            print("  -", f)

if __name__ == "__main__":
    main()