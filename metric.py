import os
import cv2
import sys
from tqdm import tqdm
import json
import statistics

from comparator.comparator import PatchColorComparatorBase

def batch_compare_folders(ref_dir, gen_dir, out_dir, patch_size=16, region="upper"):
    comparator = PatchColorComparatorBase(patch_size=patch_size,
        server_addr="172.16.2.47:8080", 
        corner_config="adaptive",
        region=region)
    os.makedirs(out_dir, exist_ok=True)

    # 获取所有ref_dir中的图片文件名
    ref_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    results_all = {}
    match_ratios = []
    s_values = []
    Emax=20
    b1=0.7
    b2=0.3
    for fname in tqdm(ref_files, desc="批量比对中"):
        ref_path = os.path.join(ref_dir, fname)
        gen_path = os.path.join(gen_dir, fname)
        if not os.path.isfile(gen_path):
            print(f"文件 {fname} 在生成文件夹中未找到，跳过")
            continue

        # 单独创建每对图片的输出子目录
        file_base, _ = os.path.splitext(fname)
        cur_out_dir = os.path.join(out_dir, file_base)
        
        # 检查是否已经处理过该图片
        if os.path.exists(cur_out_dir) and os.path.exists(os.path.join(cur_out_dir, 'result.txt')):
            print(f"文件 {fname} 已处理，跳过")
            continue
        
        os.makedirs(cur_out_dir, exist_ok=True)

        try:
            result = comparator.compare(ref_path, gen_path, output_dir=cur_out_dir)
        except Exception as e:
            print(f"{fname} 处理失败: {e}")
            continue

        # 保存结果
        txt_path = os.path.join(cur_out_dir, 'result.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for k, v in result.items():
                f.write(f"{k}: {v}\n")
        match_ratios.append(round(result['match_ratio'], 4))
        spal=1-min(1,result['score']['palette_deltaE']/Emax)
        smean=1-min(1,result['score']['mean_color_deltaE']/Emax)
        s_values.append(round(b1 * spal + b2 * smean, 4))
    P = round(statistics.mean(match_ratios), 4)
    Q = round(statistics.mean(s_values), 4)
    total = round((P + Q) / 2, 4)
    print(f"Spattern: {P}")
    print(f"Scolor: {Q}")
    print(f"Sfinal = {total}")
    # 保存所有结果汇总
    with open(os.path.join(out_dir, 'summary_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
        f.write(f"\nSpattern: {P}\nScolor: {Q}\nSfinal: {total}\n")

if __name__ == "__main__":
    ref_folder = "/home/fangjingwu/data/dataset/vitonhdsample/ref"
    gen_folder = "/home/fangjingwu/data/dataset/vitonhdsample/noise/salt/10"
    out_folder = "/home/fangjingwu/data/dataset/vitonhdsample/noise/salt/10_metric"
    batch_compare_folders(ref_folder, gen_folder, out_folder, patch_size=32, region="upper")
