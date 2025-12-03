#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import csv
import cv2
import math
import argparse
import statistics
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import lpips
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

from cleanfid import fid as CLEANFID
from comparator.comparator import PatchColorComparatorBase


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def is_image_file(name: str) -> bool:
    return Path(name).suffix.lower() in IMG_EXTS

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_aligned_pairs(ref_dir: Path, gen_dir: Path) -> List[Tuple[Path, Path, str]]:
    """以 ref 为基准，找 gen 中同名文件"""
    pairs = []
    for name in sorted(os.listdir(ref_dir)):
        if not is_image_file(name):
            continue
        rp = ref_dir / name
        gp = gen_dir / name
        if gp.is_file():
            pairs.append((rp, gp, name))
    return pairs

def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """以中心旋转；边界反射填充，避免黑边"""
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

def elastic_deform(img_bgr: np.ndarray, alpha: float=20.0, sigma: float=6.0, fill=(255,255,255)) -> np.ndarray:
    """
    Simard 弹性形变：随机位移 -> 高斯平滑 -> remap
    alpha: 像素位移强度；sigma: 平滑尺度
    """
    h, w = img_bgr.shape[:2]
    # 随机位移场（-1~1）
    dx = (np.random.rand(h, w).astype(np.float32) * 2 - 1)
    dy = (np.random.rand(h, w).astype(np.float32) * 2 - 1)
    # 高斯平滑并按 alpha 缩放到像素位移
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
    # 像素坐标映射
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = x + dx
    map_y = y + dy
    # remap，越界用白色填充
    return cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=fill)

def save_image(path: Path, img: np.ndarray) -> bool:
    ensure_dir(path.parent)
    return cv2.imwrite(str(path), img)

def load_image_rgb(path: Path) -> Optional[np.ndarray]:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def save_txt_like_result(out_txt: Path, result: Dict[str, Any]):
    """保存成 k: v 文本；score 为 dict 则 JSON 串"""
    ensure_dir(out_txt.parent)
    with open(out_txt, 'w', encoding='utf-8') as f:
        for k, v in result.items():
            if k == 'score' and isinstance(v, dict):
                f.write(f"{k}: {json.dumps(v, ensure_ascii=False)}\n")
            else:
                f.write(f"{k}: {v}\n")

def read_txt_like_result(txt_path: Path) -> Optional[Dict[str, Any]]:
    if not txt_path.is_file(): return None
    data: Dict[str, Any] = {}
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                k, v = line.split(':', 1)
                k, v = k.strip(), v.strip()
                if k == 'score' and v.startswith('{') and v.endswith('}'):
                    try:
                        data['score'] = json.loads(v)
                    except Exception:
                        data['score'] = {}
                else:
                    try:
                        if '.' in v or 'e' in v.lower():
                            data[k] = float(v)
                        else:
                            data[k] = int(v)
                    except Exception:
                        data[k] = v
        return data
    except Exception:
        return None

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def get_score_block(result: Dict[str, Any]) -> Dict[str, Any]:
    return result.get('score', result)

def compute_scolor(result_ref_gen: Dict[str, Any], Emax=20.0, b1=0.7, b2=0.3) -> float:
    """Scolor = b1*(1 - min(paletteΔE/Emax,1)) + b2*(1 - min(meanΔE/Emax,1))"""
    sb = get_score_block(result_ref_gen)
    pal = safe_float(sb.get('palette_deltaE', 0.0), 0.0)
    mean_de = safe_float(sb.get('mean_color_deltaE', 0.0), 0.0)
    if Emax <= 0:
        return 0.0
    spal = 1.0 - min(1.0, pal / Emax)
    smean = 1.0 - min(1.0, mean_de / Emax)
    return round(b1 * spal + b2 * smean, 4)

# ---------- 比对执行 ----------

def run_compare(comparator: PatchColorComparatorBase, a_path: Path, b_path: Path, out_dir: Path, overwrite: bool) -> Optional[Dict[str, Any]]:
    """运行或复用 compare；写 result.txt"""
    ensure_dir(out_dir)
    txt_path = out_dir / 'result.txt'
    if txt_path.is_file() and not overwrite:
        res = read_txt_like_result(txt_path)
        if res is not None:
            return res
    try:
        res = comparator.compare(str(a_path), str(b_path), output_dir=str(out_dir))
        save_txt_like_result(txt_path, res)
        return res
    except Exception as e:
        print(f"[失败] compare({a_path.name}, {b_path.name}): {e}")
        return None

def compute_spattern(result_ref_gen: Dict[str, Any], result_ref_perturb: Dict[str, Any], eps: float) -> float:
    m_ref_gen = safe_float(result_ref_gen.get('match_ratio', 0.0), 0.0)
    m_ref_per = safe_float(result_ref_perturb.get('match_ratio', 0.0), 0.0)
    sp = m_ref_gen / max(m_ref_per, eps)
    sp = max(0.0, min(1.0, sp))
    return round(sp, 4)


FIX_W, FIX_H = 768, 1024  # 宽×高

def resize_to_fixed(img_rgb: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(img_rgb).resize((FIX_W, FIX_H), Image.BILINEAR))

def compute_folder_ssim_lpips_psnr(
    pairs: List[Tuple[Path, Path, str]],
    lpips_net: str = 'alex',
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    对齐同名文件，双方统一 resize 到 (768,1024) 后计算 SSIM/LPIPS/PSNR。
    返回均值/中位数以及每张图的记录。
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # metric_lpips = lpips.LPIPS(net=lpips_net).to(device).eval()
    metric_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval()

    ssim_list, lpips_list, psnr_list = [], [], []
    per_rows = []

    with torch.no_grad():
        for ref_path, gen_path, name in tqdm(pairs, desc="SSIM/LPIPS/PSNR (768x1024)"):
            ref = load_image_rgb(ref_path)
            gen = load_image_rgb(gen_path)
            if ref is None or gen is None:
                continue
            ref_r = resize_to_fixed(ref)
            gen_r = resize_to_fixed(gen)

            # SSIM（uint8, channel_axis=2）
            ssim_val = structural_similarity(ref_r, gen_r, data_range=255, channel_axis=2)

            # LPIPS（float32 [0,1]，NCHW）
            t1 = torch.tensor(ref_r).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            t2 = torch.tensor(gen_r).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            lpips_val = metric_lpips(t1, t2).item()

            # PSNR（uint8, data_range=255）
            psnr_val = peak_signal_noise_ratio(ref_r, gen_r, data_range=255)

            ssim_list.append(ssim_val)
            lpips_list.append(lpips_val)
            psnr_list.append(psnr_val)
            per_rows.append({
                'filename': name,
                'SSIM': ssim_val,
                'LPIPS': lpips_val,
                'PSNR': psnr_val
            })

    def mean0(x): return float(np.mean(x)) if x else float('nan')
    def med0(x):  return float(np.median(x)) if x else float('nan')

    return {
        'ssim_mean': mean0(ssim_list), 'ssim_median': med0(ssim_list),
        'lpips_mean': mean0(lpips_list), 'lpips_median': med0(lpips_list),
        'psnr_mean': mean0(psnr_list), 'psnr_median': med0(psnr_list),
        'per_image': per_rows,
        'resize_policy': f'force_fixed_{FIX_W}x{FIX_H}'
    }

def save_per_image_metrics_csv(rows: List[Dict[str, Any]], csv_path: Path):
    ensure_dir(csv_path.parent)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'SSIM', 'LPIPS', 'PSNR'])
        writer.writeheader()
        writer.writerows(rows)

def compute_fid_kid_cleanfid(gen_dir: Path, ref_dir: Path) -> Dict[str, Any]:
    """使用 cleanfid"""
    fid_val = CLEANFID.compute_fid(str(gen_dir), str(ref_dir))
    kid_val = CLEANFID.compute_kid(str(gen_dir), str(ref_dir)) * 1000.0  
    return {'fid': float(fid_val), 'kid': float(kid_val), 'lib': 'cleanfid'}

# ---------- 主流程 ----------

def parse_args():
    p = argparse.ArgumentParser(description="Compare with perturb (Spattern normalize) + folder metrics (FID/KID/SSIM/LPIPS/PSNR @768x1024)")
    p.add_argument('--ref', required=True, help='参考图目录（GT）')
    p.add_argument('--gen', required=True, help='生成图目录（与 ref 同名对齐）')
    p.add_argument('--out', required=True, help='输出目录（本脚本产物）')
    p.add_argument('--perturb_dir', required=True, help='扰动参考图保存目录')

    # comparator 参数
    p.add_argument('--patch_size', type=int, default=32)
    p.add_argument('--region', type=str, default='upper', choices=['upper','lower','dress','full','upper_lower'])
    p.add_argument('--server_addr', type=str, default='172.16.2.47:8080')
    p.add_argument('--corner_config', type=str, default='adaptive')

    # 扰动
    p.add_argument('--perturb', type=str, default='rotate', choices=['rotate','elastic'])
    p.add_argument('--angle', type=float, default=2.0)

    # 颜色分
    p.add_argument('--Emax', type=float, default=20.0)
    p.add_argument('--b1', type=float, default=0.7)
    p.add_argument('--b2', type=float, default=0.3)

    # 统计控制
    p.add_argument('--exclude_zero_match', action='store_true')
    p.add_argument('--eps', type=float, default=1e-6)
    p.add_argument('--overwrite', action='store_true')

    # 整夹指标控制
    p.add_argument('--compute_folder_metrics', action='store_true', help='开启后计算 FID/KID/SSIM/LPIPS/PSNR')
    p.add_argument('--lpips_net', type=str, default='alex', choices=['alex','vgg','squeeze'])
    p.add_argument('--lpips_device', type=str, default=None, help='cuda/cpu；默认自动')

    return p.parse_args()

def main():
    args = parse_args()
    ref_dir = Path(args.ref)
    gen_dir = Path(args.gen)
    out_dir = Path(args.out)
    per_dir = Path(args.perturb_dir)

    ensure_dir(out_dir)
    ensure_dir(per_dir)

    # 初始化 comparator
    comparator = PatchColorComparatorBase(
        patch_size=args.patch_size,
        server_addr=args.server_addr,
        corner_config=args.corner_config,
        region=args.region
    )

    # 逐样本对齐列表
    pairs = list_aligned_pairs(ref_dir, gen_dir)
    if not pairs:
        print("[警告] 对齐样本为空；请检查 --ref 与 --gen")
        return

    spattern_list, scolor_list, sfinal_list = [], [], []
    gen_f1score,disturb_f1score = [],[]
    ok_count, fail_count = 0, 0

    # 逐样本评测 + 扰动归一化
    for ref_path, gen_path, name in tqdm(pairs, desc="逐样本评测"):
        # 准备扰动参考图
        per_img_path = per_dir / name
        if args.overwrite or not per_img_path.is_file():
            bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[跳过] 无法读取参考图：{name}")
                fail_count += 1
                continue
            if args.perturb == 'rotate':
                bgr_per = rotate_image(bgr, args.angle)
            elif args.perturb == 'elastic':
                bgr_per = elastic_deform(bgr, alpha=30.0, sigma=7.0, fill=(255,255,255))
                bgr_per = cv2.GaussianBlur(bgr_per, (5, 5), sigmaX=2, sigmaY=2)
            else:
                bgr_per = bgr
            if not save_image(per_img_path, bgr_per):
                print(f"[警告] 扰动图保存失败：{per_img_path}")

        # 路径与输出目录
        file_base = Path(name).stem
        cur_dir = out_dir / 'per_sample' / file_base
        out_orig = cur_dir / 'original'
        out_pert = cur_dir / 'perturb'

        # (ref, gen)
        res_ref_gen = run_compare(comparator, ref_path, gen_path, out_orig, overwrite=args.overwrite)
        if res_ref_gen is None:
            fail_count += 1
            continue
        # (ref, ref_perturbed)
        res_ref_per = run_compare(comparator, ref_path, per_img_path, out_pert, overwrite=args.overwrite)
        if res_ref_per is None:
            fail_count += 1
            continue

        # 合成指标
        spattern = compute_spattern(res_ref_gen, res_ref_per, eps=args.eps)
        scolor = compute_scolor(res_ref_gen, Emax=args.Emax, b1=args.b1, b2=args.b2)
        sfinal = round((spattern + scolor) / 2.0, 4)

        recall=(res_ref_gen["filtered_matches"]/res_ref_gen["total_keypoints"])
        precision=(res_ref_gen["filtered_matches"]/res_ref_gen["gen_strategy_info"]["keypoint_count"])
        F1score=2*recall*precision/(recall+precision)

        recall2=(res_ref_per["filtered_matches"]/res_ref_per["total_keypoints"])
        precision2=(res_ref_per["filtered_matches"]/res_ref_per["gen_strategy_info"]["keypoint_count"])
        F1score2=2*recall2*precision2/(recall2+precision2)


        combined = {
            "filename": name,
            "perturb": {"type": args.perturb, "angle": float(args.angle), "path": str(per_img_path)},
            "ref_vs_gen": {"match_ratio": safe_float(res_ref_gen.get("match_ratio", 0.0), 0.0),
                           "score": res_ref_gen.get("score", {})},
            "ref_vs_perturb": {"match_ratio": safe_float(res_ref_per.get("match_ratio", 0.0), 0.0),
                               "score": res_ref_per.get("score", {})},
            "gen_F1score": F1score,
            "distrub_F1score": F1score2,
            "Spattern": spattern, "Scolor": scolor, "Sfinal": sfinal

        }
        ensure_dir(cur_dir)
        with open(cur_dir / "combined.json", "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        if not (args.exclude_zero_match and safe_float(res_ref_gen.get("match_ratio", 0.0), 0.0) == 0.0):
            spattern_list.append(spattern)
        scolor_list.append(scolor)
        sfinal_list.append(sfinal)
        gen_f1score.append(F1score)
        disturb_f1score.append(F1score2)
        ok_count += 1

    # 聚合逐样本统计
    def mean0(x): return round(statistics.mean(x), 6) if x else float('nan')
    def med0(x):  return round(statistics.median(x), 6) if x else float('nan')

    spattern_mean, spattern_median = mean0(spattern_list), med0(spattern_list)
    scolor_mean, scolor_median     = mean0(scolor_list),  med0(scolor_list)
    sfinal_mean, sfinal_median     = mean0(sfinal_list),  med0(sfinal_list)
    gen_f1score_mean, gen_f1score_median = mean0(gen_f1score), med0(gen_f1score)
    disturb_f1score_mean, disturb_f1score_median = mean0(disturb_f1score), med0(disturb_f1score)
    # 通用指标
    folder_metrics = {}
    if args.compute_folder_metrics:
        # FID/KID
        fk = compute_fid_kid_cleanfid(gen_dir, ref_dir)
        folder_metrics['fid'] = fk['fid']
        folder_metrics['kid'] = fk['kid']
        folder_metrics['lib'] = fk['lib']
        fm_dir = out_dir / 'folder_metrics'
        ensure_dir(fm_dir)
        with open(fm_dir / 'pair_fid_kid.json', 'w', encoding='utf-8') as f:
            json.dump(fk, f, indent=2, ensure_ascii=False)

        # SSIM/LPIPS/PSNR @ 768×1024
        ssp = compute_folder_ssim_lpips_psnr(pairs, lpips_net=args.lpips_net, device=args.lpips_device)
        with open(fm_dir / 'pair_ssim_lpips_psnr.json', 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in ssp.items() if k != 'per_image'}, f, indent=2, ensure_ascii=False)
        save_per_image_metrics_csv(ssp['per_image'], fm_dir / 'per_image_ssim_lpips_psnr.csv')
        folder_metrics.update({k: v for k, v in ssp.items() if k != 'per_image'})

    # 总结文件
    summary = {
        "samples_total": len(pairs),
        "samples_ok": ok_count,
        "samples_failed": fail_count,
        "exclude_zero_match": bool(args.exclude_zero_match),
        "spattern_mean": spattern_mean, "spattern_median": spattern_median,
        "scolor_mean": scolor_mean,     "scolor_median": scolor_median,
        "sfinal_mean": sfinal_mean,     "sfinal_median": sfinal_median,
        "gen_f1score_mean": gen_f1score_mean, "gen_f1score_median": gen_f1score_median,
        "disturb_f1score_mean": disturb_f1score_mean, "disturb_f1score_median": disturb_f1score_median,
        "params": {
            "patch_size": args.patch_size, "region": args.region,
            "server_addr": args.server_addr, "corner_config": args.corner_config,
            "perturb": args.perturb, "angle": float(args.angle),
            "Emax": float(args.Emax), "b1": float(args.b1), "b2": float(args.b2),
            "resize_policy": f'force_fixed_{FIX_W}x{FIX_H}',
            "compute_folder_metrics": bool(args.compute_folder_metrics),
            "lpips_net": args.lpips_net, "lpips_device": args.lpips_device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        }
    }
    summary.update(folder_metrics)
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n新指标")
    print(f"Spattern_mean: {spattern_mean} | median: {spattern_median}")
    print(f"Scolor_mean:   {scolor_mean}   | median: {scolor_median}")
    print(f"Sfinal_mean:   {sfinal_mean}   | median: {sfinal_median}")
    print(f"Gen F1score_mean: {gen_f1score_mean} | median: {gen_f1score_median}")
    print(f"Disturb F1score_mean: {disturb_f1score_mean} | median: {disturb_f1score_median}")
    if args.compute_folder_metrics:
        print("通用指标")
        print(f"FID:  {folder_metrics.get('fid'):.4f} | KID(*1e3): {folder_metrics.get('kid'):.4f}")
        print(f"SSIM_mean: {folder_metrics.get('ssim_mean'):.4f} | LPIPS_mean: {folder_metrics.get('lpips_mean'):.4f} | PSNR_mean: {folder_metrics.get('psnr_mean'):.4f}")
        print(f"Resize policy: {folder_metrics.get('resize_policy')}")

if __name__ == "__main__":
    main()
