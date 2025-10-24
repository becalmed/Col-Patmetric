import os
import ast
import re
import statistics

RE_MATCH_RATIO = re.compile(r"match_ratio:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
RE_PALETTE_DE  = re.compile(r"palette_deltaE'\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", re.DOTALL)
RE_MEAN_DE     = re.compile(r"mean_color_deltaE'\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", re.DOTALL)

def parse_result_file(path):
    """从 result.txt 提取 match_ratio、palette_deltaE、mean_color_deltaE"""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    mr  = RE_MATCH_RATIO.search(txt)
    pde = RE_PALETTE_DE.search(txt)
    mde = RE_MEAN_DE.search(txt)
    match_ratio = float(mr.group(1)) if mr else None
    palette_deltaE = float(pde.group(1)) if pde else None
    mean_color_deltaE = float(mde.group(1)) if mde else None
    return match_ratio, palette_deltaE, mean_color_deltaE


def collect_results(root_dir="metric_result"):
    match_ratios = []
    s_values = []

    for subdir, _, files in os.walk(root_dir):
        if "result.txt" in files:
            file_path = os.path.join(subdir, "result.txt")
            match_ratio, palette_deltaE, mean_color_deltaE = parse_result_file(file_path)
            if match_ratio is not None and palette_deltaE is not None and mean_color_deltaE is not None:
                # 保留 4 位小数
                Emax=20
                match_ratios.append(round(match_ratio, 4))
                spal=1-min(1,palette_deltaE/Emax)
                smean=1-min(1,mean_color_deltaE/Emax)
                b1=0.7
                b2=0.3
                s_values.append(round(b1 * spal + b2 * smean, 4))


    if match_ratios and s_values:
        P = round(statistics.mean(match_ratios), 4)
        Q = round(statistics.mean(s_values), 4)
        total = round((P + Q) / 2, 4)
        return P, Q, total
    else:
        return None, None, None


if __name__ == "__main__":
    P, Q, total = collect_results("/home/fangjingwu/data/dataset/test_dataset/upper/ootd_metric_res_cie")
    if P is None:
        print("没有找到有效的 result.txt 文件或内容缺失。")
    else:
        print(f"Spattern: {P}")
        print(f"Scolor: {Q}")
        print(f"Sfinal = {total}")
