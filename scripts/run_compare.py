import os
import sys

# 添加metric目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from comparator.comparator import PatchColorComparatorBase


if __name__ == "__main__":
    # ref_img = "/home/fangjingwu/data/dataset/test_dataset/upper/paired_image/100504.jpg"
    # gen_img = "/home/fangjingwu/data/dataset/test_dataset/upper/ours_test_res/100504.jpg"

    ref_img = "/home/fangjingwu/data/general_metrics/vitonmodel/01416_00.jpg"
    gen_img = "/home/fangjingwu/data/general_metrics/ours_test_res/01416_00.jpg"
    out_dir = "./vis_demo"

    comp = PatchColorComparatorBase(
        patch_size=32,
        server_addr="172.16.2.47:8080",
        corner_config="adaptive",
        region="upper"                  
    )

    os.makedirs(out_dir, exist_ok=True)
    result = comp.compare(ref_img, gen_img, output_dir=out_dir, clear_output=True)  
    print("\n💾 结果保存目录:", out_dir)
    #print(result)

