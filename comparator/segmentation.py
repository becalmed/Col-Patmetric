import cv2
import numpy as np
from PIL import Image
from .vton_human_parsing import run_human_tasks_grpc, pil_b64

UPPER_LABELS = (23, 20, 14)  # top / outerwear / one-piece
LOWER_LABELS = (13, 12)      # pants / skirt

def _labels_to_mask(parse_arr, labels):
    m = np.zeros_like(parse_arr, dtype=np.uint8)
    for lb in labels:
        m |= (parse_arr == lb).astype(np.uint8)
    return (m * 255).astype(np.uint8)

def get_interest_area(img_bgr, server_addr: str, region: str = 'upper'):
    """返回 (bbox, PIL灰度mask)。默认 upper=23,20,14。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    mask_img, parse_img = run_human_tasks_grpc(server_addr, pil_b64(pil_img), 'all')
    if mask_img is None or parse_img is None:
        raise ValueError("无法获取人体分割结果，请检查输入图像或服务器连接。")

    parse_arr = np.array(parse_img)
    # 轻度形态学以平滑掩码
    kernel = np.ones((5, 5), np.uint8)
    parse_arr = cv2.erode(parse_arr, kernel, 2)
    parse_arr = cv2.dilate(parse_arr, kernel, 2)
    parse_arr = cv2.erode(parse_arr, kernel, 2)
    parse_arr = cv2.dilate(parse_arr, kernel, 2)

    if region == 'upper':
        mask = _labels_to_mask(parse_arr, UPPER_LABELS)
    elif region == 'lower':
        mask = _labels_to_mask(parse_arr, LOWER_LABELS) 
    elif region == 'all':
        mask = ((parse_arr > 0).astype(np.uint8) * 255)
    elif region == 'dress':
        mask = _labels_to_mask(parse_arr, (14, 13, 12, 23, 20))
    else:
        mask = _labels_to_mask(parse_arr, UPPER_LABELS)

    pil_mask = Image.fromarray(mask, mode='L')
    bbox = pil_mask.getbbox()
    return (bbox, pil_mask) if bbox else (None, None)

def crop_by_bbox(img_bgr, bbox):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    crop = pil.crop(bbox)
    return cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)

def auto_crop_interest_area_separate(img1_bgr, img2_bgr, server_addr: str, region: str = 'upper'):
    """各自独立裁剪到自身bbox；返回：(bbox1,bbox2), img1_crop, img2_crop, mask1_crop, mask2_crop"""
    bbox1, mask1_pil = get_interest_area(img1_bgr, server_addr, region)
    bbox2, mask2_pil = get_interest_area(img2_bgr, server_addr, region)
    if not bbox1 or not bbox2:
        raise ValueError("无法获取感兴趣区域的边界框，请检查输入图像。")

    mask1 = (np.array(mask1_pil) > 0).astype(np.uint8)
    mask2 = (np.array(mask2_pil) > 0).astype(np.uint8)
    img1 = img1_bgr * mask1[..., None] if img1_bgr.ndim == 3 else img1_bgr * mask1
    img2 = img2_bgr * mask2[..., None] if img2_bgr.ndim == 3 else img2_bgr * mask2

    img1_crop = crop_by_bbox(img1, bbox1)
    img2_crop = crop_by_bbox(img2, bbox2)
    mask1_crop = crop_by_bbox((mask1 * 255).astype(np.uint8), bbox1)
    mask2_crop = crop_by_bbox((mask2 * 255).astype(np.uint8), bbox2)
    return (bbox1, bbox2), img1_crop, img2_crop, mask1_crop, mask2_crop

def crop_interest_area_separate(img1_gry, img2_gry, server_addr: str, region: str = 'upper', img1_bgr=None, img2_bgr=None):
    """依照bbox裁剪；返回：(bbox1,bbox2), img1_crop, img2_crop, mask1_crop, mask2_crop"""

    bbox1, mask1_pil = get_interest_area(img1_bgr, server_addr, region)
    bbox2, mask2_pil = get_interest_area(img2_bgr, server_addr, region)
    
    if not bbox1 or not bbox2:
        raise ValueError("无法获取感兴趣区域的边界框，请检查输入图像。")

    mask1 = (np.array(mask1_pil) > 0).astype(np.uint8)
    mask2 = (np.array(mask2_pil) > 0).astype(np.uint8)
    img1 = img1_gry * mask1[..., None] if img1_gry.ndim == 3 else img1_gry * mask1
    img2 = img2_gry * mask2[..., None] if img2_gry.ndim == 3 else img2_gry * mask2

    img1_crop = crop_by_bbox(img1, bbox1)
    img2_crop = crop_by_bbox(img2, bbox2)
    mask1_crop = crop_by_bbox((mask1 * 255).astype(np.uint8), bbox1)
    mask2_crop = crop_by_bbox((mask2 * 255).astype(np.uint8), bbox2)
    return (bbox1, bbox2), img1_crop, img2_crop, mask1_crop, mask2_crop

def resize_to_larger(ref_color, gen_color, ref_gray, gen_gray, ref_mask, gen_mask):
    """将较小 resize 到较大；返回六元组。"""
    rh, rw = ref_color.shape[:2]
    gh, gw = gen_color.shape[:2]
    r_area, g_area = rh * rw, gh * gw
    if r_area == g_area and rh == gh and rw == gw:
        return ref_color, gen_color, ref_gray, gen_gray, ref_mask, gen_mask

    if r_area >= g_area:
        target = (rw, rh)
        gen_color = cv2.resize(gen_color, target, cv2.INTER_LINEAR)
        gen_gray  = cv2.resize(gen_gray,  target, cv2.INTER_LINEAR)
        gen_mask  = cv2.resize(gen_mask,  target, cv2.INTER_NEAREST)
    else:
        target = (gw, gh)
        ref_color = cv2.resize(ref_color, target, cv2.INTER_LINEAR)
        ref_gray  = cv2.resize(ref_gray,  target, cv2.INTER_LINEAR)
        ref_mask  = cv2.resize(ref_mask,  target, cv2.INTER_NEAREST)
    return ref_color, gen_color, ref_gray, gen_gray, ref_mask, gen_mask
