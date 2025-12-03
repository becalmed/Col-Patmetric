from __future__ import print_function

import grpc
from pathlib import Path
import sys
import cv2
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import json
from google.protobuf.json_format import MessageToJson, Parse
from graphite_grpc.log.my_log import MyLog
from graphite_grpc.rpc.proto import common_pb2
from graphite_grpc.rpc.proto.common import handler_common_pb2_grpc, handler_common_pb2
import re
import base64
from io import BytesIO
from PIL import Image, ImageOps
import os, json, hashlib, tempfile
from functools import lru_cache

r_b64_prefix = r'^data:image/.+;base64,'

SHEIN_CLASSES = {
    "background": 0,
    "left arm": 1,
    "right arm": 2,
    "face": 3,
    "left foot": 4,
    "right foot": 5,
    "hair": 6,
    "left hand": 7,
    "right hand": 8,
    "left leg": 9,
    "right leg": 10,
    "torso": 11,
    "belt": 12,
    "bottoms": 13,
    "one-piece": 14,
    "left socks": 15,
    "right socks": 16,
    "left footwear": 17,
    "right footwear": 18,
    "headwear": 19,
    "outerwear": 20,
    "eyewear": 21,
    "neckwear": 22,
    "top": 23,
    "handholders": 24,
}

def b64_pil(base64Data):
    base64Data = re.sub(r_b64_prefix, '', base64Data)
    imgData = base64.b64decode(base64Data)
    image = Image.open(BytesIO(imgData))
    return image

def pil_b64(pil_img, fmt='jpeg'):
    if pil_img is None:
        return None
    output_buffer = BytesIO()

    pil_img.save(output_buffer, format=fmt, quality = 95)
    byte_data = output_buffer.getvalue()

    return base64.b64encode(byte_data).decode('utf-8')

def run_human_tasks_grpc(
    address,
    input_image_b64,
    mode,
    *,
    cache_dir: str = None,
    filename: str = None,     
    timeout: int = 60
):

    if mode not in ['seg', 'parse', 'all']:
        return None, None

    if filename is None:
        raise ValueError("必须传入 filename")

    mask_path = None
    parse_path = None

    if cache_dir:
        cache_subdir = os.path.join(cache_dir, filename.split('.')[0])
        os.makedirs(cache_subdir, exist_ok=True)

        mask_path = os.path.join(cache_subdir, "mask.png")
        parse_path = os.path.join(cache_subdir, "parse.png")

        # 尝试命中缓存
        mask_img, parse_img = None, None

        if mode in ('seg', 'all') and os.path.exists(mask_path):
            mask_img = Image.open(mask_path).copy()

        if mode in ('parse', 'all') and os.path.exists(parse_path):
            parse_img = Image.open(parse_path).copy()

        # 根据 mode 判断是否完全命中
        if mode == 'seg' and mask_img is not None:
            return mask_img, None
        if mode == 'parse' and parse_img is not None:
            return None, parse_img
        if mode == 'all' and (mask_img is not None and parse_img is not None):
            return mask_img, parse_img
        print(f"缓存未命中，调用服务获取 {mode} 结果...")
    channel = grpc.insecure_channel(
        address,
        options=[
            ('grpc.max_send_message_length', 268435456),
            ('grpc.max_receive_message_length', 268435456),
        ]
    )
    handler_service = handler_common_pb2_grpc.HandlerServiceStub(channel)

    request_json = json.dumps({
        "method": mode,
        "data": {"image": input_image_b64, "read_from_url": False}
    })
    params = common_pb2.Value(string_value=request_json)
    list_value = common_pb2.ListValue(elements=[params])
    request = handler_common_pb2.Request(serviceName='vton_human_tasks', arguments=list_value)

    response = handler_service.handle(request, timeout=timeout)

    # 解析响应
    try:
        params_out = json.loads(response.data.string_value)
        if params_out.get("code") != 1000:
            return None, None

        data = params_out.get("data", {})

        mask_img = None
        parse_img = None

        if mode in ["seg", "all"]:
            masks = data.get("image_masks") or []
            if masks:
                mask_img = b64_pil(masks[0])

        if mode in ["parse", "all"]:
            parses = data.get("image_parses") or []
            if parses:
                parse_img = b64_pil(parses[0])
        if cache_dir:
            if mask_img is not None:
                mask_img.save(mask_path)
            if parse_img is not None:
                parse_img.save(parse_path)

        return mask_img, parse_img

    except Exception:
        return None, None


# def run_human_tasks_grpc(address, input_image_b64, mode):
#     # human-segmentation-app:8080
#     # human-segmentation-app.zddev.metac-inc.com:8080
#     # channel = grpc.insecure_channel(address)

#     channel = grpc.insecure_channel(address, options= [('grpc.max_send_message_length', int('268435456')), ('grpc.max_receive_message_length', int('268435456'))])

#     handler_service = handler_common_pb2_grpc.HandlerServiceStub(channel)
    
#     if mode not in ['seg', 'parse','all']:
#         return None, None
    
#     string_value = {
#         "method": mode,
#         "data": {
#                     "image": input_image_b64,
#                     "read_from_url": False
#                 }
#         }
    
#     string_value = json.dumps(string_value)
#     params = common_pb2.Value(string_value=string_value)
#     list_value = common_pb2.ListValue(elements=[params])
#     request = handler_common_pb2.Request(serviceName='vton_human_tasks', arguments=list_value)
#     response = handler_service.handle(request, timeout=60)

#     try:
#         params_out = json.loads(response.data.string_value)
#         if params_out['code'] != 1000:
#             return None, None
#         else:
#             if mode == 'all':
#                 image_mask_img = b64_pil(params_out['data']['image_masks'][0])
#                 image_parse_img = b64_pil(params_out['data']['image_parses'][0])
#             elif mode == 'seg':
#                 image_mask_img = b64_pil(params_out['data']['image_masks'][0])
#                 image_parse_img = None
#             else:
#                 image_mask_img = None
#                 image_parse_img = b64_pil(params_out['data']['image_parses'][0])
            
#             return image_mask_img, image_parse_img
#     except:
#         return None, None
    
if __name__ == '__main__':
    img = Image.open('/home/fangjingwu/data/dataset/1028test/upper/ours_test_res/100001.jpg')
    image_mask_img, image_parse_img = run_human_tasks_grpc('172.16.0.17:8080', pil_b64(img), 'all',cache_dir='./cache_dir')
    image_parse_mask= np.array(image_parse_img)
    top_mask = (image_parse_mask == 23).astype(np.uint8) * 255  
    outerwear_mask = (image_parse_mask == 20).astype(np.uint8) * 255
    one_piece_mask = (image_parse_mask == 14).astype(np.uint8) * 255

    # 取三个集合的并集作为上半身的mask
    upper_body_mask = cv2.bitwise_or(top_mask, outerwear_mask)
    upper_body_mask = cv2.bitwise_or(upper_body_mask, one_piece_mask)
    image_parse_img = Image.fromarray(upper_body_mask, mode='L')  # 转换为灰度图
    if image_mask_img is not None:
        image_mask_img.save('mask.png')
    if image_parse_img is not None:
        image_parse_img.save('parse.png')
    # 获取image_parse_img的最小外接矩形
    bbox = image_parse_img.getbbox()
    # 保存外接矩形图
    if bbox:
        cropped_image = img.crop(bbox)
        cropped_image.save('parse_cropped.png')
    else:
        print("No bounding box found, cannot crop the image.")