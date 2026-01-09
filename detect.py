import os
import cv2
import re
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ======================
# 配置区（按需修改）
# ======================
IMG_DIR = r"G:\liscence_detect\dataset\ccpd_yolo\images\val"      # CCPD 图片目录
MODEL_PATH = r"runs\detect\train\weights\best.pt"
SAVE_DIR = r"G:\liscence_detect\crop_results"   # 裁剪结果保存目录

CONF_THRES = 0.3

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# 初始化模型
# ======================
det_model = YOLO(MODEL_PATH)

ocr = PaddleOCR(
    lang="ch",
    use_textline_orientation=True,
    
)

# ======================
# CCPD 文件名解析真值
# ======================
PROVINCE_MAP = {
    "0": "皖", "1": "沪", "2": "津", "3": "渝", "4": "冀", "5": "晋",
    "6": "蒙", "7": "辽", "8": "吉", "9": "黑", "10": "苏", "11": "浙",
    "12": "京", "13": "闽", "14": "赣", "15": "鲁", "16": "豫", "17": "鄂",
    "18": "湘", "19": "粤", "20": "桂", "21": "琼", "22": "川", "23": "贵",
    "24": "云", "25": "藏", "26": "陕", "27": "甘", "28": "青", "29": "宁",
    "30": "新"
}

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ"
ADS = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

def parse_ccpd_label(filename):
    """
    CCPD 格式：
    xxxx-xx_xx-...-0_0_7_26_17_33_29-xx.jpg
    """
    try:
        code = filename.split("-")[4].split("_")
        prov = PROVINCE_MAP[code[0]]
        alpha = ALPHABET[int(code[1])]
        rest = "".join([ADS[int(x)] for x in code[2:]])
        return prov + alpha + rest
    except:
        return ""

# ======================
# 主评估流程
# ======================
total = 0
correct = 0

img_list = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]

for img_name in img_list:
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    gt_plate = parse_ccpd_label(img_name)

    # YOLO 检测
    results = det_model(img, conf=CONF_THRES, device=0)[0]

    pred_plate = ""

    if results.boxes is not None and len(results.boxes) > 0:
        box = results.boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        # ===== 扩边裁剪（关键）=====
        h, w, _ = img.shape
        pad_x = int(0.05 * (x2 - x1))
        pad_y = int(0.25 * (y2 - y1))

        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)
        x2p = min(w, x2 + pad_x)
        y2p = min(h, y2 + pad_y)

        plate_img = img[y1p:y2p, x1p:x2p]

        if plate_img.size > 0:
            # OCR 友好 resize
            plate_img = cv2.resize(plate_img, (320, 96))

            # OCR
            ocr_result = ocr.predict(plate_img)

            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                item = ocr_result[0]
                if isinstance(item, dict) and "rec_text" in item:
                    txt = item["rec_text"]
                    if isinstance(txt, list) and len(txt) > 0:
                        pred_plate = txt[0]

            # ===== 保存裁剪图像 =====
            safe_pred = pred_plate if pred_plate else "NONE"
            save_name = f"{os.path.splitext(img_name)[0]}_{safe_pred}.jpg"
            save_path = os.path.join(SAVE_DIR, save_name)
            cv2.imwrite(save_path, plate_img)

    # ===== 统计 =====
    total += 1
    if pred_plate == gt_plate:
        correct += 1

    # 打印结果
    print(img_name)
    print("  真值:", gt_plate)
    print("  预测:", pred_plate)
    print("-" * 40)

# ======================
# 总体准确率
# ======================
acc = correct / total if total > 0 else 0
print(f"\n总样本数: {total}")
print(f"完全匹配数: {correct}")
print(f"整牌准确率: {acc:.4f}")
