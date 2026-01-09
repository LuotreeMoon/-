import os
import cv2
import re
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# =========================
# 1. 路径配置（按需修改）
# =========================
YOLO_MODEL_PATH = r"runs/detect/train/weights/best.pt"
IMAGE_DIR = r"dataset/ccpd_yolo/images/val"
OUTPUT_DIR = r"crop_results"
result_file = open("results_ccpd.txt", "w", encoding="utf-8")
result_file.write("filename,gt,pred\n")


os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. 初始化模型
# =========================
model = YOLO(YOLO_MODEL_PATH)

ocr = PaddleOCR(
    lang="ch",
    use_textline_orientation=True   # 新参数，替代 use_angle_cls
)

# =========================
# 3. CCPD 文件名解析函数
# =========================
PROVINCES = [
    "皖","沪","津","渝","冀","晋","蒙","辽","吉","黑",
    "苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
    "桂","琼","川","贵","云","藏","陕","甘","青","宁","新"
]

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"

def parse_ccpd_label(filename):
    """
    从 CCPD 文件名解析车牌真值
    """
    name = os.path.basename(filename)
    parts = name.split("-")
    if len(parts) < 5:
        return ""

    label_ids = parts[4].split("_")
    plate = PROVINCES[int(label_ids[0])]
    for i in label_ids[1:]:
        plate += ALPHABET[int(i)]
    return plate

# =========================
# 4. 主评估循环
# =========================
total = 0
correct = 0

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    gt_plate = parse_ccpd_label(img_name)

    # ---------- YOLO 检测 ----------
    results = model(img, conf=0.25, iou=0.5, verbose=False)

    pred_plate = ""

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)

            # ---------- 扩边裁剪 ----------
            h, w, _ = img.shape
            pad_x = int(0.05 * (x2 - x1))
            pad_y = int(0.25 * (y2 - y1))

            x1p = max(0, x1 - pad_x)
            y1p = max(0, y1 - pad_y)
            x2p = min(w, x2 + pad_x)
            y2p = min(h, y2 + pad_y)

            plate_img = img[y1p:y2p, x1p:x2p]
            if plate_img.size == 0:
                continue

            # ---------- resize + RGB ----------
            plate_img = cv2.resize(plate_img, (320, 96))
            plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)

            # ---------- OCR ----------
            ocr_result = ocr.predict(plate_rgb)

                # 兼容 PaddleOCR / PaddleX 返回格式：优先使用 rec_scores 选取最高分的 rec_texts
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                    item = ocr_result[0]
                    rec_texts = None
                    rec_scores = None
                    if isinstance(item, dict):
                        rec_texts = item.get('rec_texts') or item.get('rec_text')
                        rec_scores = item.get('rec_scores')
                    else:
                        try:
                            rec_texts = item['rec_texts']
                            rec_scores = item['rec_scores']
                        except Exception:
                            rec_texts = getattr(item, 'rec_texts', None) or getattr(item, 'rec_text', None)
                            rec_scores = getattr(item, 'rec_scores', None)

                    if rec_texts:
                        # 若有得分信息，则选分最高的识别结果；否则取最后一个非空结果
                        if rec_scores is not None and len(rec_scores) == len(rec_texts):
                            idx = int(np.argmax(rec_scores))
                            cand = rec_texts[idx]
                            if isinstance(cand, str):
                                pred_plate = cand.strip()
                        else:
                            for t in reversed(rec_texts):
                                if t:
                                    pred_plate = t.strip()
                                    break
            break  # 只取第一个车牌

    # ---------- 清洗 OCR 结果 ----------
    pred_plate = re.sub(r"[^A-Z0-9\u4e00-\u9fa5]", "", pred_plate)

    # ---------- 保存裁剪车牌 ----------
    save_name = f"{os.path.splitext(img_name)[0]}_pred_{pred_plate}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), plate_img)

    # ---------- 统计 ----------
    total += 1
    if pred_plate == gt_plate:
        correct += 1

    # ---------- 打印 ----------
    print(img_name)
    print(f"  真值: {gt_plate}")
    print(f"  预测: {pred_plate}")
    print("-" * 40)
    result_file.write(f"{img_name},{gt_plate},{pred_plate}\n")


# =========================
# 5. 最终准确率
# =========================
result_file.close()
acc = correct / total if total > 0 else 0
print(f"\n总样本数: {total}")
print(f"完全匹配正确数: {correct}")
print(f"整牌识别准确率: {acc:.4f}")
