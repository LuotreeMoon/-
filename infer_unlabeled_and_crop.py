import os
import cv2
import re
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# =========================
# 1. 路径配置
# =========================
IMG_DIR = r"dataset/teacher_images"
OUTPUT_DIR = r"outputs_unlabeled"
MODEL_PATH = r"runs/detect/train/weights/best.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULT_TXT = os.path.join(OUTPUT_DIR, "unlabeled_results.txt")

# =========================
# 2. 初始化模型
# =========================
model = YOLO(MODEL_PATH)

ocr = PaddleOCR(
    lang="ch",
    use_textline_orientation=True
)

# =========================
# 3. 推理增强工具函数
# =========================

# —— 3.1 车牌去倾斜（deskew）
def deskew_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None:
        return img

    angles = []

    # 关键修复点：兼容 (N,1,2) 结构
    for line in lines[:10]:
        if len(line) == 1:
            rho, theta = line[0]
        else:
            rho, theta = line

        angle = (theta - np.pi / 2) * 180 / np.pi
        angles.append(angle)

    if len(angles) == 0:
        return img

    avg_angle = np.mean(angles)

    # 小角度不处理，防止抖动
    if abs(avg_angle) < 1.0:
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), avg_angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)


# —— 3.2 字符集约束
VALID_CHARS = (
    "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
)

CONFUSION_MAP = {
    "汤": "浙",
    "湘": "粤",
    "O": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8"
}

def normalize_plate(text):
    text = "".join(c for c in text if c in VALID_CHARS)
    text = "".join(CONFUSION_MAP.get(c, c) for c in text)
    return text

# —— 3.3 多角度 OCR
ROT_ANGLES = [-15, -8, 0, 8, 15]

def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

# =========================
# 4. 开始推理
# =========================
with open(RESULT_TXT, "w", encoding="utf-8") as f:
    f.write("filename,pred\n")

    for idx, img_name in enumerate(os.listdir(IMG_DIR)):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        pred_plate = ""
        plate_img = None

        # ---------- YOLO 检测 ----------
        results = model(img, conf=0.25, verbose=False)

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                break

            x1, y1, x2, y2 = r.boxes.xyxy[0].cpu().numpy().astype(int)

            # ---------- 更激进 & 非对称扩边 ----------
            h, w, _ = img.shape
            bw, bh = x2 - x1, y2 - y1

            x1p = max(0, x1 - int(0.15 * bw))
            x2p = min(w, x2 + int(0.15 * bw))
            y1p = max(0, y1 - int(0.20 * bh))
            y2p = min(h, y2 + int(0.55 * bh))

            plate_img = img[y1p:y2p, x1p:x2p]
            if plate_img.size == 0:
                break

            # ---------- OCR 前处理 ----------
            plate_img = cv2.resize(plate_img, (320, 96))
            plate_img = deskew_plate(plate_img)

            best_text = ""
            best_score = -1

            # ---------- 多角度 OCR ----------
            for ang in ROT_ANGLES:
                rot = rotate_image(plate_img, ang)
                rot_rgb = cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)

                ocr_result = ocr.predict(rot_rgb)

                if isinstance(ocr_result, list) and len(ocr_result) > 0:
                    item = ocr_result[0]
                    rec_texts = item.get('rec_texts') or item.get('rec_text')
                    rec_scores = item.get('rec_scores')

                    if rec_texts:
                        if rec_scores is not None and len(rec_scores) == len(rec_texts):
                            i = int(np.argmax(rec_scores))
                            if rec_scores[i] > best_score:
                                best_score = rec_scores[i]
                                best_text = rec_texts[i]
                        else:
                            best_text = rec_texts[-1]

            pred_plate = normalize_plate(best_text.strip())
            break

        # ---------- 文件保存 ----------
        if pred_plate:
            save_name = f"{pred_plate}.jpg"
        else:
            save_name = f"NONE_{idx:03d}.jpg"

        save_path = os.path.join(OUTPUT_DIR, save_name)
        if plate_img is None or plate_img.size == 0:
            cv2.imwrite(save_path, img)
        else:
            cv2.imwrite(save_path, plate_img)

        f.write(f"{img_name},{pred_plate}\n")
        print(f"{img_name} -> {pred_plate}")

print("\n无标注图片识别完成（已启用推理增强），结果保存在:", OUTPUT_DIR)
