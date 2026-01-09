import os
import cv2
from tqdm import tqdm

# ===== 路径配置 =====
CCPD_IMG_DIR = r"G:\BaiduNetdiskDownload\CCPD2019.tar\CCPD2019\CCPD2019\ccpd_base"
OUTPUT_IMG_DIR = r"G:\liscence_detect\dataset\ccpd_yolo\images"
OUTPUT_LABEL_DIR = r"G:\liscence_detect\dataset\ccpd_yolo\labels"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# ===== 遍历部分图片（只处理最多 MAX_IMAGES 张） =====
MAX_IMAGES = 1000

# 收集所有 jpg 文件并只取前 MAX_IMAGES 个
jpg_files = [f for f in os.listdir(CCPD_IMG_DIR) if f.lower().endswith('.jpg')]
selected_files = jpg_files[:MAX_IMAGES]

for img_name in tqdm(selected_files, total=min(MAX_IMAGES, len(jpg_files))):
    img_path = os.path.join(CCPD_IMG_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: failed to read {img_name}, skipped")
        continue

    h, w, _ = img.shape

    try:
        # CCPD 文件名解析
        parts = img_name.split('-')
        bbox_part = parts[2]          # x1&y1_x2&y2
        x1y1, x2y2 = bbox_part.split('_')

        x1, y1 = map(int, x1y1.split('&'))
        x2, y2 = map(int, x2y2.split('&'))

        # YOLO 格式
        xc = (x1 + x2) / 2.0 / w
        yc = (y1 + y2) / 2.0 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # 写 label
        label_path = os.path.join(
            OUTPUT_LABEL_DIR,
            img_name.replace(".jpg", ".txt")
        )
        with open(label_path, "w") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # 复制图片
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, img_name), img)

    except Exception as e:
        print(f"Error parsing {img_name}: {e}")
