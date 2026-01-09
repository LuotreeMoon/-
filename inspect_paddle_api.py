from paddleocr import PaddleOCR
import os

IMG_DIR = r"G:\liscence_detect\dataset\ccpd_yolo\images\val"
img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')]
img_path = os.path.join(IMG_DIR, img_files[0]) if img_files else None

ocr = PaddleOCR(lang='ch', use_textline_orientation=True)
print('PaddleOCR instance methods:', [m for m in dir(ocr) if not m.startswith('_')])
print('Inspecting OCR on:', img_path)
# Use the current public API `predict` and print detailed structure
res = ocr.predict(img_path)
print('OCR result type:', type(res))
print('OCR result sample (repr):')
print(repr(res))
if isinstance(res, list) and len(res) > 0:
	print('First item type:', type(res[0]))
	print('First item repr:', repr(res[0]))
