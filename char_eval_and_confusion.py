import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULT_FILE = "results_ccpd.txt"

total_chars = 0
correct_chars = 0

confusion = defaultdict(lambda: defaultdict(int))
all_chars = set()

with open(RESULT_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gt = row["gt"]
        pred = row["pred"]

        if not gt or not pred:
            continue

        min_len = min(len(gt), len(pred))
        for i in range(min_len):
            g, p = gt[i], pred[i]
            total_chars += 1
            all_chars.add(g)
            all_chars.add(p)
            if g == p:
                correct_chars += 1
            else:
                confusion[g][p] += 1

# ---------- 字符级准确率 ----------
char_acc = correct_chars / total_chars if total_chars else 0
print(f"字符级准确率: {char_acc:.4f}")
print(f"字符总数: {total_chars}")

# ---------- 构造混淆矩阵 ----------
chars = sorted(list(all_chars))
char_to_idx = {c: i for i, c in enumerate(chars)}

matrix = np.zeros((len(chars), len(chars)), dtype=int)

for g in confusion:
    for p in confusion[g]:
        matrix[char_to_idx[g], char_to_idx[p]] += confusion[g][p]

# ---------- 画混淆矩阵 ----------
plt.figure(figsize=(12, 10))
sns.heatmap(
    matrix,
    xticklabels=chars,
    yticklabels=chars,
    cmap="Reds",
    fmt="d"
)
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.title("Character Confusion Matrix")
plt.tight_layout()
plt.savefig("char_confusion_matrix.png", dpi=300)
plt.show()
