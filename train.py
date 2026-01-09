import os
import multiprocessing
import platform
import torch
from ultralytics import YOLO


def main():
    # ===== 使用 CPU 训练（适用于无 GPU 的云实例） =====
    device = "cpu"

    # CPU 信息与并行 workers 自动设置
    n_cpus = os.cpu_count() or multiprocessing.cpu_count()
    workers = max(1, min(8, n_cpus - 1))

    print("运行平台：", platform.platform())
    print("训练设备：CPU")
    print(f"可用 CPU 核心: {n_cpus}, 设置 workers={workers}")

    # ===== 加载模型（使用工作区内的 yolov8n.pt） =====
    model = YOLO("yolov8n.pt")

    # ===== 开始训练（CPU 模式） =====
    # 在 CPU 上训练时需减小 batch 大小，并禁用 AMP/half 精度
    model.train(
        data="configs/ccpd.yaml",
        imgsz=640,
        epochs=50,
        batch=8,
        device=device,
        name="ccpd_lp_cpu",
        project="runs",
        workers=workers,
        amp=False,
        half=False,
    )


if __name__ == "__main__":
    main()
