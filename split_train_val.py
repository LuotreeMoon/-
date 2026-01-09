import os
import random
import shutil

DATA_DIR = r"G:\liscence_detect\dataset\ccpd_yolo"
TRAIN_RATIO = 0.8

imgs = os.listdir(os.path.join(DATA_DIR, "images"))
random.shuffle(imgs)

train_num = int(len(imgs) * TRAIN_RATIO)

train_imgs = imgs[:train_num]
val_imgs = imgs[train_num:]

def move_files(img_list, split):
    img_out = os.path.join(DATA_DIR, "images", split)
    label_out = os.path.join(DATA_DIR, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    for img in img_list:
        shutil.move(
            os.path.join(DATA_DIR, "images", img),
            os.path.join(img_out, img)
        )
        shutil.move(
            os.path.join(DATA_DIR, "labels", img.replace(".jpg", ".txt")),
            os.path.join(label_out, img.replace(".jpg", ".txt"))
        )

move_files(train_imgs, "train")
move_files(val_imgs, "val")
