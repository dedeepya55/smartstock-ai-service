import os
import random
import shutil

BASE = os.path.join(os.path.dirname(__file__), "dataset")

IMG_DIR = os.path.join(BASE, "images")
LBL_DIR = os.path.join(BASE, "labels")

IMG_TRAIN = os.path.join(IMG_DIR, "train")
IMG_VAL = os.path.join(IMG_DIR, "val")
LBL_TRAIN = os.path.join(LBL_DIR, "train")
LBL_VAL = os.path.join(LBL_DIR, "val")

os.makedirs(IMG_TRAIN, exist_ok=True)
os.makedirs(IMG_VAL, exist_ok=True)
os.makedirs(LBL_TRAIN, exist_ok=True)
os.makedirs(LBL_VAL, exist_ok=True)

# ✅ INCLUDE .webp
images = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp",".JPG"))
]

print(f"🖼️ Found {len(images)} images")

random.shuffle(images)
split = int(0.8 * len(images))

train_imgs = images[:split]
val_imgs = images[split:]

def move(files, img_dst, lbl_dst):
    for img in files:
        lbl = os.path.splitext(img)[0] + ".txt"
        img_src = os.path.join(IMG_DIR, img)
        lbl_src = os.path.join(LBL_DIR, lbl)

        if os.path.exists(lbl_src):
            shutil.move(img_src, os.path.join(img_dst, img))
            shutil.move(lbl_src, os.path.join(lbl_dst, lbl))
        else:
            print(f"⚠️ Missing label for {img}")

move(train_imgs, IMG_TRAIN, LBL_TRAIN)
move(val_imgs, IMG_VAL, LBL_VAL)

print("✅ Dataset split completed successfully")
