import pandas as pd
import os
import json
from PIL import Image

CSV_PATH = "dataset/annotations.csv"
IMG_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"

os.makedirs(LABEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for _, row in df.iterrows():
    region_attr = json.loads(row["region_attributes"])
    if region_attr.get("type") != "damaged":
        continue  # only damaged products

    shape = json.loads(row["region_shape_attributes"])
    img_path = os.path.join(IMG_DIR, row["filename"])

    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path)
    img_w, img_h = img.size

    x = shape["x"]
    y = shape["y"]
    w = shape["width"]
    h = shape["height"]

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h

    label_file = os.path.join(
        LABEL_DIR, row["filename"].replace(".JPG", ".txt")
    )

    with open(label_file, "a") as f:
        f.write(f"0 {x_center} {y_center} {w} {h}\n")

print("✅ CSV converted to YOLO labels (damaged only)")
