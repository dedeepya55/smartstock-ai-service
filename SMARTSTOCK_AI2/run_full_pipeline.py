import cv2
import numpy as np
import json
import os
import argparse
from scripts.check_arrangement import check_arrangement

# ===============================
# YOLO IMPORT
# ===============================
try:
    from ultralytics import YOLO
    USE_YOLO = True
    print("✓ YOLOv8 loaded successfully\n")
except ImportError:
    USE_YOLO = False
    print("✗ YOLOv8 not found. Install with: pip install ultralytics\n")

# ===============================
# EASYOCR INIT (ONLY OCR ENGINE)
# ===============================
try:
    import easyocr
    OCR_READER = easyocr.Reader(['en'], gpu=False)
    USE_EASYOCR = True
    print("✓ EasyOCR initialized\n")
except:
    OCR_READER = None
    USE_EASYOCR = False
    print("✗ EasyOCR not available\n")

# ===============================
# ARGUMENT PARSING
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True, help='Path to input image')
args = parser.parse_args()
image_path = args.image

# ===============================
# PATH SETUP
# ===============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(SCRIPT_DIR, "models", "best.pt")

output_folder = os.path.join(SCRIPT_DIR, "results")
os.makedirs(output_folder, exist_ok=True)

# ===============================
# LOAD IMAGE
# ===============================
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

original_image = image.copy()
image_height, image_width = original_image.shape[:2]

# ===============================
# YOLO DETECTION
# ===============================
product_regions = []

if USE_YOLO:
    model = YOLO(model_path)
    results = model(image_path, conf=0.3)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            product_regions.append({
                "bbox": (x1, y1, x2, y2),
                "x_center": (x1 + x2) // 2,
                "y_center": (y1 + y2) // 2,
                "region_img": original_image[y1:y2, x1:x2]
            })

    print(f"✓ Detected {len(product_regions)} products\n")

# ===============================
# FALLBACK SPLIT
# ===============================
if not product_regions:
    mid_y = image_height // 2
    product_regions = [
        {
            "bbox": (0, 0, image_width, mid_y),
            "x_center": image_width // 2,
            "y_center": mid_y // 2,
            "region_img": original_image[:mid_y]
        },
        {
            "bbox": (0, mid_y, image_width, image_height),
            "x_center": image_width // 2,
            "y_center": (mid_y + image_height) // 2,
            "region_img": original_image[mid_y:]
        }
    ]

# ===============================
# OCR FUNCTION (EASYOCR ONLY)
# ===============================
def extract_text_from_region(region_img):
    if not USE_EASYOCR or not OCR_READER:
        return []

    h, w = region_img.shape[:2]

    if max(h, w) > 400:
        scale = 400 / max(h, w)
        region_img = cv2.resize(region_img, (int(w * scale), int(h * scale)))

    results = OCR_READER.readtext(region_img, detail=1)
    return [text.strip() for (_, text, conf) in results if conf > 0.3]

# ===============================
# PRODUCT NAME EXTRACTION
# ===============================
def extract_product_name(text_list):
    if not text_list:
        return "unknown"

    clean = []
    for t in text_list:
        t = t.strip("[](){}'\".,;:!?")
        if any(c.isalpha() for c in t):
            clean.append(t)

    return max(clean, key=len) if clean else "unknown"

# ===============================
# GROUP BY ROWS
# ===============================
def group_by_rows(regions, row_distance=30):
    regions = sorted(regions, key=lambda r: r["y_center"])
    rows = []
    current = [regions[0]]

    for r in regions[1:]:
        if abs(r["y_center"] - np.mean([x["y_center"] for x in current])) < row_distance:
            current.append(r)
        else:
            rows.append(current)
            current = [r]

    rows.append(current)
    return rows

rows = group_by_rows(product_regions)

# ===============================
# BUILD DETECTIONS
# ===============================
detections_for_arrangement = []
results_by_rows = []
product_names_by_row = []

for row in rows:
    row = sorted(row, key=lambda r: r["x_center"])
    row_names = []
    row_texts = []

    for product in row:
        text_list = extract_text_from_region(product["region_img"])
        name = extract_product_name(text_list)

        row_names.append(name)
        row_texts.append(text_list)

        detections_for_arrangement.append({
            "label": name,
            "ocr_text": text_list,
            "x_center": product["x_center"],
            "y_center": product["y_center"],
            "bbox": product["bbox"]
        })

    results_by_rows.append(row_texts)
    product_names_by_row.append(row_names)

# ===============================
# CHECK ARRANGEMENT
# ===============================
status, messages, wrong_boxes = check_arrangement(
    detections_for_arrangement, row_thresh=30
)

# ===============================
# ANNOTATION
# ===============================
annotated_img = original_image.copy()

for det in detections_for_arrangement:
    x1, y1, x2, y2 = det["bbox"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    radius = max(x2 - x1, y2 - y1) // 2

    if det["bbox"] in wrong_boxes:
        cv2.circle(annotated_img, (cx, cy), radius, (0, 0, 255), 3)
        cv2.putText(
            annotated_img,
            f"MISPLACED:{det['label']}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    else:
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_img,
            det["label"],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

cv2.putText(
    annotated_img,
    f"STATUS: {status}",
    (30, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0) if status == "CORRECT" else (0, 0, 255),
    3,
)

# ===============================
# SAVE OUTPUTS
# ===============================
output_image_path = os.path.join(output_folder, "annotated_" + os.path.basename(image_path))
cv2.imwrite(output_image_path, annotated_img)

json_path = os.path.join(output_folder, "detection_results.json")
with open(json_path, "w") as f:
    json.dump(
        {
            "status": status,
            "messages": messages,
            "misplaced_boxes": [
                {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in wrong_boxes
            ],
            "row_wise_ocr": results_by_rows,
            "row_wise_products": product_names_by_row,
        },
        f,
        indent=2,
    )

print(f"✓ Annotated image saved: {output_image_path}")
print(f"✓ JSON saved: {json_path}")