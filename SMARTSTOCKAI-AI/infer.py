import sys
import json
import cv2
import os
from ultralytics import YOLO

os.environ["YOLO_VERBOSE"] = "False"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_path = sys.argv[1]

model = YOLO(MODEL_PATH)

img = cv2.imread(image_path)

results = model(image_path, verbose=False)

defective = False

for r in results:
    if r.boxes is not None and len(r.boxes) > 0:
        defective = True
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = max(x2 - x1, y2 - y1) // 2
            cv2.circle(img, (cx, cy), radius, (0, 0, 255), 3)

output_image = os.path.join(
    OUTPUT_DIR, "result_" + os.path.basename(image_path)
)

cv2.imwrite(output_image, img)

filename = "result_" + os.path.basename(image_path)

response = {
    "status": "NOT_OK" if defective else "OK",
    "message": "❌ Defective product detected" if defective else "✅ Product is OK",
    "output_image_path": f"/outputs/{filename}"
}

print(json.dumps(response))
