from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")

image_path = "test_images/shelf.jpg"
results = model(image_path)

img = cv2.imread(image_path)

boxes = []
for r in results:
    for b in r.boxes.xyxy:
        x1, y1, x2, y2 = map(int, b)
        boxes.append((x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imshow("Detected Products", img)
cv2.waitKey(0)
