from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data/SKU110K_fixed/data.yaml",
    epochs=40,
    imgsz=416,
    batch=32
)
