from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # base pretrained model

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        project="outputs",
        name="smartstock_train",
        device="cpu"   # safe for Windows
    )

if __name__ == "__main__":
    main()
