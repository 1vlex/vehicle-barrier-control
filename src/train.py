from ultralytics import YOLO
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--model-type", default="yolov8l.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=540)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = YOLO(args.model_type)
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name="vehicles_detection",
        exist_ok=True
    )
