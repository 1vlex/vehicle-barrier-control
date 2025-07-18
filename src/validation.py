from ultralytics import YOLO
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data_yaml,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        split="val",
        name="vehicles_validation",
        exist_ok=True
    )

    print("\n=== Результаты валидации ===")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
