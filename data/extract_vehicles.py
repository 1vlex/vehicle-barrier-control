import argparse
import os
import shutil
import yaml
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract vehicle images and labels from COCO annotations."
    )
    parser.add_argument(
        "--train-coco-json", type=str, required=True,
        help="Path to COCO instances_train2017.json"
    )
    parser.add_argument(
        "--train-images-dir", type=str, required=True,
        help="Directory with COCO train images (e.g. train2017)"
    )
    parser.add_argument(
        "--val-coco-json", type=str, required=True,
        help="Path to COCO instances_val2017.json"
    )
    parser.add_argument(
        "--val-images-dir", type=str, required=True,
        help="Directory with COCO val images (e.g. val2017)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output base directory for vehicles_dataset"
    )
    return parser.parse_args()


CATEGORIES = ['bus', 'car', 'motorcycle', 'truck']


def extract_vehicles(ann_file: str, images_src: str, images_dest: str, labels_dest: str):
    print(f"\nProcessing: {ann_file} -> {images_dest}")
    coco = COCO(ann_file)

    cat_ids = coco.getCatIds(catNms=CATEGORIES)
    print(f"Categories: {CATEGORIES} -> IDs: {cat_ids}")

    img_ids = set()
    for cid in cat_ids:
        img_ids.update(coco.getImgIds(catIds=cid))
    print(f"Found images: {len(img_ids)}")

    os.makedirs(images_dest, exist_ok=True)
    os.makedirs(labels_dest, exist_ok=True)

    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        src_path = os.path.join(images_src, info['file_name'])
        dst_img = os.path.join(images_dest, info['file_name'])

        if not os.path.exists(src_path):
            continue
        shutil.copy(src_path, dst_img)

        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)

        txt_name = os.path.splitext(info['file_name'])[0] + '.txt'
        dst_txt = os.path.join(labels_dest, txt_name)
        with open(dst_txt, 'w') as f:
            for ann in anns:
                x, y, w, h = ann['bbox']
                iw, ih = info['width'], info['height']
                x_c = (x + w/2) / iw
                y_c = (y + h/2) / ih
                w_n = w / iw
                h_n = h / ih
                # YOLO class index: all vehicles -> 0
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

    return len(img_ids)


def create_yaml_config(output_dir: str):
    yaml_path = os.path.join(output_dir, 'data.yaml')
    config = {
        'path': output_dir,
        'train': os.path.relpath(os.path.join(output_dir, 'train', 'images'), output_dir),
        'val': os.path.relpath(os.path.join(output_dir, 'val', 'images'), output_dir),
        'names': {0: 'vehicle'},
        'nc': 1
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"\nYOLO config saved to: {yaml_path}")
    return yaml_path


def main():
    args = parse_args()
    out = args.output
    # Prepare output directories
    for mode in ['train', 'val']:
        os.makedirs(os.path.join(out, mode, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out, mode, 'labels'), exist_ok=True)

    # Process train
    train_count = extract_vehicles(
        args.train_coco_json,
        args.train_images_dir,
        os.path.join(out, 'train', 'images'),
        os.path.join(out, 'train', 'labels')
    )

    # Process val
    val_count = extract_vehicles(
        args.val_coco_json,
        args.val_images_dir,
        os.path.join(out, 'val', 'images'),
        os.path.join(out, 'val', 'labels')
    )

    # Create YOLO data.yaml
    yaml_path = create_yaml_config(out)

    print(f"\nSummary:")
    print(f"  Train: {train_count} images, labels in {os.path.join(out, 'train', 'labels')}")
    print(f"  Val:   {val_count} images, labels in {os.path.join(out, 'val', 'labels')}")
    print(f"  Config YAML: {yaml_path}")


if __name__ == '__main__':
    main()
