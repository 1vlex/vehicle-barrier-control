from pycocotools.coco import COCO
import os
import shutil
import yaml

base_dir = r"D:\test"
annotations_dir = os.path.join(base_dir, "annotations")
output_dit = os.path.join(base_dir, "vehicles_dataset")

COCO_PATHS = {
    "train": {
        "images": os.path.join(base_dir, "train2017"),
        "ann_file": os.path.join(annotations_dir, "instances_train2017.json")
    },
    "val": {
        "images": os.path.join(base_dir, "val2017"),
        "ann_file": os.path.join(annotations_dir, "instances_val2017.json")
    }
}

CATEGORIES = ['bus',"car", "motorcycle",'truck']

def extract_vehicles(coco_mode):
    print(f"\nОбработка {coco_mode} данных...")
    coco = COCO(COCO_PATHS[coco_mode]["ann_file"])
    
    cat_ids = coco.getCatIds(catNms=CATEGORIES)
    print(f"Категории: {CATEGORIES} -> ID: {cat_ids}")
    
    img_ids = set()
    for cat_id in cat_ids:
        img_ids.update(coco.getImgIds(catIds=cat_id))
    
    print(f"Найдено изображений: {len(img_ids)}")
    
    images_dir = os.path.join(output_dit, coco_mode, "images")
    labels_dir = os.path.join(output_dit, coco_mode, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        src_img = os.path.join(COCO_PATHS[coco_mode]["images"], img_info["file_name"])
        dst_img = os.path.join(images_dir, img_info["file_name"])
        
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
            
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
            anns = coco.loadAnns(ann_ids)
            
            txt_file = os.path.splitext(img_info["file_name"])[0] + ".txt"
            dst_txt = os.path.join(labels_dir, txt_file)
            
            with open(dst_txt, "w") as f:
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    img_w, img_h = img_info["width"], img_info["height"]
                    
                    x_center = (x + w / 2) / img_w
                    y_center = (y + h / 2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h

                    f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")

    return images_dir

def create_yaml_config(train_path, val_path):
    """Создает data.yaml для YOLO"""
    yaml_path = os.path.join(output_dit, "data.yaml")
    
    config = {
        "path": output_dit,
        "train": os.path.relpath(train_path, output_dit),
        "val": os.path.relpath(val_path, output_dit),
        "names": {0: "vehicle"},
        "nc": 1
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"\nСоздан YOLO конфиг: {yaml_path}")
    return yaml_path

def main():
    os.makedirs(output_dit, exist_ok=True)
    
    train_path = extract_vehicles("train")
    val_path = extract_vehicles("val")
    yaml_path = create_yaml_config(train_path, val_path)

    train_labels = os.path.join(output_dit, "train", "labels")
    val_labels = os.path.join(output_dit, "val", "labels")
    
    print(f"  - Train: {len(os.listdir(train_path))} изображений, {len(os.listdir(train_labels))} аннотаций")
    print(f"  - Val: {len(os.listdir(val_path))} изображений, {len(os.listdir(val_labels))} аннотаций")
    print(f"  - YAML: {yaml_path}")

if __name__ == "__main__":
    main()