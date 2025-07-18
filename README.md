# Vehicle Detection & Barrier Control

Автоматическая детекция и трекинг транспортных средств на видео с управлением «шлагбаумом».

---

## 📂 Структура проекта

```
vehicle-barrier-control/
├── .gitignore                 
├── LICENSE                    # MIT License
├── README.md                  
├── requirements.txt           # список зависимостей
├── data/
│   └── extract_vehicles.py    # скрипт подготовки COCO → YOLO‑датасет
├── config/
│   └── data.yaml              # конфиг для обучения YOLOv8
├── notebook/
│   └── CV-Tracking.ipynb      # оригинальный Jupyter‑ноутбук (опционально)
├── src/
│   ├── train.py               # скрипт обучения модели
│   ├── validation.py          # скрипт валидации модели
│   └── main.py                # демо: детекция + управление шлагбаумом
```

---

## 🚀 Установка

```bash
git clone https://github.com/1vlex/vehicle-barrier-control.git
cd vehicle-barrier-control

python3 -m venv env
source env/bin/activate    # Linux/macOS
env\Scripts\activate       # Windows

pip install -r requirements.txt
```

---

## 🔧 Конфигурация

В `configs/vehicles.yaml` укажите пути и параметры:

```yaml
train: ../data/vehicles_dataset/train/images
val:   ../data/vehicles_dataset/val/images

nc: 1
names: ['vehicle']
```

---

## 📈 Подготовка данных

```bash
# Для train
python data/extract_vehicles.py \
  --coco-json /path/to/instances_train2017.json \
  --images-dir /path/to/train2017 \
  --output data/vehicles_dataset

# Для val
python data/extract_vehicles.py \
  --coco-json /path/to/instances_val2017.json \
  --images-dir /path/to/val2017 \
  --output data/vehicles_dataset
```

Результат:

```
data/vehicles_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## Обучение

```bash
python src/train.py \
  --data-yaml data/vehicles_dataset/data.yaml \
  --model-type yolov8l.pt \
  --epochs 30 \
  --imgsz 540 \
  --batch 16 \
  --device cuda
```

---

## Валидация

```bash
python src/validation.py \
  --data-yaml data/vehicles_dataset/data.yaml \
  --weights runs/detect/vehicles_detection/weights/best.pt \
  --imgsz 640 \
  --batch 16 \
  --device cuda
```

---

## 🎥 Демонстрация (Barrier Control)

```bash
python src/main.py --video /path/to/your/video.avi
```

Результат откроет окно с видео, отрисованными рамками, траекториями и текущим состоянием шлагбаума (OPEN/CLOSED).

---

## Лицензия

Этот проект распространяется под MIT License.
