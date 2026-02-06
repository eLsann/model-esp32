# Smart Cane Object Detection - ESP32-CAM

Object detection system based on **YOLOv5n** for ESP32-CAM, designed as a navigation aid for the visually impaired.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLO](https://img.shields.io/badge/YOLO-v5n-green)
![ESP32](https://img.shields.io/badge/ESP32-CAM-orange)

---

## Features

- **6 Class Detection**: dinding (wall), kendaraan_parkir (parked vehicle), orang (person), pintu (door), pohon (tree), tiang (pole)
- **Realtime Detection**: ~30 FPS on PC, ~5 FPS on ESP32
- **Auto-Annotation**: Automatic annotation from folder structure
- **Dataset Balancing**: 150 images/class with augmentation
- **ONNX Export**: Optimized ~10MB model for embedded deployment

---

## Tech Stack

### 🖥️ Hardware
- **ESP32-CAM**: The main microcontroller for deployment and inference (AI-on-Edge).
- **PC / GPU**: Used for model training and dataset processing (NVIDIA GPU recommended for faster training).
- **Camera Module**: OV2640 (standard ESP32-CAM module).

### 🧠 AI & Machine Learning
- **YOLOv5n (Ultralytics)**: Nano version of YOLOv5 architecture, optimized for speed and low resources.
- **PyTorch**: The deep learning framework backend used for training the model.
- **ONNX (Open Neural Network Exchange)**: Format used for interoperability and deploying the model to edge devices.

### 🛠️ Software & Libraries
- **Python 3.10+**: Core programming language for training and data scripts.
- **OpenCV (cv2)**: Image processing, resizing, and real-time visualization.
- **NumPy & Pandas**: Data manipulation and dataset handling.
- **Matplotlib & Seaborn**: Visualization of dataset distribution and training metrics.
- **Jupyter Notebook**: Interactive environment for training workflow.
- **Git**: Version control system.

---

## Pipeline Flow

```mermaid
flowchart LR
    A["Dataset<br/>per Class"] --> B["Auto<br/>Annotation"]
    B --> C["Balancing<br/>150/class"]
    C --> D["Training<br/>YOLOv5n"]
    D --> E["Export<br/>ONNX"]
    E --> F["Deploy<br/>ESP32"]
```

---

## Training Flow Detail

```mermaid
flowchart TD
    subgraph INPUT["INPUT"]
        A[("Images per Folder")]
    end

    subgraph PROCESS["PROCESSING"]
        B["Auto-Annotation"]
        C{"Count >= 150?"}
        D["Undersample"]
        E["Oversample + Aug"]
        F["Split 70/20/10"]
    end

    subgraph TRAINING["TRAINING"]
        G["YOLOv5n 100 Epochs"]
        H["Evaluate Test Set"]
        I["Export ONNX"]
    end

    A --> B --> C
    C -->|Yes| D --> F
    C -->|No| E --> F
    F --> G --> H --> I
```

---

## Realtime Detection Flow

```mermaid
flowchart LR
    A["Camera"] --> B["Resize 96x96"]
    B --> C["ONNX Inference"]
    C --> D{"Detected?"}
    D -->|Yes| E["ACTIVE<br/>Draw BBox"]
    D -->|No| F["IDLE"]
```

---

## Feedback System (Smart Cane)

```mermaid
flowchart TD
    A["Detection"] --> B{"Priority?"}
    B -->|High<br/>person, pole| C["Strong Alert"]
    B -->|Medium<br/>vehicle, tree| D["Medium Alert"]
    B -->|Low<br/>door| E["Info"]
    B -->|Skip<br/>wall| F["Ignore"]
    
    C --> G{"Position X?"}
    D --> G
    G -->|Left| H["Vibrate LEFT"]
    G -->|Right| I["Vibrate RIGHT"]
    G -->|Center| J["Vibrate CENTER"]
```

---

## Folder Structure

```
my_dataset/
├── dataset/                    # Raw images per class
│   ├── dinding/
│   ├── kendaraan_parkir/
│   ├── orang/
│   ├── pintu/
│   ├── pohon/
│   └── tiang/
├── model_training/
│   ├── train_yolo.ipynb       # Training notebook
│   ├── train_yolo.py          # Training script
│   └── balance_dataset.py     # Dataset processing
├── testing_app/
│   └── test_yolo_realtime.py  # Webcam testing script
└── yolo_output/
    └── yolo_training/weights/best.onnx
```

---

## Usage

### Training Workflow
```bash
# Via Notebook (recommended)
jupyter notebook model_training/train_yolo.ipynb

# Via Python Scripts
python model_training/balance_dataset.py
python model_training/train_yolo.py
```

### Realtime Testing
```bash
python testing_app/test_yolo_realtime.py
```

**Controls:**
- `Q` = Quit
- `D` = Toggle wall detection

---

## Performance

| Metric | Value |
|--------|-------|
| mAP50 | **93.4%** |
| mAP50-95 | 89.4% |
| Precision | > 95% |
| Recall | > 90% |
| Model Size | ~10 MB |
| Input Size | 96×96 |