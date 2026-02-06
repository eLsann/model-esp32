# 🦯 Smart Cane Object Detection - ESP32-CAM

Sistem deteksi objek berbasis **YOLOv5n** untuk ESP32-CAM sebagai alat bantu navigasi tunanetra.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLO](https://img.shields.io/badge/YOLO-v5n-green)
![ESP32](https://img.shields.io/badge/ESP32-CAM-orange)

---

## 📋 Fitur

- ✅ **6 Class Detection**: dinding, kendaraan_parkir, orang, pintu, pohon, tiang
- ✅ **Realtime Detection**: ~30 FPS pada PC, ~5 FPS pada ESP32
- ✅ **Auto-Annotation**: Anotasi otomatis dari struktur folder
- ✅ **Dataset Balancing**: 150 gambar/class dengan augmentasi
- ✅ **ONNX Export**: Model ~10MB untuk embedded deployment

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Training | Ultralytics YOLO, PyTorch |
| Inference | OpenCV DNN, ONNX |
| Target | ESP32-CAM |
| Language | Python 3.10+ |

---

## 📊 Pipeline Flow

```mermaid
flowchart LR
    A["📁 Dataset<br/>per Class"] --> B["📝 Auto<br/>Annotation"]
    B --> C["⚖️ Balancing<br/>150/class"]
    C --> D["🎯 Training<br/>YOLOv5n"]
    D --> E["📦 Export<br/>ONNX"]
    E --> F["🚀 Deploy<br/>ESP32"]
```

---

## 📊 Training Flow Detail

```mermaid
flowchart TD
    subgraph INPUT["📁 INPUT"]
        A[("Gambar per Folder")]
    end

    subgraph PROCESS["🔄 PROCESSING"]
        B["Auto-Annotation"]
        C{"Count >= 150?"}
        D["Undersample"]
        E["Oversample + Aug"]
        F["Split 85/15"]
    end

    subgraph TRAINING["🎯 TRAINING"]
        G["YOLOv5n 50 Epochs"]
        H["Evaluate mAP"]
        I["Export ONNX"]
    end

    A --> B --> C
    C -->|Ya| D --> F
    C -->|Tidak| E --> F
    F --> G --> H --> I
```

---

## 📊 Realtime Detection Flow

```mermaid
flowchart LR
    A["📷 Camera"] --> B["Resize 96x96"]
    B --> C["ONNX Inference"]
    C --> D{"Detected?"}
    D -->|Ya| E["🟢 ACTIVE<br/>Draw BBox"]
    D -->|Tidak| F["⚪ IDLE"]
```

---

## 📊 Feedback System (Tongkat)

```mermaid
flowchart TD
    A["Detection"] --> B{"Priority?"}
    B -->|🔴 High<br/>orang, tiang| C["Alert Kuat"]
    B -->|🟡 Medium<br/>kendaraan, pohon| D["Alert Sedang"]
    B -->|🟢 Low<br/>pintu| E["Info"]
    B -->|⚪ Skip<br/>dinding| F["Ignore"]
    
    C --> G{"Position X?"}
    D --> G
    G -->|Kiri| H["Vibrate LEFT"]
    G -->|Kanan| I["Vibrate RIGHT"]
    G -->|Tengah| J["Vibrate CENTER"]
```

---

## 📁 Struktur Folder

```
my_dataset/
├── dataset/                    # Gambar per class
│   ├── dinding/
│   ├── kendaraan_parkir/
│   ├── orang/
│   ├── pintu/
│   ├── pohon/
│   └── tiang/
├── model_training/
│   ├── train_yolo.ipynb       # Notebook training
│   ├── train_yolo.py
│   └── balance_dataset.py
├── testing_app/
│   └── test_yolo_realtime.py  # Webcam testing
└── yolo_output/
    └── yolo_training/weights/best.onnx
```

---

## 🚀 Cara Menjalankan

### Training
```bash
# Via Notebook (recommended)
jupyter notebook model_training/train_yolo.ipynb

# Via Python
python model_training/balance_dataset.py
python model_training/train_yolo.py
```

### Testing Realtime
```bash
python testing_app/test_yolo_realtime.py
```

**Kontrol:**
- `Q` = Quit
- `D` = Toggle deteksi dinding

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| mAP50 | **93.4%** |
| mAP50-95 | 89.4% |
| Precision | 95%+ |
| Recall | 90%+ |
| Model Size | ~10 MB |
| Input Size | 96×96 |