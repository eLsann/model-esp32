# Deteksi Objek Tongkat Pintar - ESP32-CAM

Sistem deteksi objek berbasis **YOLOv5n** untuk ESP32-CAM, dirancang sebagai alat bantu navigasi bagi penyandang tunanetra.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLO](https://img.shields.io/badge/YOLO-v5n-green)
![ESP32](https://img.shields.io/badge/ESP32-CAM-orange)

---

## Fitur

- **Deteksi 6 Kelas**: dinding, kendaraan_parkir (motor/mobil), orang, pintu, pohon, tiang
- **Deteksi Realtime**: ~30 FPS di PC, ~5 FPS di ESP32
- **Anotasi Otomatis**: Anotasi otomatis berdasarkan struktur folder
- **Penyeimbangan Dataset**: 150 gambar/kelas dengan augmentasi
- **Ekspor ONNX**: Model yang dioptimalkan ~10MB untuk deployment di perangkat embedded

---

## Teknologi yang Digunakan

### 🖥️ Hardware
- **ESP32-CAM**: Mikrokontroler utama untuk deployment dan inferensi (AI-on-Edge).
- **PC / GPU**: Digunakan untuk pelatihan model dan pemrosesan dataset (disarankan menggunakan GPU NVIDIA agar lebih cepat).
- **Modul Kamera**: OV2640 (modul standar ESP32-CAM).

### 🧠 AI & Machine Learning
- **YOLOv5n (Ultralytics)**: Versi nano dari arsitektur YOLOv5, dioptimalkan untuk kecepatan dan sumber daya rendah.
- **PyTorch**: Framework deep learning yang digunakan sebagai backend untuk pelatihan model.
- **ONNX (Open Neural Network Exchange)**: Format untuk interoperabilitas dan deployment model ke perangkat edge.

### 🛠️ Software & Libraries
- **Python 3.10+**: Bahasa pemrograman utama untuk skripting dan pelatihan.
- **OpenCV (cv2)**: Pemrosesan citra, resizing, dan visualisasi realtime.
- **NumPy & Pandas**: Manipulasi data dan pengelolaan dataset.
- **Matplotlib & Seaborn**: Visualisasi distribusi dataset dan metrik pelatihan.
- **Jupyter Notebook**: Lingkungan interaktif untuk alur kerja pelatihan.
- **Git**: Sistem kontrol versi.

---

## Alur Pipeline

```mermaid
flowchart LR
    A["Dataset<br/>per Kelas"] --> B["Anotasi<br/>Otomatis"]
    B --> C["Balancing<br/>150/kelas"]
    C --> D["Pelatihan<br/>YOLOv5n"]
    D --> E["Ekspor<br/>ONNX"]
    E --> F["Deploy<br/>ESP32"]
```

---

## Detail Alur Pelatihan

```mermaid
flowchart TD
    subgraph INPUT["INPUT"]
        A[("Gambar per Folder")]
    end

    subgraph PROCESS["PEMROSESAN"]
        B["Anotasi Otomatis"]
        C{"Jumlah >= 150?"}
        D["Undersample"]
        E["Oversample + Aug"]
        F["Split 70/20/10"]
    end

    subgraph TRAINING["PELATIHAN"]
        G["YOLOv5n 100 Epoch"]
        H["Evaluasi Test Set"]
        I["Ekspor ONNX"]
    end

    A --> B --> C
    C -->|Ya| D --> F
    C -->|Tidak| E --> F
    F --> G --> H --> I
```

---

## Alur Deteksi Realtime

```mermaid
flowchart LR
    A["Kamera"] --> B["Resize 96x96"]
    B --> C["Inferensi ONNX"]
    C --> D{"Terdeteksi?"}
    D -->|Ya| E["AKTIF<br/>Gambar BBox"]
    D -->|Tidak| F["IDLE"]
```

---

## Sistem Umpan Balik (Tongkat Pintar)

```mermaid
flowchart TD
    A["Deteksi"] --> B{"Prioritas?"}
    B -->|Tinggi<br/>orang, tiang| C["Peringatan Kuat"]
    B -->|Sedang<br/>kendaraan, pohon| D["Peringatan Sedang"]
    B -->|Rendah<br/>pintu| E["Info"]
    B -->|Abaikan<br/>dinding| F["Abaikan"]
    
    C --> G{"Posisi X?"}
    D --> G
    G -->|Kiri| H["Getar KIRI"]
    G -->|Kanan| I["Getar KANAN"]
    G -->|Tengah| J["Getar TENGAH"]
```

---

## Struktur Folder

```
my_dataset/
├── dataset/                    # Gambar mentah per kelas
│   ├── dinding/
│   ├── kendaraan_parkir/
│   ├── orang/
│   ├── pintu/
│   ├── pohon/
│   └── tiang/
├── model_training/
│   ├── train_yolo.ipynb       # Notebook pelatihan
│   ├── train_yolo.py          # Script pelatihan
│   └── balance_dataset.py     # Pemrosesan dataset
├── testing_app/
│   └── test_yolo_realtime.py  # Script testing webcam
└── yolo_output/
    └── yolo_training/weights/best.onnx
```

---

## Cara Penggunaan

### Alur Pelatihan
```bash
# Melalui Notebook (disarankan)
jupyter notebook model_training/train_yolo.ipynb

# Melalui Script Python
python model_training/balance_dataset.py
python model_training/train_yolo.py
```

### Testing Realtime
```bash
python testing_app/test_yolo_realtime.py
```

**Kontrol:**
- `Q` = Keluar
- `D` = Toggle deteksi dinding

---

## Performa

| Metrik | Nilai |
|--------|-------|
| mAP50 | **93.4%** |
| mAP50-95 | 89.4% |
| Presisi | > 95% |
| Recall | > 90% |
| Ukuran Model | ~10 MB |
| Ukuran Input | 96×96 |