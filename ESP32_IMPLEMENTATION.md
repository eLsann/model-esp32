# Panduan Implementasi ESP32-CAM

Dokumentasi lengkap untuk men-deploy model YOLOv5n ke ESP32-CAM.

---

## Daftar Isi

1. [Kebutuhan Hardware](#kebutuhan-hardware)
2. [Kebutuhan Software](#kebutuhan-software)
3. [Persiapan Model](#persiapan-model)
4. [Setup Arduino IDE](#setup-arduino-ide)
5. [Konversi Model](#konversi-model)
6. [Upload ke ESP32](#upload-ke-esp32)
7. [Troubleshooting](#troubleshooting)

---

## Kebutuhan Hardware

| Komponen | Spesifikasi | Keterangan |
|----------|-------------|------------|
| **ESP32-CAM** | AI-Thinker atau sejenisnya | Modul dengan kamera OV2640 |
| **FTDI Programmer** | USB to TTL (3.3V) | Untuk upload program |
| **Kabel Jumper** | Female-Female | Koneksi FTDI ke ESP32 |
| **Power Supply** | 5V / 2A | Opsional, bisa via USB |
| **MicroSD Card** | 4GB+ (Class 10) | Untuk menyimpan model |

### Wiring FTDI ke ESP32-CAM

```
FTDI          ESP32-CAM
────────────────────────
VCC   ──────► 5V
GND   ──────► GND
TX    ──────► U0R (GPIO3)
RX    ──────► U0T (GPIO1)
              
              IO0 ──► GND (saat upload)
```

> **Catatan:** Hubungkan IO0 ke GND hanya saat upload. Lepaskan setelah selesai.

---

## Kebutuhan Software

### PC / Laptop
- **Python 3.10+** - Untuk konversi model
- **Arduino IDE 2.x** - Untuk programming ESP32
- **ESP32 Board Package** - Di Arduino IDE

### Library Arduino
```
ESP32 Camera Driver (built-in)
TensorFlow Lite Micro (tflite-micro)
```

### Python Packages
```bash
pip install ultralytics onnx onnx2tf tensorflow
```

---

## Persiapan Model

### 1. Pastikan Model ONNX Tersedia

Setelah training, model tersimpan di:
```
yolo_output/yolo_training/weights/best.onnx
```

### 2. Verifikasi Ukuran Model

```python
import os
model_path = "yolo_output/yolo_training/weights/best.onnx"
size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"Model size: {size_mb:.2f} MB")
```

> **Target:** Model harus < 4MB untuk ESP32 (PSRAM)

---

## Konversi Model ke TFLite

ESP32 memerlukan format **TensorFlow Lite** dengan **INT8 Quantization**.

### Script Konversi

```python
# convert_to_tflite.py
import tensorflow as tf
import numpy as np

# Load ONNX dan konversi ke TF SavedModel dulu
# Gunakan onnx2tf atau tf-onnx

# Quantization untuk ESP32
def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 96, 96, 3).astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open('model_int8.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")
```

### Konversi ke C Array

```bash
xxd -i model_int8.tflite > model_data.h
```

---

## Setup Arduino IDE

### 1. Install ESP32 Board

1. Buka **File > Preferences**
2. Tambahkan URL di "Additional Board Manager URLs":
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Buka **Tools > Board > Board Manager**
4. Cari "esp32" dan install

### 2. Pilih Board

- **Board:** AI Thinker ESP32-CAM
- **Partition Scheme:** Huge APP (3MB No OTA)
- **PSRAM:** Enabled

### 3. Install TFLite Micro

Download dari: https://github.com/tensorflow/tflite-micro-arduino-examples

---

## Kode ESP32

### Struktur Project

```
esp32_inference/
├── esp32_inference.ino    # Main sketch
├── model_data.h           # Model dalam C array
├── camera_config.h        # Konfigurasi kamera
└── inference.h            # Helper inference
```

### Main Sketch

```cpp
// esp32_inference.ino
#include "esp_camera.h"
#include "model_data.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// Kamera pins (AI-Thinker)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// TFLite globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Class names
const char* class_names[] = {
    "dinding", "kendaraan_parkir", "orang", 
    "pintu", "pohon", "tiang"
};

void setup() {
    Serial.begin(115200);
    
    // Init camera
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_96X96;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x", err);
        return;
    }
    
    // Load model
    model = tflite::GetModel(g_model_data);
    
    // Setup interpreter
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    
    interpreter->AllocateTensors();
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.println("Setup complete!");
}

void loop() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return;
    }
    
    // Copy to input tensor
    memcpy(input->data.int8, fb->buf, input->bytes);
    
    // Run inference
    unsigned long start = millis();
    interpreter->Invoke();
    unsigned long duration = millis() - start;
    
    // Get results
    int8_t* output_data = output->data.int8;
    int max_idx = 0;
    int8_t max_val = output_data[0];
    
    for (int i = 1; i < 6; i++) {
        if (output_data[i] > max_val) {
            max_val = output_data[i];
            max_idx = i;
        }
    }
    
    Serial.printf("Detected: %s (conf: %d, time: %lu ms)\n", 
                  class_names[max_idx], max_val, duration);
    
    esp_camera_fb_return(fb);
    delay(100);
}
```

---

## Upload ke ESP32

### Langkah Upload

1. Hubungkan FTDI ke ESP32-CAM
2. Hubungkan **IO0 ke GND**
3. Tekan tombol **RESET** pada ESP32-CAM
4. Klik **Upload** di Arduino IDE
5. Tunggu proses upload selesai
6. **Lepaskan IO0 dari GND**
7. Tekan **RESET** lagi

### Verifikasi

Buka Serial Monitor (115200 baud):
```
Setup complete!
Detected: orang (conf: 127, time: 150 ms)
Detected: tiang (conf: 98, time: 148 ms)
...
```

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Upload gagal | Pastikan IO0 terhubung ke GND saat upload |
| Camera init failed | Periksa wiring dan power supply |
| Out of memory | Kurangi `kTensorArenaSize` atau gunakan model lebih kecil |
| Inference lambat | Pastikan menggunakan INT8 quantization |
| Model tidak load | Verifikasi format C array di `model_data.h` |

---

## Tips Optimasi

1. **Gunakan PSRAM** - Enable PSRAM untuk buffer lebih besar
2. **Reduce Frame Size** - Gunakan 96x96 untuk kecepatan
3. **INT8 Quantization** - Wajib untuk ESP32
4. **Skip Frames** - Proses setiap 2-3 frame untuk hemat resource
5. **Sleep Mode** - Gunakan deep sleep saat idle

---

## Referensi

- [ESP32-CAM Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
