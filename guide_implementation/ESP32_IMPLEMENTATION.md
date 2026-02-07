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

---

# BAGIAN 2: Implementasi ESP32-S3 (Recommended)

ESP32-S3 adalah upgrade signifikan dari ESP32-CAM dengan performa AI yang lebih baik.

---

## Perbandingan ESP32-CAM vs ESP32-S3

| Fitur | ESP32-CAM | ESP32-S3 |
|-------|-----------|----------|
| CPU | Dual-core 240MHz | Dual-core 240MHz |
| RAM | 520KB + 4MB PSRAM | 512KB + 8MB PSRAM |
| Flash | 4MB | 8-16MB |
| AI Acceleration | ❌ Tidak ada | ✅ Vector Instructions |
| USB | ❌ Tidak ada | ✅ USB OTG Native |
| WiFi | 802.11 b/g/n | 802.11 b/g/n |
| Bluetooth | BT 4.2 | BLE 5.0 |
| Harga | ~50-80rb | ~100-200rb |
| **Rekomendasi** | Budget/Prototype | **Production** |

---

## Kebutuhan Hardware ESP32-S3

### Modul yang Direkomendasikan

| Modul | Spesifikasi | Harga (IDR) | Link |
|-------|-------------|-------------|------|
| **ESP32-S3-DevKitC-1** | 8MB PSRAM, 16MB Flash | 150-200rb | Espressif Official |
| **XIAO ESP32S3 Sense** | Camera included, compact | 200-250rb | Seeed Studio |
| **Freenove ESP32-S3-WROOM** | Camera module included | 180-220rb | Freenove |

### Komponen Tambahan

| Komponen | Spesifikasi | Keterangan |
|----------|-------------|------------|
| **Kamera OV2640** | 2MP, SCCB interface | Jika modul tanpa kamera |
| **Motor Vibration** | 3V DC | Untuk haptic feedback |
| **Buzzer** | 5V Active | Untuk audio feedback |
| **Baterai LiPo** | 3.7V 1000mAh | Power portable |
| **USB-C Cable** | Data + Power | Upload tanpa FTDI |

---

## Wiring ESP32-S3 + Camera

### ESP32-S3-DevKitC dengan OV2640

```
ESP32-S3          OV2640 Camera
─────────────────────────────────
3.3V     ──────►  VCC
GND      ──────►  GND
GPIO10   ──────►  SIOD (SDA)
GPIO11   ──────►  SIOC (SCL)
GPIO12   ──────►  VSYNC
GPIO13   ──────►  HREF
GPIO14   ──────►  PCLK
GPIO15   ──────►  XCLK
GPIO16   ──────►  D7
GPIO17   ──────►  D6
GPIO18   ──────►  D5
GPIO8    ──────►  D4
GPIO9    ──────►  D3
GPIO3    ──────►  D2
GPIO46   ──────►  D1
GPIO4    ──────►  D0
```

### Wiring Haptic Feedback (Motor Getar)

```
ESP32-S3          Motor Driver (L298N Mini)
────────────────────────────────────────────
GPIO5    ──────►  IN1 (Motor Kiri)
GPIO6    ──────►  IN2 (Motor Kanan)
GPIO7    ──────►  IN3 (Motor Tengah)
3.3V     ──────►  VCC
GND      ──────►  GND
```

---

## Setup Arduino IDE untuk ESP32-S3

### 1. Install Board ESP32-S3

Board Manager URL sudah sama:
```
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```

### 2. Pilih Board ESP32-S3

- **Board:** ESP32S3 Dev Module
- **USB CDC On Boot:** Enabled
- **USB Mode:** USB-OTG (TinyUSB)
- **PSRAM:** OPI PSRAM
- **Flash Size:** 16MB (atau sesuai modul)
- **Partition Scheme:** Huge APP (3MB No OTA/1MB SPIFFS)

### 3. Upload Mode

ESP32-S3 mendukung **USB Native**, tidak perlu FTDI!

1. Hubungkan USB-C ke komputer
2. Tekan dan tahan **BOOT** button
3. Tekan **RESET** button (sambil tahan BOOT)
4. Lepaskan kedua tombol
5. Upload dari Arduino IDE

---

## Kode ESP32-S3 (Optimized)

### Struktur Project

```
esp32s3_inference/
├── esp32s3_inference.ino      # Main sketch (optimized for S3)
├── model_data.h               # Model dalam C array
├── camera_pins.h              # Pin configuration
├── haptic_feedback.h          # Motor control
└── config.h                   # Global settings
```

### camera_pins.h (ESP32-S3)

```cpp
// camera_pins.h - Pin definitions for ESP32-S3 + OV2640

#ifndef CAMERA_PINS_H
#define CAMERA_PINS_H

// ESP32-S3-DevKitC-1 with OV2640
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    15
#define SIOD_GPIO_NUM    10
#define SIOC_GPIO_NUM    11

#define Y9_GPIO_NUM      16
#define Y8_GPIO_NUM      17
#define Y7_GPIO_NUM      18
#define Y6_GPIO_NUM      8
#define Y5_GPIO_NUM      9
#define Y4_GPIO_NUM      3
#define Y3_GPIO_NUM      46
#define Y2_GPIO_NUM      4

#define VSYNC_GPIO_NUM   12
#define HREF_GPIO_NUM    13
#define PCLK_GPIO_NUM    14

// Alternative: XIAO ESP32S3 Sense (built-in camera)
// Uncomment if using XIAO
/*
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    10
#define SIOD_GPIO_NUM    40
#define SIOC_GPIO_NUM    39
#define Y9_GPIO_NUM      48
#define Y8_GPIO_NUM      11
#define Y7_GPIO_NUM      12
#define Y6_GPIO_NUM      14
#define Y5_GPIO_NUM      16
#define Y4_GPIO_NUM      18
#define Y3_GPIO_NUM      17
#define Y2_GPIO_NUM      15
#define VSYNC_GPIO_NUM   38
#define HREF_GPIO_NUM    47
#define PCLK_GPIO_NUM    13
*/

#endif
```

### haptic_feedback.h

```cpp
// haptic_feedback.h - Haptic motor control

#ifndef HAPTIC_FEEDBACK_H
#define HAPTIC_FEEDBACK_H

#define MOTOR_LEFT   5
#define MOTOR_CENTER 6
#define MOTOR_RIGHT  7

void haptic_init() {
    pinMode(MOTOR_LEFT, OUTPUT);
    pinMode(MOTOR_CENTER, OUTPUT);
    pinMode(MOTOR_RIGHT, OUTPUT);
    
    // All motors off initially
    digitalWrite(MOTOR_LEFT, LOW);
    digitalWrite(MOTOR_CENTER, LOW);
    digitalWrite(MOTOR_RIGHT, LOW);
}

void haptic_alert(int position, int intensity) {
    // position: 0=left, 1=center, 2=right
    // intensity: 0-255 (PWM)
    
    int pin;
    switch(position) {
        case 0: pin = MOTOR_LEFT; break;
        case 1: pin = MOTOR_CENTER; break;
        case 2: pin = MOTOR_RIGHT; break;
        default: return;
    }
    
    // Short vibration pulse
    analogWrite(pin, intensity);
    delay(100);
    analogWrite(pin, 0);
}

void haptic_pattern(int priority) {
    // priority: 0=low, 1=medium, 2=high
    switch(priority) {
        case 2: // High - double pulse
            haptic_alert(1, 255);
            delay(50);
            haptic_alert(1, 255);
            break;
        case 1: // Medium - single pulse
            haptic_alert(1, 180);
            break;
        case 0: // Low - gentle
            haptic_alert(1, 100);
            break;
    }
}

#endif
```

### Main Sketch (esp32s3_inference.ino)

```cpp
// esp32s3_inference.ino - Optimized for ESP32-S3

#include "esp_camera.h"
#include "model_data.h"
#include "camera_pins.h"
#include "haptic_feedback.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// ESP32-S3 can use larger tensor arena (8MB PSRAM)
constexpr int kTensorArenaSize = 200 * 1024;  // 200KB for S3
uint8_t* tensor_arena;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Class definitions
const char* class_names[] = {
    "dinding", "kendaraan_parkir", "orang", 
    "pintu", "pohon", "tiang"
};

// Priority levels: 0=skip, 1=low, 2=medium, 3=high
const int class_priority[] = {0, 2, 3, 1, 2, 3};

void setup() {
    Serial.begin(115200);
    Serial.println("ESP32-S3 Object Detection");
    Serial.println("==========================");
    
    // Allocate tensor arena in PSRAM (ESP32-S3 advantage)
    tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
    if (!tensor_arena) {
        Serial.println("Failed to allocate PSRAM!");
        return;
    }
    Serial.println("PSRAM allocated successfully");
    
    // Initialize haptic feedback
    haptic_init();
    Serial.println("Haptic motors initialized");
    
    // Camera configuration
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
    config.fb_count = 2;  // Double buffering for S3
    config.fb_location = CAMERA_FB_IN_PSRAM;  // Use PSRAM
    config.grab_mode = CAMERA_GRAB_LATEST;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return;
    }
    Serial.println("Camera initialized");
    
    // Load TFLite model
    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        return;
    }
    
    // Setup interpreter
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors!");
        return;
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.println("\n=== READY ===");
    Serial.printf("Input shape: [%d, %d, %d]\n", 
                  input->dims->data[1], 
                  input->dims->data[2], 
                  input->dims->data[3]);
}

void loop() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        delay(100);
        return;
    }
    
    // Preprocess and copy to input tensor
    for (int i = 0; i < input->bytes; i++) {
        input->data.int8[i] = (int8_t)(fb->buf[i] - 128);
    }
    
    // Run inference
    unsigned long start = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long duration = micros() - start;
    
    if (invoke_status != kTfLiteOk) {
        Serial.println("Inference failed!");
        esp_camera_fb_return(fb);
        return;
    }
    
    // Parse output
    int8_t* output_data = output->data.int8;
    int detected_class = -1;
    int8_t max_conf = -128;
    
    for (int i = 0; i < 6; i++) {
        if (output_data[i] > max_conf && class_priority[i] > 0) {
            max_conf = output_data[i];
            detected_class = i;
        }
    }
    
    // Output results
    if (detected_class >= 0 && max_conf > 0) {
        int priority = class_priority[detected_class];
        
        Serial.printf("[%s] conf:%d prio:%d time:%.1fms\n",
                      class_names[detected_class],
                      max_conf,
                      priority,
                      duration / 1000.0);
        
        // Haptic feedback based on priority
        if (priority >= 2) {
            haptic_pattern(priority - 1);
        }
    }
    
    esp_camera_fb_return(fb);
    delay(50);  // ~20 FPS target
}
```

---

## Performa ESP32-S3

| Metric | ESP32-CAM | ESP32-S3 |
|--------|-----------|----------|
| Inference Time | ~200ms | ~80-100ms |
| FPS | ~5 FPS | ~10-15 FPS |
| Model Max Size | 4MB | 8MB |
| PSRAM Usage | 100KB | 200KB+ |
| Power (Active) | ~200mA | ~250mA |

---

## Troubleshooting ESP32-S3

| Masalah | Solusi |
|---------|--------|
| USB tidak terdeteksi | Install driver CP210x atau tekan BOOT+RESET |
| PSRAM allocation failed | Pastikan board support PSRAM dan di-enable |
| Camera init failed | Cek wiring dan pastikan pin sesuai |
| Inference terlalu lambat | Gunakan model INT8, reduce input size |
| Modul panas | Tambahkan heatsink atau kurangi FPS |

---

## Tips Produksi

1. **Gunakan XIAO ESP32S3 Sense** - Compact, sudah include camera
2. **Custom PCB** - Untuk form factor tongkat yang lebih baik
3. **LiPo Battery** - 3.7V 2000mAh untuk ~4-6 jam penggunaan
4. **3D Printed Case** - Desain custom untuk mounting di tongkat
5. **OTA Update** - Implementasi WiFi OTA untuk update firmware

---

## Referensi ESP32-S3

- [ESP32-S3 Datasheet](https://www.espressif.com/sites/default/files/documentation/esp32-s3_datasheet_en.pdf)
- [ESP32-S3 Technical Reference](https://www.espressif.com/sites/default/files/documentation/esp32-s3_technical_reference_manual_en.pdf)
- [XIAO ESP32S3 Wiki](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/)

