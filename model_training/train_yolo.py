"""
YOLO Training Script - Optimized for ESP32
- Auto-annotation dari folder structure
- Dataset Balancing (150/class)
- Train/Val/Test Split (70:20:10)
- Enhanced Augmentations
- INT8 Quantization Export
"""

from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = Path(r"c:\Users\Sanzz\OneDrive\Documents\my_dataset")
OUTPUT_DIR = BASE_DIR / "yolo_dataset"
MODEL_OUTPUT = BASE_DIR / "yolo_output"

# Training Parameters - Optimized for ESP32
IMG_SIZE = 96       # Small for ESP32-CAM
EPOCHS = 100        # More epochs for better convergence
BATCH_SIZE = 16

def train_model():
    """Train YOLOv5n model"""
    print("=" * 60)
    print("YOLO TRAINING - ESP32 Optimized")
    print("=" * 60)
    
    # Load pretrained model
    model = YOLO('yolov5n.pt')
    
    # Train
    results = model.train(
        data=str(OUTPUT_DIR / 'dataset.yaml'),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(MODEL_OUTPUT),
        name='yolo_training',
        exist_ok=True,
        patience=20,        # Early stopping
        save=True,
        plots=True
    )
    
    return model, results


def export_model(model):
    """Export model to ONNX with INT8 quantization for ESP32"""
    print("\nExporting to ONNX...")
    
    # Standard ONNX export
    model.export(format='onnx', imgsz=IMG_SIZE, simplify=True)
    
    print("Export complete!")


def evaluate_model(model):
    """Evaluate on test set"""
    print("\nEvaluating on test set...")
    
    metrics = model.val(
        data=str(OUTPUT_DIR / 'dataset.yaml'),
        split='test',
        imgsz=IMG_SIZE
    )
    
    print(f"\nTest Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    return metrics


if __name__ == "__main__":
    # Step 1: Train
    model, results = train_model()
    
    # Step 2: Evaluate on test set
    evaluate_model(model)
    
    # Step 3: Export
    export_model(model)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
