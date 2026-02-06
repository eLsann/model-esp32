"""
Dataset Balancing & YOLO Format Conversion - Optimized for ESP32
- Auto-annotation dari folder structure
- Dataset Balancing (150/class)
- Train/Val/Test Split (70:20:10)
- Enhanced Augmentations (flip, brightness, rotation, blur)
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

# Configuration
BASE_DIR = Path(r"c:\Users\Sanzz\OneDrive\Documents\my_dataset")
DATASET_SOURCE = BASE_DIR / "dataset"
ANNOTATIONS_FILE = BASE_DIR / "annotations" / "annotations.json"
OUTPUT_DIR = BASE_DIR / "yolo_dataset"

# Training Parameters - Optimized for ESP32
TARGET_COUNT_PER_CLASS = 150
TRAIN_RATIO = 0.70  # 70% train
VAL_RATIO = 0.20    # 20% validation
TEST_RATIO = 0.10   # 10% test

CLASS_NAMES = ['dinding', 'kendaraan_parkir', 'orang', 'pintu', 'pohon', 'tiang']


def create_auto_annotations():
    """Create annotations from folder structure (center-based 60%)"""
    annotations = {}
    
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_folder = DATASET_SOURCE / class_name
        if not class_folder.exists():
            print(f"Warning: {class_folder} not found")
            continue
        
        count = 0
        for img_file in class_folder.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    
                    annotations[str(img_file)] = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'bbox': [0.2, 0.2, 0.8, 0.8],  # Center 60%
                        'width': w,
                        'height': h
                    }
                    count += 1
                except:
                    pass
        print(f"  {class_name}: {count} images")
    
    # Save annotations
    ann_dir = BASE_DIR / "annotations"
    ann_dir.mkdir(exist_ok=True)
    with open(ann_dir / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\nTotal: {len(annotations)} annotations")
    return annotations


def load_annotations():
    """Load existing annotations or create new"""
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, 'r') as f:
            return json.load(f)
    else:
        return create_auto_annotations()


def convert_bbox_to_yolo(bbox):
    """Convert [x_min, y_min, x_max, y_max] to YOLO format [cx, cy, w, h]"""
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    # Clamp values
    cx = max(0.001, min(0.999, cx))
    cy = max(0.001, min(0.999, cy))
    w = max(0.01, min(1.0, w))
    h = max(0.01, min(1.0, h))
    return cx, cy, w, h


def augment_image(img, aug_type):
    """Enhanced augmentations for better generalization"""
    if aug_type == 0:  # Horizontal flip
        return cv2.flip(img, 1), 'hflip', True
    elif aug_type == 1:  # Brightness increase
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.2, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), 'bright', False
    elif aug_type == 2:  # Brightness decrease
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 0.8, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), 'dark', False
    elif aug_type == 3:  # Slight rotation (±10°)
        angle = random.uniform(-10, 10)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h)), 'rot', False
    elif aug_type == 4:  # Gaussian blur
        return cv2.GaussianBlur(img, (3, 3), 0), 'blur', False
    else:  # Original
        return img, 'orig', False


def flip_bbox(bbox):
    """Flip bbox horizontally"""
    return [1 - bbox[2], bbox[1], 1 - bbox[0], bbox[3]]


def get_split(rand_val):
    """Determine split based on random value"""
    if rand_val < TRAIN_RATIO:
        return 'train'
    elif rand_val < TRAIN_RATIO + VAL_RATIO:
        return 'val'
    else:
        return 'test'


def process_dataset():
    """Main processing with train/val/test split"""
    print("=" * 60)
    print("DATASET BALANCING & YOLO CONVERSION (ESP32 Optimized)")
    print("=" * 60)
    
    # Load annotations
    print("\n1. Loading annotations...")
    annotations = load_annotations()
    print(f"   Total: {len(annotations)}")
    
    # Group by class
    class_items = defaultdict(list)
    for img_path, ann in annotations.items():
        class_items[ann['class_id']].append({
            'path': img_path,
            'bbox': ann['bbox'],
            'class_id': ann['class_id'],
            'class_name': ann['class_name']
        })
    
    print("\n   Original distribution:")
    for cls_id in sorted(class_items.keys()):
        print(f"   - {CLASS_NAMES[cls_id]}: {len(class_items[cls_id])}")
    
    # Balance dataset
    print(f"\n2. Balancing (target: {TARGET_COUNT_PER_CLASS}/class)...")
    balanced = {}
    
    for class_id, items in class_items.items():
        current = len(items)
        
        if current >= TARGET_COUNT_PER_CLASS:
            # Undersample
            balanced[class_id] = random.sample(items, TARGET_COUNT_PER_CLASS)
            action = "undersample"
        else:
            # Oversample with augmentation
            balanced[class_id] = items.copy()
            need = TARGET_COUNT_PER_CLASS - current
            for i in range(need):
                item = random.choice(items).copy()
                item['augment'] = i % 5  # 5 augmentation types
                item['aug_id'] = i
                balanced[class_id].append(item)
            action = "oversample"
        
        print(f"   - {CLASS_NAMES[class_id]}: {current} -> {len(balanced[class_id])} ({action})")
    
    # Create directory structure (train/val/test)
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Process and save images
    print("\n3. Processing images...")
    total = 0
    stats = {'train': defaultdict(int), 'val': defaultdict(int), 'test': defaultdict(int)}
    
    for class_id, items in balanced.items():
        for item in items:
            try:
                img = cv2.imread(item['path'])
                if img is None:
                    continue
                
                bbox = item['bbox']
                aug_suffix = ''
                need_flip_bbox = False
                
                # Apply augmentation if needed
                if 'augment' in item:
                    img, aug_name, need_flip_bbox = augment_image(img, item['augment'])
                    aug_suffix = f"_aug{item['aug_id']}_{aug_name}"
                    if need_flip_bbox:
                        bbox = flip_bbox(bbox)
                
                # Convert bbox to YOLO format
                cx, cy, bw, bh = convert_bbox_to_yolo(bbox)
                
                # Determine split
                split = get_split(random.random())
                
                # Generate filename
                base_name = Path(item['path']).stem
                new_name = f"{CLASS_NAMES[class_id]}_{base_name}{aug_suffix}"
                
                # Save image
                cv2.imwrite(str(OUTPUT_DIR / 'images' / split / f"{new_name}.jpg"), img)
                
                # Save label
                with open(OUTPUT_DIR / 'labels' / split / f"{new_name}.txt", 'w') as f:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
                total += 1
                stats[split][class_id] += 1
            except Exception as e:
                pass
        
        print(f"   Processed {CLASS_NAMES[class_id]}")
    
    # Print split statistics
    print("\n4. Split Statistics:")
    for split in ['train', 'val', 'test']:
        count = sum(stats[split].values())
        print(f"   - {split}: {count} images")
    
    # Create dataset.yaml
    yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(OUTPUT_DIR / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    # Save labels
    with open(OUTPUT_DIR / 'yolo_labels.txt', 'w') as f:
        f.write('\n'.join(CLASS_NAMES))
    
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"Total: {total} images")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_dataset()
