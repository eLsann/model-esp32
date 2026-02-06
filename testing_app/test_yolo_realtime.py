"""
YOLO-Fastest (YOLOv5n) Realtime Testing
Improved version with better object tracking and filtering
"""

import cv2
import numpy as np
import os

# Configuration
BASE_DIR = r"c:\Users\Sanzz\OneDrive\Documents\my_dataset"
MODEL_DIR = os.path.join(BASE_DIR, "yolo_output")  # Original model (mAP 93.4%)
LABELS_PATH = os.path.join(MODEL_DIR, "yolo_labels.txt")
IMG_SIZE = 96
CONF_THRESHOLD = 0.15  # Very low to catch more objects
IOU_THRESHOLD = 0.3    # Lower NMS to allow more overlapping boxes
MAX_BOX_AREA_RATIO = 0.85  # Allow larger boxes

# Classes to skip (e.g., "dinding" often covers entire frame)
SKIP_CLASSES = []  # Set to [0] to skip "dinding", empty to show all

# Fixed colors for each class (BGR format)
COLORS = [
    (255, 100, 100),   # dinding - Light Blue
    (100, 255, 100),   # kendaraan_parkir - Light Green
    (100, 100, 255),   # orang - Light Red
    (255, 255, 100),   # pintu - Cyan
    (255, 100, 255),   # pohon - Magenta
    (100, 255, 255),   # tiang - Yellow
]

def load_labels(path):
    if not os.path.exists(path):
        return ['dinding', 'kendaraan_parkir', 'orang', 'pintu', 'pohon', 'tiang']
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def find_model():
    """Search for ONNX model in yolo_output folder"""
    for root, dirs, files in os.walk(MODEL_DIR):
        for f in files:
            if f.endswith(".onnx"):
                return os.path.join(root, f), "onnx"
    return None, None

def main():
    print("=" * 50)
    print("YOLO Object Detection - Improved Version")
    print("=" * 50)
    
    # Load labels
    labels = load_labels(LABELS_PATH)
    print(f"Classes: {labels}")
    
    # Find model
    model_path, model_type = find_model()
    if not model_path:
        print("ERROR: No ONNX model found!")
        return
    
    print(f"Model: {os.path.basename(model_path)}")
    
    # Load model
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return
    
    frame_count = 0
    
    print("\nControls:")
    print("  Q - Quit")
    print("  D - Toggle 'dinding' detection")
    print("-" * 50)
    
    show_dinding = True  # Toggle for dinding class
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h_orig, w_orig = frame.shape[:2]
        frame_area = h_orig * w_orig
        
        # Preprocess
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (IMG_SIZE, IMG_SIZE), swapRB=True, crop=False)
        
        # Inference
        net.setInput(blob)
        outputs = net.forward()
        
        # Parse output [1, 10, 189] -> transpose to [189, 10]
        output_data = outputs[0]
        if output_data.shape[0] < output_data.shape[1]:
            output_data = output_data.T
        
        # Extract boxes and scores
        boxes = output_data[:, :4]
        class_scores = output_data[:, 4:]
        
        confidences = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Filter by confidence
        mask = confidences > CONF_THRESHOLD
        filtered_boxes = boxes[mask]
        filtered_scores = confidences[mask]
        filtered_class_ids = class_ids[mask]
        
        # Prepare detections
        boxes_nms = []
        scores_nms = []
        class_ids_nms = []
        
        for i in range(len(filtered_scores)):
            cx, cy, bw, bh = filtered_boxes[i]
            conf = filtered_scores[i]
            cls_id = int(filtered_class_ids[i])
            
            # Skip certain classes if configured
            if cls_id == 0 and not show_dinding:  # Skip dinding
                continue
            
            # Convert normalized to pixel coords
            center_x = cx * w_orig
            center_y = cy * h_orig
            width = bw * w_orig
            height = bh * h_orig
            
            # Filter out boxes that are too large (likely full-frame detections)
            box_area = width * height
            if box_area > frame_area * MAX_BOX_AREA_RATIO:
                continue
            
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            
            # Clamp to frame bounds
            left = max(0, left)
            top = max(0, top)
            width = min(int(width), w_orig - left)
            height = min(int(height), h_orig - top)
            
            boxes_nms.append([left, top, width, height])
            scores_nms.append(float(conf))
            class_ids_nms.append(cls_id)
        
        # Apply NMS
        detected_classes = []
        if len(boxes_nms) > 0:
            indices = cv2.dnn.NMSBoxes(boxes_nms, scores_nms, CONF_THRESHOLD, IOU_THRESHOLD)
            
            for idx in indices:
                x, y, w, h = boxes_nms[idx]
                conf = scores_nms[idx]
                cls_id = class_ids_nms[idx]
                
                if cls_id >= len(labels):
                    cls_id = cls_id % len(labels)
                
                label_name = labels[cls_id]
                detected_classes.append(f"{label_name}:{conf:.2f}")
                
                color = COLORS[cls_id % len(COLORS)]
                
                # Draw thick bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Draw label with background
                label = f"{label_name} {conf:.0%}"
                font_scale = 0.6
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Label background
                cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, -1)
                cv2.putText(frame, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                
                # Draw center point
                cx_draw = x + w // 2
                cy_draw = y + h // 2
                cv2.circle(frame, (cx_draw, cy_draw), 5, color, -1)
        
        # Determine status
        if len(detected_classes) > 0:
            status_text = "ACTIVE"
            status_color = (0, 255, 0)  # Green
        else:
            status_text = "IDLE"
            status_color = (128, 128, 128)  # Gray
        
        # Draw status bar
        cv2.rectangle(frame, (0, 0), (w_orig, 35), (30, 30, 30), -1)
        
        # Status indicator
        cv2.circle(frame, (20, 18), 8, status_color, -1)
        cv2.putText(frame, status_text, (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Detection count
        det_text = f"Objects: {len(detected_classes)}"
        cv2.putText(frame, det_text, (130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Dinding toggle status
        dinding_text = f"D: {'ON' if show_dinding else 'OFF'}"
        cv2.putText(frame, dinding_text, (w_orig - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Debug output
        if frame_count % 60 == 0:
            if detected_classes:
                print(f"Frame {frame_count}: [ACTIVE] {', '.join(detected_classes)}")
            else:
                print(f"Frame {frame_count}: [IDLE] No objects detected")
        
        cv2.imshow("YOLO Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_dinding = not show_dinding
            print(f"Dinding detection: {'ON' if show_dinding else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")

if __name__ == "__main__":
    main()
