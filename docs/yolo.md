# YOLO Detection Documentation

## 1. What is YOLO?

**YOLO** stands for **You Only Look Once**. It is a family of real-time object
detection algorithms that treat detection as a **single regression problem**
rather than a two-stage pipeline.

Traditional detection pipelines (e.g. R-CNN) work in two stages:
1. Generate candidate regions where objects might be (region proposals).
2. Classify each region separately.

YOLO does both in a **single forward pass** through a neural network:
- The image is divided into an S × S grid.
- Each grid cell simultaneously predicts bounding boxes AND class probabilities.
- One inference pass → all detections.

This "look only once" design makes YOLO extremely fast — capable of running at
30+ frames per second on a modern GPU.

---

## 2. Why YOLO for Real-Time Pothole Detection?

| Requirement | Why YOLO satisfies it |
|---|---|
| **Real-time speed** | Single forward pass; 30–100+ FPS on GPU, ~10 FPS on CPU |
| **Sufficient accuracy** | YOLOv8 achieves state-of-the-art mAP on COCO benchmark |
| **Ease of training** | Ultralytics CLI makes fine-tuning straightforward |
| **Bounding box output** | Directly outputs (x, y, w, h) + confidence — exactly what we need |
| **Edge deployment** | YOLOv8 models can be exported to ONNX/TensorRT for embedded devices |
| **Single-class simplicity** | Works well even with a single target class (D40 potholes) |

**Alternatives considered:**
- **Faster R-CNN**: More accurate, but ~5× slower — unsuitable for video streams.
- **SSD (Single Shot Detector)**: Faster than Faster R-CNN but less accurate than YOLOv8.
- **EfficientDet**: Good accuracy but more complex to fine-tune.

---

## 3. YOLOv8 Architecture Overview

YOLOv8 (released by Ultralytics in 2023) follows a three-stage architecture:

```
Input Image (416×416×3)
         │
         ▼
┌──────────────────────────────┐
│  BACKBONE (CSPDarknet)       │
│  Extracts multi-scale        │
│  feature maps                │
│  Output: [52×52, 26×26,      │
│           13×13] feature maps│
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  NECK (PAN-FPN)              │
│  Path Aggregation Network +  │
│  Feature Pyramid Network     │
│  Combines features from      │
│  all scales                  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  HEAD (Decoupled Head)       │
│  For each scale outputs:     │
│  • Bounding box (x,y,w,h)    │
│  • Objectness score          │
│  • Class probabilities       │
└──────────────────────────────┘
               │
               ▼
       NMS Post-processing
       (Non-Maximum Suppression)
               │
               ▼
    Final Detections (bbox + conf)
```

**Key YOLOv8 improvements over YOLOv5:**
- Decoupled head (separate branches for box and class prediction).
- Anchor-free detection (no pre-defined anchor boxes to tune).
- Better small-object detection via improved FPN design.

---

## 4. Training Process

### 4.1 Dataset Preparation

1. Download RDD2022 India subset.
2. Filter only D40 (pothole) annotations.
3. Convert XML → YOLO `.txt` format (see preprocessing docs).
4. Split 70% train / 20% val / 10% test.
5. Create `dataset.yaml`:

```yaml
path: ./data
train: train/images
val:   val/images
test:  test/images
nc: 1
names: ['pothole']
```

### 4.2 Training Command

```bash
yolo detect train \
  model=yolov8n.pt \
  data=dataset.yaml \
  epochs=50 \
  imgsz=416 \
  batch=16 \
  lr0=0.01 \
  patience=10 \
  project=runs/detect \
  name=pothole_v1
```

| Parameter | Value | Meaning |
|---|---|---|
| `model` | `yolov8n.pt` | Start from COCO pre-trained nano weights (transfer learning) |
| `epochs` | 50 | Maximum training iterations over full dataset |
| `imgsz` | 416 | Input image size (width = height) |
| `batch` | 16 | Images per gradient update |
| `lr0` | 0.01 | Initial learning rate |
| `patience` | 10 | Early stopping if val mAP doesn't improve for 10 epochs |

### 4.3 Transfer Learning

We start from `yolov8n.pt` (pre-trained on COCO's 80 classes) rather than
random weights. The backbone has already learned general visual features
(edges, textures, shapes). Fine-tuning adapts the head to detect potholes.

This dramatically reduces training time and the amount of data required.

### 4.4 Output Artifacts

After training, the best model weights are saved at:
```
runs/detect/pothole_v1/weights/best.pt
```

This file is loaded for all inference operations.

---

## 5. Inference Process

```python
from ultralytics import YOLO

model = YOLO("runs/detect/pothole_v1/weights/best.pt")
results = model.predict(source="road_image.jpg", conf=0.5, imgsz=416)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]   # pixel coordinates
        confidence      = box.conf[0]   # confidence score
        class_id        = box.cls[0]    # always 0 (pothole)
```

**Steps during inference:**
1. Image is resized to 416 × 416 with letterboxing.
2. Pixel values normalised to [0, 1].
3. Forward pass through the network → raw predictions.
4. **NMS** (Non-Maximum Suppression) removes overlapping duplicate boxes.
5. Boxes with `confidence < threshold` are discarded.
6. Remaining boxes are returned in original image coordinates.

---

## 6. Bounding Box Explanation

A **bounding box** is the smallest axis-aligned rectangle that completely
encloses a detected pothole.

**YOLO format (stored in `.txt` annotation files):**
```
class_id  x_center  y_center  width  height
     0      0.3125    0.2083   0.3125  0.2083
```
- All values are **normalised** (0.0 to 1.0) relative to image dimensions.
- `x_center`, `y_center` = centre of the box as fractions of image width/height.
- `width`, `height` = box dimensions as fractions of image dimensions.

**Pascal VOC format (in RDD2022 XML files):**
```xml
<xmin>100</xmin> <ymin>50</ymin> <xmax>300</xmax> <ymax>150</ymax>
```
- Pixel coordinates of the top-left and bottom-right corners.

**Conversion formula:**
```
x_center = (xmin + xmax) / 2  /  img_width
y_center = (ymin + ymax) / 2  /  img_height
width    = (xmax - xmin)      /  img_width
height   = (ymax - ymin)      /  img_height
```

---

## 7. Confidence Score Explanation

The **confidence score** (0.0 – 1.0) encodes two probabilities:

```
confidence = P(Object exists in box) × P(Class = pothole | Object exists)
```

- A score of **0.95** means the model is 95% sure the box contains a pothole.
- Our default threshold is **0.50** — boxes below this are discarded as noise.
- A higher threshold → fewer but more reliable detections.
- A lower threshold → more detections but more false positives.

The confidence score is stored directly in the `Detection_Fact` table for
retrospective filtering and analysis.

---

## 8. How to Train This Model

### Prerequisites
```bash
pip install ultralytics opencv-python numpy
```

### Full training pipeline

```bash
# Step 1: Download and prepare dataset
python utils/prepare_dataset.py --source ./data/RDD2022/India

# Step 2: Convert annotations
python utils/convert_annotations.py

# Step 3: Split dataset
python utils/split_dataset.py --train 0.7 --val 0.2 --test 0.1

# Step 4: Train
yolo detect train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=416

# Step 5: Evaluate
yolo detect val model=runs/detect/pothole_v1/weights/best.pt data=dataset.yaml

# Step 6: Copy best weights for the app
cp runs/detect/pothole_v1/weights/best.pt models/best.pt
```

### Monitoring training

During training, YOLO logs metrics to `runs/detect/pothole_v1/results.csv`:
- `mAP50` — mean Average Precision at IoU threshold 0.50
- `mAP50-95` — mAP averaged over IoU thresholds 0.50 to 0.95
- `val/box_loss` — validation bounding box regression loss

Training curves are saved as `results.png` for visual inspection.

---

*YOLO documentation — Pothole Detection & Analysis System v1.0*
