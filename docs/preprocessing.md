# Data Preprocessing Documentation

## 1. Why Preprocessing Is Needed

Raw image data from a real-world road-damage dataset is almost never ready for
direct use in a machine-learning pipeline. Preprocessing is essential because:

- **Inconsistent sizes**: RDD2022 images come in varying resolutions.
  Neural networks require a fixed input size (416 × 416 for our model).
- **Label format mismatch**: RDD2022 ships with Pascal VOC (XML) annotations,
  while YOLOv8 needs normalised YOLO `.txt` format.
- **Class imbalance**: Most road images contain far more background than potholes.
  Augmentation helps balance the distribution.
- **Data quality issues**: Some images are blurry, over-/under-exposed, or
  contain annotation errors. Validation filters these out.
- **Generalisation**: A model trained on limited conditions (e.g. only daytime
  dry roads) will fail in rain or at night. Augmentation artificially adds variety.

---

## 2. The RDD2022 Dataset

**RDD2022** (Road Damage Dataset 2022) is the largest publicly available
road-surface damage dataset.

| Attribute | Detail |
|---|---|
| Images | ~47 000 labelled road images |
| Countries | Japan, India, USA, Czech Republic, Norway, China |
| Annotation format | Pascal VOC (XML) with bounding boxes |
| Classes | D00 (longitudinal cracks), D10 (transverse cracks), D20 (aligator cracks), **D40 (potholes)**, and others |
| License | Creative Commons CC BY 4.0 |
| Source | [sekilab.github.io/RoadDamageDetector](https://sekilab.github.io/RoadDamageDetector/) |

We use **only the India subset** of RDD2022 because:
- Indian roads share surface characteristics relevant to our target deployment.
- The India subset provides sufficient D40 examples (~5 000+ pothole annotations).
- Country-specific training reduces cross-domain noise.

---

## 3. The D40 Class — What It Means

In RDD2022's taxonomy, road damage is divided into:

```
D00  Longitudinal cracks   (cracks running along the road direction)
D10  Transverse cracks     (cracks running across the road)
D20  Alligator cracks      (web-like cracking pattern)
D40  POTHOLES              ← our target class
D44  Open joints
```

**D40** represents a **pothole** — a bowl-shaped depression in the road surface
caused by water erosion and traffic stress. Potholes are the most hazardous
type of road damage for vehicles and pedestrians.

We train YOLOv8 as a **single-class detector** targeting only D40, discarding
all other annotations. This simplification:
- Reduces model complexity.
- Improves precision on the one class we care about.
- Aligns with our project objective: pothole severity mapping.

---

## 4. Preprocessing Pipeline Steps

### Step 1 — Image Validation (`is_valid_image`)

Before any processing, each file is checked:
1. Does the file path exist on disk?
2. Can OpenCV decode the file (not corrupted, correct format)?
3. Does the image have 3 channels (RGB/BGR)?
4. Is the resolution above a minimum threshold (e.g. 100 × 100)?

Images failing any check are logged and skipped.

```python
valid, reason = is_valid_image("path/to/image.jpg")
if not valid:
    print(f"Skipping: {reason}")
```

---

### Step 2 — Annotation Conversion (VOC → YOLO)

RDD2022 annotations are in **Pascal VOC** format (XML):
```xml
<bndbox>
  <xmin>100</xmin> <ymin>50</ymin>
  <xmax>300</xmax> <ymax>150</ymax>
</bndbox>
```

YOLOv8 requires **YOLO normalised** format (`.txt`):
```
class_id  x_center  y_center  width  height
```
where all values are fractions of image width/height.

**Conversion formula:**
```
x_center = (xmin + xmax) / 2  /  image_width
y_center = (ymin + ymax) / 2  /  image_height
width    = (xmax - xmin)      /  image_width
height   = (ymax - ymin)      /  image_height
```

This normalisation ensures coordinates are resolution-independent, making the
model architecture agnostic to input image size.

---

### Step 3 — Image Resizing (`resize_image`)

All images are resized to **416 × 416 pixels** before feeding into the YOLO model.

**Letterboxing** is used to preserve the original aspect ratio:
- Compute scale factor to fit the longer dimension into 416.
- Scale both dimensions proportionally.
- Pad the remaining space with grey pixels (value 114 — the YOLO convention).

This prevents distortion (stretching) that would make square objects look
rectangular in the model's perspective.

```
Original 640×480  →  Scale by 416/640 = 0.65  →  416×312  →  Pad top/bottom by 52px  →  416×416
```

---

### Step 4 — Augmentation

Augmentation artificially increases dataset diversity.

#### 4.1 Horizontal Flip (`augment_flip`)

Mirrors the image left-to-right. Bounding box coordinates are adjusted:
```
new_xmin = image_width − old_xmax
new_xmax = image_width − old_xmin
```
Rationale: Potholes appear equally on left and right sides of roads.

#### 4.2 Vertical Flip

Less common but applied occasionally for completeness.
Effective for aerial/drone imagery.

#### 4.3 Brightness Adjustment (`augment_brightness`)

Randomly increases or decreases brightness by ±30 units:
- Simulates time-of-day variation (bright noon vs. overcast).
- Pixel values are clipped to [0, 255] after adjustment.

#### 4.4 Rotation (optional)

Small rotations (±10°) simulate camera tilt.
Applied sparingly — extreme rotation misrepresents real dashcam imagery.

#### 4.5 Gaussian Noise (optional)

Adds random noise to simulate low-quality cameras.

---

### Step 5 — Bounding Box Area Ratio (`compute_bbox_area_ratio`)

After detection at inference time, we compute:

```
bbox_area_ratio = (bbox_width × bbox_height) / (image_width × image_height)
```

This scalar is stored in the warehouse and used to determine severity:

| ratio | Severity |
|---|---|
| < 0.03 | Low |
| 0.03 – 0.07 | Medium |
| ≥ 0.07 | High |

---

## 5. Train / Validation / Test Split

We use a **70 / 20 / 10** split:

| Split | Percentage | Purpose |
|---|---|---|
| Train | 70% | Actual gradient-based model training |
| Validation | 20% | Hyperparameter tuning; early stopping |
| Test | 10% | Final unbiased performance evaluation |

**Why these ratios?**
- **70% training** ensures the model sees enough variety to learn robust features.
- **20% validation** is large enough to give stable metric estimates during tuning.
- **10% test** is held out strictly — never used until final reporting — to give an unbiased mAP score.

**Critical rule — no data leakage:**
The split is done **per image**, not per detection box. An image and all its
augmented versions belong to the same split. This prevents a training image's
augmented twin from appearing in the test set, which would inflate accuracy.

Files are split using `train_test_split` from scikit-learn with `random_state=42`
for reproducibility.

---

## 6. Summary of Tools Used

| Task | Tool |
|---|---|
| Image reading / writing | OpenCV (`cv2`) |
| Array manipulation | NumPy |
| Augmentation | Custom functions (NumPy) / Albumentations (optional) |
| Annotation parsing | Python `xml.etree.ElementTree` |
| Dataset splitting | scikit-learn `train_test_split` |

---

*Preprocessing documentation — Pothole Detection & Analysis System v1.0*
