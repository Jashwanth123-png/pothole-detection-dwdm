# REAL-TIME POTHOLE DETECTION USING DATA MINING & DATA WAREHOUSING

> **College Project — Data Warehousing and Data Mining (DWDM) Course**

---

## Project Overview

This project is a college-level submission for the **Data Warehousing and Data Mining (DWDM)** course. It demonstrates a complete, end-to-end data engineering and machine learning pipeline applied to a real-world civil infrastructure problem: **pothole detection and analysis**.

### Problem Statement

Potholes are a persistent and dangerous road infrastructure problem:

- 🚗 **Road safety risk** — Potholes cause thousands of accidents and injuries every year.
- 🔧 **Vehicle damage** — Suspension, tyre, and axle damage cost vehicle owners billions annually.
- 💰 **Economic loss** — Delayed pothole repairs escalate road maintenance costs significantly.
- 🏙️ **Urban planning gap** — Municipal authorities lack systematic, data-driven tools to prioritize pothole repairs.

### What This System Does (End-to-End)

1. **Data Acquisition** — Uses the RDD2022 (Road Damage Dataset 2022) benchmark dataset containing real road images with annotated potholes.
2. **Data Preprocessing** — Cleans, filters, converts, resizes, augments, and splits the dataset into training/validation/test sets.
3. **Model Training** — Trains a YOLOv8 object detection model to detect potholes in road images.
4. **Inference** — Runs the trained YOLO model on new road images to produce detection records (bounding boxes, confidence scores, severity estimates).
5. **Data Warehousing** — Stores all detection records in a structured Star Schema data warehouse (SQLite via SQLAlchemy).
6. **OLAP Analysis** — Applies Roll-Up, Drill-Down, Slice, and Dice operations on the warehouse for multi-dimensional analysis.
7. **Data Mining** — Applies K-Means clustering to discover pothole hotspot patterns across spatial and temporal dimensions.
8. **Dashboard** — Presents all results through an interactive 12-page Streamlit dashboard with charts, maps, and analytics.

---

## Objectives

1. **Detect potholes** in road images using a state-of-the-art YOLOv8 deep learning model.
2. **Preprocess and clean** the RDD2022 dataset using a structured pipeline (filtering, annotation conversion, resizing, augmentation).
3. **Design and implement a Star Schema data warehouse** to store and manage detection records efficiently.
4. **Apply OLAP operations** (Roll-Up, Drill-Down, Slice, Dice) to enable multi-dimensional analysis of pothole data.
5. **Apply K-Means clustering** to mine pothole hotspots and identify patterns across locations and time periods.
6. **Estimate severity** of each detected pothole using a project-defined heuristic based on bounding box size and confidence score.
7. **Build an interactive dashboard** with Streamlit that presents all analytics, maps, and data mining results to end users.
8. **Demonstrate real-world data engineering** by integrating computer vision, data warehousing, OLAP, and data mining into a single cohesive system.

---

## System Architecture

```
Road Camera / Video
        ↓
Pothole Detection (YOLO)
        ↓
Detection Records
        ↓
Data Warehouse (Star Schema)
        ↓
OLAP Analysis
        ↓
K-Means Data Mining
        ↓
Pothole Hotspot Analytics
        ↓
Interactive Dashboard
```

### Detailed Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    RDD2022 Dataset                       │
│          (India road images + XML annotations)           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               Data Preprocessing Pipeline                │
│  clean → filter D40 → convert XML→YOLO → resize →       │
│  augment → split (70/20/10)                              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              YOLOv8 Object Detection Model               │
│         Training (ml/train_yolo.py)                      │
│         Inference → Detection Records                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          Data Warehouse  (SQLite + SQLAlchemy)           │
│                   Star Schema                            │
│   FactDetection ── DimLocation ── DimTime ── DimSeverity │
└──────────┬──────────────────────┬──────────────────────-┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌───────────────────────────────────┐
│  OLAP Operations │   │       K-Means Clustering           │
│  Roll-Up         │   │  Feature Engineering → Normalize → │
│  Drill-Down      │   │  K-Means → Hotspot Labels          │
│  Slice           │   └────────────────┬──────────────────┘
│  Dice            │                    │
└──────────┬───────┘                    │
           └────────────┬───────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│            Streamlit Interactive Dashboard               │
│                    (12 Pages)                            │
└─────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Category | Technology | Version |
|---|---|---|
| **Language** | Python | 3.10+ |
| **Object Detection** | Ultralytics YOLOv8 | `ultralytics>=8.0.0` |
| **Deep Learning Framework** | PyTorch | `torch>=2.0.0` |
| **Image Processing** | OpenCV | `opencv-python>=4.8.0` |
| **Image Processing** | Pillow (PIL) | `Pillow>=10.0.0` |
| **Data Processing** | Pandas | `pandas>=2.0.0` |
| **Numerical Computing** | NumPy | `numpy>=1.24.0` |
| **Machine Learning (K-Means)** | scikit-learn | `scikit-learn>=1.3.0` |
| **Database ORM** | SQLAlchemy | `SQLAlchemy>=2.0.0` |
| **Database Engine** | SQLite | Built-in with Python |
| **Dashboard** | Streamlit | `streamlit>=1.28.0` |
| **Charts & Visualization** | Plotly | `plotly>=5.17.0` |
| **Geospatial Maps** | Folium + streamlit-folium | `folium>=0.14.0` |
| **Configuration** | PyYAML | `pyyaml>=6.0` |
| **Progress Bars** | tqdm | `tqdm>=4.66.0` |
| **Logging** | Loguru | `loguru>=0.7.0` |
| **Testing** | pytest | `pytest>=7.4.0` |
| **Image Deduplication** | imagehash | `imagehash>=4.3.1` (optional) |

---

## Dataset — RDD2022

### What is RDD2022?

**RDD2022 (Road Damage Dataset 2022)** is a large-scale, publicly available benchmark dataset for road damage detection. It was collected by researchers from Tohoku University, Japan (sekilab), and contains road images from multiple countries including **Japan, India, the Czech Republic, Norway, and the United States**.

The dataset includes:
- High-resolution road surface images captured from vehicles
- Bounding-box annotations in XML (Pascal VOC) format
- Multiple damage categories (longitudinal cracks, transverse cracks, alligator cracks, potholes, etc.)

### Why RDD2022 Was Chosen

- ✅ **Real-world data** — Genuine road images, not synthetic
- ✅ **India subset available** — Relevant to the Indian road infrastructure context
- ✅ **Standardised annotations** — Pascal VOC XML format, well-documented
- ✅ **Academically recognised** — Used in international IEEE Big Data Challenge competitions
- ✅ **Free and open** — Publicly available for academic use
- ✅ **Pothole class included** — Contains the `D40` class specifically for potholes

### The D40 (Pothole) Class

RDD2022 uses a structured damage classification system. This project uses **only the D40 class**:

| Code | Description | Used in This Project |
|---|---|---|
| D00 | Longitudinal Crack | ❌ Filtered out |
| D10 | Transverse Crack | ❌ Filtered out |
| D20 | Alligator Crack | ❌ Filtered out |
| **D40** | **Pothole** | ✅ **Yes — Primary class** |

The preprocessing pipeline (`scripts/filter_d40.py`) filters the dataset to retain only images that contain at least one D40 (pothole) annotation.

### Dataset Structure

After downloading and placing RDD2022, the expected structure is:

```
dataset/
└── rdd2022/
    └── India/
        ├── images/
        │   └── train/
        │       ├── India_000001.jpg
        │       ├── India_000002.jpg
        │       └── ...
        └── annotations/
            └── xmls/
                ├── India_000001.xml
                ├── India_000002.xml
                └── ...
```

> **Note:** The `dataset/rdd2022/` directory is included in `.gitignore` and is NOT committed to version control due to its large size (~12 GB).

### How to Download RDD2022

**Official Repository:**
🔗 [https://github.com/sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)

**Steps:**

1. Visit the official link above and follow the dataset download instructions.
2. Download the **India** subset (or the full dataset if desired).
3. Extract the downloaded archive.
4. Place the files under `dataset/rdd2022/` as shown in the structure above.
5. Run the preparation script:
   ```bash
   python scripts/prepare_dataset.py
   ```

---

## Data Preprocessing Pipeline

The preprocessing pipeline transforms raw RDD2022 data into a clean, YOLO-ready dataset. Each step is implemented as a separate, reusable script.

### Step-by-Step Pipeline

```
Raw RDD2022 Images + XML Annotations
            │
            ▼
  Step 1 — CLEANING (scripts/clean_dataset.py)
  • Remove corrupt/unreadable images
  • Remove XML files with no valid annotations
  • Remove duplicate images using perceptual hashing (imagehash)
  • Log all removed files
            │
            ▼
  Step 2 — FILTERING (scripts/filter_d40.py)
  • Parse each XML annotation file
  • Retain only images that contain at least one D40 (pothole) label
  • Discard images with only D00/D10/D20 labels
  • Produces a focused pothole-only subset
            │
            ▼
  Step 3 — ANNOTATION CONVERSION (scripts/convert_annotations.py)
  • Convert Pascal VOC XML format → YOLO TXT format
  • YOLO format: <class_id> <cx> <cy> <width> <height> (normalised 0–1)
  • Class ID for D40 (pothole) = 0
  • One .txt file created per image
            │
            ▼
  Step 4 — RESIZING (scripts/resize_images.py)
  • Resize all images to 416×416 pixels (YOLOv8 default input size)
  • Maintain aspect ratio with letterboxing to avoid distortion
  • Update bounding box coordinates accordingly
            │
            ▼
  Step 5 — AUGMENTATION (utils/preprocessing.py)
  • Horizontal flip (mirror) — increases variety
  • Random brightness/contrast adjustments — handles lighting variation
  • Slight rotation (±5°) — handles camera angle variation
  • Applied only to the training split to prevent data leakage
            │
            ▼
  Step 6 — SPLITTING (utils/preprocessing.py)
  • Randomly shuffle the filtered dataset
  • Split into 3 subsets (see rationale below)
  • Output structure placed in data/ for YOLO training config
```

### The 70 / 20 / 10 Split

| Split | Proportion | Purpose |
|---|---|---|
| **Training set** | 70% | Used to train the YOLOv8 model weights |
| **Validation set** | 20% | Used during training to tune hyperparameters and monitor overfitting |
| **Test set** | 10% | Held-out set used only at the end to report final model performance |

> **Why this ratio?** The 70/20/10 split is a widely used standard in machine learning. The larger training set ensures the model sees sufficient examples, the validation set is large enough for reliable hyperparameter tuning, and the test set, while smaller, remains representative for final evaluation.

---

## YOLO Pothole Detection

### What is YOLOv8?

**YOLO (You Only Look Once)** is a family of real-time object detection models. **YOLOv8** (by Ultralytics, 2023) is the latest generation, offering:

- **Single-pass detection** — The entire image is processed in one forward pass through the neural network, making it extremely fast.
- **Anchor-free architecture** — Improves detection of small and irregularly shaped objects (like potholes).
- **State-of-the-art accuracy** — Achieves high mAP (mean Average Precision) on standard benchmarks.
- **Easy training API** — The `ultralytics` Python library makes training, validation, and inference straightforward.

### Training (`ml/train_yolo.py`)

The training script:
1. Loads a pre-trained YOLOv8 base model (e.g., `yolov8n.pt` for nano).
2. Fine-tunes it on the preprocessed pothole dataset using transfer learning.
3. Saves the best model weights to `models/pothole_yolo.pt`.
4. Logs training metrics (loss, mAP50, mAP50-95) per epoch.

**Key hyperparameters** (configurable in `config.yaml`):
- `epochs`: Number of training epochs (default: 50)
- `batch`: Batch size (default: 16)
- `imgsz`: Input image size (default: 640)
- `conf`: Confidence threshold for detections (default: 0.25)

### Inference

When `models/pothole_yolo.pt` exists, the system runs inference on new road images:
- Outputs bounding box coordinates `(x1, y1, x2, y2)`
- Outputs confidence score (0–1)
- Derives severity label via a project-defined heuristic (see `utils/severity.py`)
- Stores each detection as a record in the data warehouse

### Demo Mode

If no trained model is available, the system operates in **Demo Mode**:
- Uses pre-seeded synthetic detection records (`warehouse/seed_demo_data.py`)
- All dashboard features remain fully functional
- Demo data is clearly labelled — it is **NOT** from real RDD2022 inference

---

## Data Warehouse

### Star Schema Design

The data warehouse uses a **Star Schema** — the industry-standard structure for OLAP and analytical workloads. It consists of one central fact table surrounded by dimension tables.

```
          DimTime
             │
             │
DimLocation──┼──FactDetection──DimSeverity
             │
             │
          DimRoad
```

### Tables

#### `FactDetection` (Fact Table)
The central table — one row per detected pothole.

| Column | Type | Description |
|---|---|---|
| `detection_id` | INTEGER PK | Unique detection identifier |
| `location_id` | INTEGER FK | Links to DimLocation |
| `time_id` | INTEGER FK | Links to DimTime |
| `severity_id` | INTEGER FK | Links to DimSeverity |
| `confidence_score` | FLOAT | YOLO model confidence (0–1) |
| `bbox_area_px` | FLOAT | Bounding box area in pixels |
| `image_filename` | TEXT | Source image filename |

#### `DimLocation` (Dimension Table)
Spatial information for each detection.

| Column | Type | Description |
|---|---|---|
| `location_id` | INTEGER PK | Unique location key |
| `latitude` | FLOAT | GPS latitude (simulated/future) |
| `longitude` | FLOAT | GPS longitude (simulated/future) |
| `city` | TEXT | City name |
| `district` | TEXT | District name |
| `road_name` | TEXT | Road identifier |
| `zone` | TEXT | Zone/area label |

#### `DimTime` (Dimension Table)
Temporal hierarchy for OLAP Roll-Up/Drill-Down.

| Column | Type | Description |
|---|---|---|
| `time_id` | INTEGER PK | Unique time key |
| `timestamp` | DATETIME | Full detection timestamp |
| `hour` | INTEGER | Hour of day (0–23) |
| `day` | INTEGER | Day of month |
| `month` | INTEGER | Month (1–12) |
| `quarter` | INTEGER | Quarter (1–4) |
| `year` | INTEGER | Year |
| `day_of_week` | TEXT | e.g., Monday |
| `is_weekend` | BOOLEAN | True if Saturday/Sunday |

#### `DimSeverity` (Dimension Table)
Project-defined severity classification.

| Column | Type | Description |
|---|---|---|
| `severity_id` | INTEGER PK | Unique severity key |
| `severity_label` | TEXT | LOW / MEDIUM / HIGH / CRITICAL |
| `severity_score` | INTEGER | Numeric score (1–4) |
| `description` | TEXT | Human-readable description |

> **Note:** Severity is a **project-defined heuristic** derived from bounding box area and YOLO confidence score. It is NOT a label from the RDD2022 dataset.

---

## OLAP Operations

OLAP (Online Analytical Processing) operations enable multi-dimensional analysis of the data warehouse without writing complex SQL queries each time.

### Roll-Up (`olap/rollup.py`)
Aggregates data from a **lower level to a higher level** of a dimension hierarchy.

- **Example:** Detection counts aggregated from `Day → Month → Quarter → Year`
- **Use case:** "How many potholes were detected per month this year?"

### Drill-Down (`olap/drilldown.py`)
The reverse of Roll-Up — navigates from a **higher level to a lower level** for more detail.

- **Example:** Zoom from `City → District → Road → Specific Location`
- **Use case:** "Which specific road in Bangalore has the most potholes this quarter?"

### Slice (`olap/slice.py`)
Selects a **single value** for one dimension and analyses the remaining dimensions.

- **Example:** Fix `severity = 'CRITICAL'` and analyse detections across all locations and times.
- **Use case:** "Show me all critical potholes regardless of location or time."

### Dice (`olap/dice.py`)
Selects a **sub-cube** by applying filters on **two or more dimensions** simultaneously.

- **Example:** Filter to `city = 'Mumbai'` AND `month = 'June'` AND `severity = 'HIGH'`
- **Use case:** "Show me high-severity potholes in Mumbai during June."

---

## K-Means Clustering

K-Means is an unsupervised machine learning algorithm applied here to discover **pothole hotspots** — geographic and temporal clusters with high detection density.

### Feature Engineering (`mining/prepare_features.py`)

Each detection record is represented by the following features for clustering:

| Feature | Description |
|---|---|
| `latitude` | Spatial coordinate |
| `longitude` | Spatial coordinate |
| `hour_of_day` | Time of detection (0–23) |
| `day_of_week` | Day as integer (0=Monday) |
| `severity_score` | Numeric severity (1–4) |
| `confidence_score` | YOLO confidence (0–1) |

### Normalisation

All features are normalised using **StandardScaler** (zero mean, unit variance) from scikit-learn before clustering. This prevents high-magnitude features (like lat/lon) from dominating the distance calculations.

### Clustering (`mining/kmeans.py`)

- **Algorithm:** K-Means (`sklearn.cluster.KMeans`)
- **K selection:** The optimal number of clusters `K` is determined using the **Elbow Method** (plotting inertia vs. K values)
- **Initialization:** `k-means++` for better convergence
- **Runs:** Multiple random initializations (`n_init=10`) to avoid local minima

### Cluster Interpretation

After clustering, each cluster is profiled by its centroid values:

| Cluster Profile | Interpretation |
|---|---|
| High density, high severity, central zone | **Priority 1 Hotspot** — Immediate repair needed |
| Medium density, mixed severity | **Regular maintenance zone** |
| Low density, low severity, peripheral zone | **Low-priority area** — Monitor only |

---

## Dashboard

The interactive dashboard is built with **Streamlit** and **Plotly**. It provides 12 pages accessible from the sidebar.

| # | Page | Description |
|---|---|---|
| 1 | **🏠 Home / Overview** | Project summary, key metrics (total detections, hotspots, severity breakdown), system status indicator |
| 2 | **📊 Detection Summary** | Total detection counts, daily/weekly trends, confidence score distribution histogram |
| 3 | **🗺️ Pothole Map** | Interactive Folium map showing all detected potholes colour-coded by severity |
| 4 | **📅 Time Analysis** | OLAP Roll-Up charts — detections by hour, day, month, quarter, year |
| 5 | **📍 Location Analysis** | OLAP Drill-Down from city → district → road; top roads by detection count |
| 6 | **⚠️ Severity Analysis** | OLAP Slice — filter by severity level; pie chart, bar chart of severity distribution |
| 7 | **🔍 OLAP Dice** | Interactive OLAP Dice tool — user selects dimension filters to query sub-cubes |
| 8 | **🔥 Hotspot Clusters** | K-Means clustering results — scatter map of clusters, elbow curve, cluster profiles |
| 9 | **📈 Trend Analysis** | Long-term pothole detection trends; peak hours and peak seasons |
| 10 | **🖼️ Detection Gallery** | Browse sample detection images with bounding boxes and confidence annotations |
| 11 | **⚙️ Data Pipeline Status** | Step-by-step pipeline status: dataset ready? model trained? warehouse populated? |
| 12 | **ℹ️ About / Documentation** | Project description, tech stack, academic information, disclaimer about demo data |

---

## Installation

### Prerequisites

- Python **3.10 or higher**
- `pip` package manager
- (Optional) NVIDIA GPU with CUDA for faster YOLO training

```bash
# 1. Clone or navigate to project
cd project_pathhole

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU Users:** If you have a CUDA-capable GPU, install the CUDA version of PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> Then run `pip install -r requirements.txt`.

---

## Dataset Setup

```
1. Download RDD2022 from: https://github.com/sekilab/RoadDamageDetector

2. Place files at: dataset/rdd2022/
   Structure expected:
   dataset/rdd2022/
       India/
           images/
               train/  (contains .jpg files)
           annotations/
               xmls/  (contains .xml files)

3. Run: python scripts/prepare_dataset.py
```

The `prepare_dataset.py` script will automatically run the full pipeline:
- Clean → Filter D40 → Convert annotations → Resize → Augment → Split (70/20/10)

---

## Training YOLO

```bash
# After dataset preparation:
python ml/train_yolo.py

# Model saved to: models/pothole_yolo.pt
```

Training progress and metrics are logged to the `logs/` directory and displayed in the terminal with a progress bar. Training time depends on hardware:
- CPU only: ~2–6 hours for 50 epochs
- GPU (NVIDIA): ~15–45 minutes for 50 epochs

---

## Initializing the Database

```bash
python -c "from warehouse.database import initialize_database; initialize_database(); print('DB initialized')"
```

This creates the SQLite database file (`data/pothole_warehouse.db`) and sets up all Star Schema tables.

---

## Running Demo Mode

```bash
# Seed demo data (clearly labeled, NOT RDD2022)
python warehouse/seed_demo_data.py

# Run dashboard
streamlit run app.py
```

Demo mode populates the warehouse with **synthetic detection records** so that all dashboard pages, OLAP operations, and K-Means clustering are fully functional without requiring the RDD2022 dataset or a trained model.

---

## Running Full Application

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Data preprocessing utilities
- Annotation conversion correctness
- Database schema integrity
- OLAP query outputs
- K-Means feature pipeline
- Severity classification logic

---

## Project Structure

```
project_pathhole/
│
├── app.py                          # Main Streamlit dashboard entry point
├── config.yaml                     # Project-wide configuration (paths, hyperparameters)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules (dataset, models, venv, etc.)
│
├── scripts/                        # Data preprocessing scripts
│   ├── __init__.py
│   ├── clean_dataset.py            # Step 1: Remove corrupt/duplicate images
│   ├── filter_d40.py               # Step 2: Keep only D40 (pothole) annotations
│   ├── convert_annotations.py      # Step 3: XML → YOLO TXT format
│   ├── resize_images.py            # Step 4: Resize images to 416×416
│   └── prepare_dataset.py          # Master script: runs all steps in order
│
├── ml/                             # Machine learning (YOLO)
│   ├── __init__.py
│   ├── train_yolo.py               # YOLOv8 training script
│   └── evaluate_yolo.py            # Model evaluation (mAP, precision, recall)
│
├── warehouse/                      # Data warehouse layer
│   ├── __init__.py
│   ├── schema.sql                  # Raw SQL schema definition
│   ├── models.py                   # SQLAlchemy ORM models (Star Schema tables)
│   ├── database.py                 # DB connection, initialization, session management
│   ├── ingestion.py                # Ingest detection records into the warehouse
│   ├── queries.py                  # Reusable warehouse query functions
│   └── seed_demo_data.py           # Seed synthetic demo data for demo mode
│
├── olap/                           # OLAP operations
│   ├── __init__.py
│   ├── queries.py                  # Shared OLAP query helpers
│   ├── rollup.py                   # Roll-Up aggregation functions
│   ├── drilldown.py                # Drill-Down navigation functions
│   ├── slice.py                    # Slice (single-dimension filter) functions
│   └── dice.py                     # Dice (multi-dimension filter) functions
│
├── mining/                         # Data mining (K-Means clustering)
│   ├── __init__.py
│   ├── prepare_features.py         # Feature engineering for clustering
│   └── kmeans.py                   # K-Means clustering, elbow method, profiling
│
├── utils/                          # Shared utility functions
│   ├── __init__.py
│   ├── preprocessing.py            # Augmentation and train/val/test split logic
│   ├── severity.py                 # Severity classification heuristic
│   └── helpers.py                  # General helper functions
│
├── dashboard/                      # Dashboard page modules
│   └── __init__.py                 # Dashboard package initializer
│
├── tests/                          # Unit and integration tests
│   └── __init__.py
│
├── dataset/
│   └── rdd2022/                    # ← Place RDD2022 dataset here (not in git)
│       └── .gitkeep
│
├── models/
│   ├── README.md                   # Instructions for obtaining the trained model
│   └── pothole_yolo.pt             # ← Trained YOLOv8 weights (generated by training)
│
├── data/
│   ├── demo/                       # Demo data outputs
│   │   └── .gitkeep
│   └── outputs/                    # Inference outputs (annotated images, CSV exports)
│       └── .gitkeep
│
└── logs/                           # Training logs, pipeline logs
    └── .gitkeep
```

---

## Demo Mode vs Real Mode

### Demo Mode

Demo mode is designed for situations where:
- The RDD2022 dataset has not been downloaded
- The YOLO model has not been trained yet
- You want to explore the dashboard's full functionality immediately

**What demo mode does:**
1. Runs `warehouse/seed_demo_data.py` to insert ~500 synthetic detection records into the warehouse.
2. The records have realistic-looking values (lat/lon coordinates, timestamps, severity levels, confidence scores).
3. All 12 dashboard pages work fully — OLAP operations, K-Means clustering, maps, and charts all render.

> [!CAUTION]
> **Demo data is NOT RDD2022 results.** It is synthetically generated data for demonstration purposes only. It does **not** represent real pothole detections from the RDD2022 dataset or any real road. Do not cite demo mode results as experimental findings.

### How to Switch to Real Mode

1. Download RDD2022 and place it at `dataset/rdd2022/`
2. Run `python scripts/prepare_dataset.py`
3. Run `python ml/train_yolo.py` to train the model
4. Run inference on images to populate the warehouse with real detections
5. Run `streamlit run app.py` — the dashboard automatically uses real data if the warehouse is populated with real records

---

## Project Limitations

| Limitation | Details |
|---|---|
| **No trained model included** | Due to file size constraints, `models/pothole_yolo.pt` is not committed to the repository. Training must be performed by the user. |
| **GPS unavailable without hardware** | Real GPS coordinates require a GPS-equipped camera or vehicle sensor. In this project, location data is either simulated or derived from dataset metadata. |
| **Severity is project-defined** | The LOW/MEDIUM/HIGH/CRITICAL severity classification is a **heuristic defined for this project** based on bounding box size and confidence score. It is NOT a label from RDD2022 or any validated severity standard. |
| **Demo data is synthetic** | The demo mode seed data is randomly generated and does NOT reflect real-world pothole detections. |
| **Single damage class** | Only the D40 (pothole) class from RDD2022 is used. Other damage types (cracks, etc.) are filtered out. |
| **SQLite database** | SQLite is used for simplicity and portability. For production deployments, PostgreSQL or MySQL would be more appropriate. |
| **No real-time video stream** | The current implementation processes individual images. Real-time video stream inference requires additional integration work. |

---

## Future Scope

| Enhancement | Description |
|---|---|
| **GPS Integration** | Integrate with vehicle-mounted GPS to record exact pothole coordinates automatically |
| **Mobile Application** | Build an Android/iOS app allowing citizens to report potholes with photos and auto-GPS tagging |
| **Cloud Deployment** | Deploy the dashboard on AWS / GCP / Azure with a cloud-hosted database for city-wide use |
| **Real-Time Alert System** | Send automatic SMS/email alerts to municipal authorities when a new pothole cluster exceeds a severity threshold |
| **More Damage Classes** | Extend the YOLO model to detect D00 (longitudinal crack), D10 (transverse crack), and D20 (alligator crack) in addition to D40 (pothole) |
| **Repair Tracking** | Add a repair status workflow — mark potholes as "Reported → Assigned → Under Repair → Fixed" |
| **Predictive Analytics** | Use time-series forecasting (ARIMA, LSTM) to predict future pothole formation hotspots based on seasonal trends |
| **Road Authority API** | Build a REST API layer so road authority dashboards can query pothole data programmatically |

---

## Academic Information

This project was built as a complete demonstration of the following DWDM course topics:

| Topic | Implementation in This Project |
|---|---|
| **Data Preprocessing** | `scripts/` — cleaning, filtering, annotation conversion, resizing, augmentation, splitting |
| **Data Warehousing** | `warehouse/` — Star Schema design with Fact + Dimension tables (SQLAlchemy + SQLite) |
| **OLAP** | `olap/` — Roll-Up, Drill-Down, Slice, Dice operations on the warehouse |
| **Data Mining (K-Means)** | `mining/` — Feature engineering, StandardScaler normalisation, K-Means clustering, elbow method, cluster profiling |
| **Computer Vision (YOLO)** | `ml/` — YOLOv8 object detection trained on RDD2022 D40 (pothole) class |
| **Visualisation** | `app.py` + `dashboard/` — 12-page Streamlit dashboard with Plotly charts and Folium maps |

All algorithms, techniques, and system components align with the project specification document for the DWDM course.

### Key Concepts Demonstrated

- **ETL Pipeline** — Extract (RDD2022), Transform (preprocessing pipeline), Load (warehouse ingestion)
- **Star Schema** — Dimensional modelling for analytical queries
- **OLAP Cube Operations** — Multi-dimensional analysis without raw SQL complexity
- **Unsupervised Learning** — K-Means clustering for hotspot discovery
- **Transfer Learning** — Fine-tuning a pre-trained YOLOv8 on a domain-specific dataset
- **Dashboard Design** — End-to-end data product for non-technical stakeholders

---

## Authors

> **[Student Project Placeholder]**
>
> This project was developed as part of the **Data Warehousing and Data Mining (DWDM)** course.
>
> | Field | Details |
> |---|---|
> | **Course** | Data Warehousing and Data Mining (DWDM) |
> | **Academic Year** | 2025–2026 |
> | **Institution** | [Your Institution Name] |
> | **Department** | [Your Department] |
> | **Guide / Supervisor** | [Faculty Name] |
> | **Team Members** | [Student Name(s) and Roll Numbers] |

---

## License

This project is submitted for academic evaluation. The RDD2022 dataset is subject to its own license — please refer to the [official RDD2022 repository](https://github.com/sekilab/RoadDamageDetector) for terms of use.

---

*Last updated: September 2026*
