# Project Architecture: Pothole Detection & Analysis System

## 1. System Overview

The **Pothole Detection & Analysis System** (project_pathhole) is an end-to-end data
mining and warehousing pipeline that:

1. **Detects** road potholes in images/video using a YOLOv8 deep-learning model trained
   on the RDD2022 dataset (specifically the D40 — "Pothole" class).
2. **Stores** every detection in a star-schema data warehouse (SQLite via SQLAlchemy).
3. **Analyses** historical detections using OLAP operations (Roll-Up, Drill-Down, Slice, Dice).
4. **Clusters** potholes geographically using K-Means to identify repair hotspots.
5. **Visualises** everything through an interactive Streamlit dashboard.

---

## 2. Data Flow Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT SOURCES                                   │
│                                                                         │
│   ┌───────────────┐   ┌───────────────┐   ┌────────────────────────┐   │
│   │  RDD2022      │   │  Uploaded     │   │  Live Webcam /         │   │
│   │  Dataset      │   │  Image File   │   │  Video Stream          │   │
│   │  (Training)   │   │  (Inference)  │   │  (Real-time)           │   │
│   └───────┬───────┘   └───────┬───────┘   └──────────┬─────────────┘   │
└───────────┼───────────────────┼──────────────────────┼─────────────────┘
            │                   │                      │
            ▼                   ▼                      ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                                │
│   • Image validation (is_valid_image)                                 │
│   • Resize to 416×416 (resize_image)                                  │
│   • Augmentation: flip, brightness, rotation                          │
│   • VOC → YOLO annotation conversion                                  │
│   • 70 / 20 / 10 Train / Val / Test split                             │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                      YOLO DETECTION LAYER                             │
│   YOLOv8 model (ultralytics)                                          │
│   • Training  : train on D40 class from RDD2022                       │
│   • Inference : returns bounding boxes + confidence scores            │
│   • Output    : Detection dict {bbox, confidence, severity, …}        │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
            ┌───────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────────┐
│                 SEVERITY CLASSIFICATION                               │
│   bbox_area_ratio → Low / Medium / High                               │
│   • Low    : ratio < 0.03                                             │
│   • Medium : 0.03 ≤ ratio < 0.07                                      │
│   • High   : ratio ≥ 0.07                                             │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│               DATA WAREHOUSE (SQLite + SQLAlchemy)                   │
│                                                                       │
│   Dimensions           │  Fact Table                                  │
│   ─────────────────    │  ──────────────────────────────────          │
│   Severity_Dim         │  Detection_Fact                              │
│   Road_Dim             │    FK → Severity_Dim                         │
│   Location_Dim         │    FK → Road_Dim                             │
│   Time_Dim             │    FK → Location_Dim                         │
│                        │    FK → Time_Dim                             │
│                        │    confidence, bbox_area_ratio               │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
            ┌───────────────────┴────────────────────┐
            │                                        │
            ▼                                        ▼
┌─────────────────────────┐            ┌─────────────────────────────┐
│     OLAP MODULE         │            │   K-MEANS CLUSTERING        │
│  • Roll-Up  (3 levels)  │            │  • Encode features           │
│  • Drill-Down           │            │  • StandardScaler            │
│  • Slice (1 filter)     │            │  • KMeans (k=3 default)      │
│  • Dice  (n filters)    │            │  • Silhouette score          │
└──────────────┬──────────┘            └────────────────┬────────────┘
               │                                        │
               └─────────────────┬──────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD (app.py)                        │
│                                                                       │
│  Tabs:                                                                │
│  ① Home        – Project overview & live detection toggle             │
│  ② Detection   – Upload image / run webcam detection                  │
│  ③ Analytics   – OLAP operations & charts                             │
│  ④ Clustering  – K-Means hotspot map                                  │
│  ⑤ Database    – Raw Detection_Fact table viewer                      │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Descriptions

### 3.1 `utils/preprocessing.py`
Handles all data preparation before training or inference:
- **`is_valid_image(path)`** — checks file existence and readability.
- **`resize_image(img, target_size)`** — resizes with letterboxing to preserve aspect ratio.
- **`augment_flip(img)`** — horizontal/vertical flip augmentation.
- **`augment_brightness(img)`** — random brightness ±30 adjustment.
- **`draw_bounding_boxes(img, boxes)`** — annotates images for visual review.
- **`voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)`** — converts Pascal VOC pixel coordinates to YOLO normalised fractions.
- **`compute_bbox_area_ratio(…)`** — ratio of bbox area to image area (used for severity).

### 3.2 `database.py`
Defines the SQLAlchemy ORM models (star schema) and CRUD helpers:
- Schema creation via `Base.metadata.create_all(engine)`.
- `seed_severity_dim(session)` — inserts Low/Medium/High rows once.
- `ingest_detection(session, record)` — idempotent insert (skips duplicates).
- `get_detection_summary(session)` — aggregate query.
- `get_monthly_trend(session)` — group-by month query.
- `get_severity_distribution(session)` — group-by severity query.

### 3.3 `olap.py`
Implements the four classical OLAP operations:
- **Roll-Up**: `rollup_by_road()`, `rollup_by_area()`, `rollup_by_city()`.
- **Drill-Down**: `drilldown_cities()`, `drilldown_to_areas(city)`.
- **Slice**: `slice_by_severity(session, severity)`.
- **Dice**: `dice(session, **filters)`.

All return pandas DataFrames for easy downstream visualisation.

### 3.4 `kmeans_clustering.py`
K-Means clustering pipeline for hotspot analysis:
- **`encode_severity(label)`** / **`encode_road_type(label)`** — ordinal encoding.
- **`prepare_feature_matrix(df)`** — builds + scales feature matrix.
- **`fit_kmeans(df, n_clusters, random_state)`** — trains and assigns cluster labels.
- **`get_cluster_summary(df)`** — per-cluster mean statistics.
- **`compute_silhouette(df, labels)`** — quality metric.

### 3.5 `app.py` — Streamlit Dashboard
Single-file multi-tab web application:
- Uses `st.camera_input` for real-time webcam detection.
- Uses `st.file_uploader` for image upload.
- Renders Plotly charts for OLAP results.
- Renders `st.map` / Plotly scatter_geo for cluster hotspots.

---

## 4. Technology Choices and Rationale

| Technology | Choice | Rationale |
|---|---|---|
| Detection model | YOLOv8 (ultralytics) | Real-time single-pass detection; best speed/accuracy trade-off for edge deployment |
| Dataset | RDD2022 | Largest publicly available road-damage dataset; multi-country coverage |
| Database | SQLite | Zero-config; portable; sufficient for a city-scale PoC |
| ORM | SQLAlchemy | Database-agnostic; easy migration to PostgreSQL/MySQL for production |
| OLAP | Custom SQL via SQLAlchemy | Demonstrates warehouse concepts without heavyweight OLAP server |
| Clustering | Scikit-learn K-Means | Simple, interpretable, fast on tabular data |
| Dashboard | Streamlit | Rapid prototyping; pure Python; no JavaScript required |
| Image processing | OpenCV + NumPy | Industry-standard CV library; native YOLO integration |
| Testing | Pytest | Widely adopted; excellent fixture support; clean syntax |

---

## 5. Deployment Diagram

```
┌─────────────────────────────────────────────┐
│             LOCAL MACHINE / SERVER           │
│                                             │
│  ┌──────────────┐    ┌────────────────────┐ │
│  │  Browser     │◄──►│  Streamlit Server  │ │
│  │  (Port 8501) │    │  app.py            │ │
│  └──────────────┘    └────────┬───────────┘ │
│                               │             │
│            ┌──────────────────┼──────────┐  │
│            │                  │          │  │
│            ▼                  ▼          ▼  │
│  ┌──────────────┐  ┌────────────┐  ┌──────┐ │
│  │ YOLOv8 Model │  │ database.py│  │olap/ │ │
│  │ best.pt      │  │ SQLite DB  │  │kmeans│ │
│  └──────────────┘  └────────────┘  └──────┘ │
└─────────────────────────────────────────────┘

To deploy remotely:
  • Replace SQLite with PostgreSQL (change connection string only)
  • Deploy Streamlit on Streamlit Cloud / AWS EC2 / Docker container
  • Mount model weights (best.pt) as a volume or download from S3
```

---

*Generated for: Data Mining & Warehousing Project — Pothole Detection System*
*Last updated: September 2026*
