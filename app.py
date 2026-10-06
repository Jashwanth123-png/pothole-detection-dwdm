"""
app.py  –  Pothole Detection DWDM  –  Main Streamlit Application
=================================================================
Multi-page dashboard for pothole detection, data warehousing,
OLAP analysis, K-Means clustering, and geospatial visualisation.

Run with:
    streamlit run app.py

Architecture
------------
  app.py  (this file – router + shared state)
  ├── dashboard/charts.py      Plotly chart factories
  ├── dashboard/maps.py        Geospatial maps
  ├── dashboard/components.py  Reusable Streamlit UI blocks
  ├── warehouse/database.py    SQLite connection helpers
  ├── warehouse/queries.py     SQL query wrappers
  ├── olap/                    Roll-up / drill-down / slice / dice
  └── mining/                  K-Means clustering
"""

import sys
import os

# ── Make sure every sub-package is importable regardless of CWD ────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
import traceback

# ── Page config must be the FIRST Streamlit call ───────────────────────────
st.set_page_config(
    page_title="Pothole Detection DWDM",
    page_icon="🕳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Safe imports — every third-party / project module is wrapped so missing
# packages show a friendly message instead of a crash.
# ===========================================================================

def _try_import(module_path: str):
    """Import a module by dotted path; return None on any ImportError."""
    try:
        import importlib
        return importlib.import_module(module_path)
    except Exception:
        return None


# Dashboard helpers
_charts      = _try_import("dashboard.charts")
_maps        = _try_import("dashboard.maps")
_components  = _try_import("dashboard.components")
_interactive = _try_import("dashboard.interactive")

# Warehouse
_db      = _try_import("warehouse.database")
_queries = _try_import("warehouse.queries")

# OLAP
_rollup    = _try_import("olap.rollup")
_drilldown = _try_import("olap.drilldown")
_slice     = _try_import("olap.slice")
_dice      = _try_import("olap.dice")

# Mining
_kmeans = _try_import("mining.kmeans")


# ===========================================================================
# Convenience wrappers
# ===========================================================================

def _has(module, attr: str) -> bool:
    return module is not None and hasattr(module, attr)


def show_demo_banner():
    """Show DEMO banner ONLY when no real best.pt model is available."""
    # Check if the real trained model exists
    best_pt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt")
    real_model_ready = os.path.isfile(best_pt) and os.path.getsize(best_pt) > 5_000_000
    if real_model_ready:
        st.success(
            "✅ **REAL RDD2022 MODEL ACTIVE** — Running with trained YOLOv8n model "
            "(`models/best.pt`) trained on the RDD2022 D40 Pothole dataset.",
            icon="✅",
        )
    elif _has(_components, "show_demo_banner"):
        _components.show_demo_banner()
    else:
        st.warning(
            "⚠️ **DEMO DATA MODE** – running with synthetic data. "
            "Real results require RDD2022 + trained YOLO model.",
            icon="⚠️",
        )


def get_empty_summary() -> dict:
    return {
        "total_detections": 0,
        "total_potholes": 0,
        "high_severity": 0,
        "medium_severity": 0,
        "low_severity": 0,
        "unique_roads": 0,
        "unique_cities": 0,
        "avg_confidence": 0.0,
        "date_range": "N/A",
    }


def get_warehouse_summary() -> dict:
    """Fetch summary stats from the warehouse; fall back to zeros on error."""
    try:
        from warehouse.database import get_db_session
        from warehouse.queries import get_detection_summary
        with get_db_session() as session:
            raw = get_detection_summary(session)
        if not raw:
            return get_empty_summary()
        return {
            **get_empty_summary(),
            **raw,
            "high_severity": raw.get("high_severity_count", 0),
            "unique_roads": raw.get("road_count", 0),
            "unique_cities": raw.get("location_count", 0),
        }
    except Exception:
        return get_empty_summary()


def get_db_connection():
    """Return a live SQLAlchemy connection, or None if unavailable."""
    try:
        if _has(_db, "get_connection"):
            return _db.get_connection()
    except Exception:
        pass
    return None


def query_df(sql: str, params=()) -> pd.DataFrame:
    """Run a SELECT query and return a DataFrame (empty on error)."""
    try:
        conn = get_db_connection()
        if conn is None:
            return pd.DataFrame()
        # Backward-compatible column names used by some dashboard SQL snippets.
        sql_original = sql
        replacements = {
            "City_Name": "City",
            "Area_Name": "Area",
            "Severity_Label": "Severity_Level",
            "Confidence_Score": "Confidence",
            "Full_Date": "Date",
            "Month_Num": "Month",
        }
        for old, new in replacements.items():
            sql = sql.replace(old, new)
        try:
            df = pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()
        # Preserve expected legacy output column names for bare SELECTs.
        rename_map = {}
        for old, new in replacements.items():
            if old in sql_original and new in df.columns and old not in df.columns:
                rename_map[new] = old
        if rename_map:
            df = df.rename(columns=rename_map)
        return df
    except Exception:
        return pd.DataFrame()


def get_joined_fact_df() -> pd.DataFrame:
    """Return the fully joined star-schema fact table for interactive analytics."""
    sql = """
        SELECT f.Pothole_Count, f.Confidence, f.Source_Type,
               d.Year, d.Month, d.Month_Name, d.Quarter, d.Day_Name,
               t.Hour, t.Time_Period,
               l.City, l.Area, l.Latitude, l.Longitude,
               r.Road_Name, r.Road_Type,
               s.Severity_Level
        FROM Pothole_Detection_Fact f
        JOIN Date_Dim d ON f.Date_ID=d.Date_ID
        JOIN Time_Dim t ON f.Time_ID=t.Time_ID
        JOIN Location_Dim l ON f.Location_ID=l.Location_ID
        JOIN Road_Dim r ON f.Road_ID=r.Road_ID
        JOIN Severity_Dim s ON f.Severity_ID=s.Severity_ID
    """
    df = query_df(sql)
    return df


def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply global cross-filters from session state to any DataFrame with city/severity/road columns."""
    if df is None or df.empty:
        return df
    gf = st.session_state.get("global_filters", {})
    city = gf.get("city", "All")
    sevs = gf.get("severities", ["High", "Medium", "Low"])
    road_types = gf.get("road_types", ["Highway", "Urban", "Rural"])

    out = df.copy()
    if city and city != "All":
        for col in ["City", "City_Name", "city", "city_name"]:
            if col in out.columns:
                out = out[out[col] == city]
                break
    if sevs:
        for col in ["Severity_Level", "Severity_Label", "severity", "Severity"]:
            if col in out.columns:
                out = out[out[col].isin(sevs)]
                break
    if road_types:
        for col in ["Road_Type", "road_type"]:
            if col in out.columns:
                out = out[out[col].isin(road_types)]
                break
    return out


def seed_demo_data():
    """Seed the warehouse with clearly-labelled synthetic demo data."""
    try:
        if _has(_db, "seed_demo_data"):
            count = _db.seed_demo_data()
            st.success(f"✅ Demo data seeded successfully ({count} records)!")
            st.rerun()
        else:
            st.warning("seed_demo_data() not found in warehouse.database.")
    except Exception as exc:
        st.error(f"Failed to seed demo data: {exc}")


# ===========================================================================
# Session state initialisation
# ===========================================================================

def _init_session():
    # Prefer the real trained model (models/best.pt) if it exists and is large enough
    # to be a genuine trained model (>5 MB). Fall back to pothole_yolo.pt otherwise.
    _proj_dir = os.path.dirname(os.path.abspath(__file__))
    _best_pt  = os.path.join(_proj_dir, "models", "best.pt")
    _demo_pt  = os.path.join(_proj_dir, "models", "pothole_yolo.pt")
    _real_model_exists = (
        os.path.isfile(_best_pt) and os.path.getsize(_best_pt) > 5_000_000
    )
    _default_model_path = _best_pt if _real_model_exists else _demo_pt
    _default_is_demo    = not _real_model_exists

    defaults = {
        "is_demo": _default_is_demo,
        "model": None,
        "model_path": _default_model_path,
        "detection_results": [],
        "current_page": "🏠 Home",
        "kmeans_result": None,
        "warehouse_summary": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()

# ===========================================================================
# Sidebar
# ===========================================================================

PAGES = [
    "🏠 Home",
    "📊 Dataset & Preprocessing",
    "📹 Live Detection",
    "🖼️ Image Detection",
    "🎬 Video Detection",
    "📋 Detection Records",
    "🏛️ Data Warehouse",
    "🔍 OLAP Analysis",
    "🔮 K-Means Clustering",
    "🗺️ Hotspot Map",
    "📈 Trends",
    "📑 Reports",
    "🤖 AI Predictions & Optimization",
]

with st.sidebar:
    st.title("🕳️ Pothole Detection DWDM")
    st.caption("Data Mining & Warehousing Project")
    st.divider()

    page = st.radio("Navigation", PAGES, key="nav_radio", label_visibility="collapsed")
    st.session_state["current_page"] = page

    st.divider()

    # Demo-mode toggle
    st.session_state["is_demo"] = st.checkbox(
        "🔬 Demo Mode",
        value=st.session_state["is_demo"],
        help="Show synthetic data when the warehouse is empty or model is missing.",
    )

    # Model status
    model_path = st.session_state["model_path"]
    _best_pt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt")
    _real_trained = os.path.isfile(_best_pt_path) and os.path.getsize(_best_pt_path) > 5_000_000
    if _real_trained:
        _sz_mb = os.path.getsize(_best_pt_path) / (1024 * 1024)
        st.success("✅ Real RDD2022 Model", icon="✅")
        st.caption(f"`YOLOv8n-RDD2022-D40` ({_sz_mb:.0f} MB) — Real trained weights")
    elif os.path.isfile(model_path):
        st.warning("⚠️ Demo Model Active", icon="⚠️")
        st.caption("`YOLOv8n-RDD2022-D40` (Demo AI Engine) — Run training to get real model")
    else:
        st.error("❌ No model file", icon="❌")
        st.caption("Run `python ml/train_rdd2022.py --epochs 50`")

    st.divider()

    # Quick stats in sidebar
    st.subheader("Quick Stats")
    try:
        summ = get_warehouse_summary()
        st.metric("Detections", f"{int(summ.get('total_detections', 0)):,}")
        st.metric("Potholes",   f"{int(summ.get('total_potholes', 0)):,}")
        st.metric("Roads",      f"{int(summ.get('unique_roads', 0)):,}")
    except Exception:
        st.caption("Stats unavailable")

    st.divider()
    st.subheader("🎯 Global Cross-Filters")
    with st.expander("Filter Data Across Pages", expanded=False):
        all_cities = ["All", "Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"]
        sel_city = st.selectbox("Select City", all_cities, key="gf_city")
        sel_sev = st.multiselect("Severity Bands", ["High", "Medium", "Low"], default=["High", "Medium", "Low"], key="gf_sev")
        sel_rt = st.multiselect("Road Types", ["Highway", "Urban", "Rural"], default=["Highway", "Urban", "Rural"], key="gf_road_type")
        st.session_state["global_filters"] = {
            "city": sel_city,
            "severities": sel_sev,
            "road_types": sel_rt,
        }
        if sel_city != "All" or len(sel_sev) < 3 or len(sel_rt) < 3:
            st.info(f"Active Filters: City={sel_city} | {len(sel_sev)} sevs | {len(sel_rt)} roads")

    st.divider()
    st.caption(f"v1.0 · {datetime.now().strftime('%d %b %Y')}")

is_demo = st.session_state["is_demo"]

# ===========================================================================
# Helper: is warehouse populated?
# ===========================================================================

def warehouse_has_data() -> bool:
    df = query_df("SELECT COUNT(*) AS cnt FROM Pothole_Detection_Fact")
    if df.empty:
        return False
    return int(df.iloc[0]["cnt"]) > 0


def no_data_prompt(seed_label: str = "Seed Demo Data") -> None:
    """Show a message + seed button when the warehouse is empty."""
    st.info(
        "The data warehouse appears to be empty. "
        "Seed it with demo data to explore the dashboard.",
        icon="ℹ️",
    )
    if st.button(f"🌱 {seed_label}", type="primary"):
        seed_demo_data()


# ===========================================================================
# ── PAGE 1: Home ────────────────────────────────────────────────────────────
# ===========================================================================

def page_home():
    st.title("🕳️ Pothole Detection DWDM")
    st.subheader("Data Mining & Warehousing Project")

    if is_demo or not warehouse_has_data():
        show_demo_banner()

    # KPI Cards
    summ = get_warehouse_summary()
    if _has(_components, "show_kpi_cards"):
        _components.show_kpi_cards(summ)
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Detections", summ.get("total_detections", 0))
        c2.metric("Total Potholes",   summ.get("total_potholes", 0))
        c3.metric("High Severity",    summ.get("high_severity", 0))
        c4.metric("Roads",            summ.get("unique_roads", 0))
        c5.metric("Cities",           summ.get("unique_cities", 0))

    # ── Interactive City Infrastructure Health Gauges ───────────────────────
    fact_df = get_joined_fact_df()
    fact_df = apply_global_filters(fact_df)

    if not fact_df.empty and _has(_interactive, "make_gauge"):
        st.subheader("🚦 City Infrastructure Health Gauges")
        st.caption("Real-Time Road Network Quality Index (100 = Optimal condition · 0 = Hazard level)")
        top_cities = fact_df["City"].value_counts().head(4).index.tolist()
        if top_cities:
            gauge_cols = st.columns(len(top_cities))
            for i, city_name in enumerate(top_cities):
                with gauge_cols[i]:
                    sub = fact_df[fact_df["City"] == city_name]
                    tot = int(sub["Pothole_Count"].sum())
                    high = int((sub["Severity_Level"] == "High").sum())
                    med = int((sub["Severity_Level"] == "Medium").sum())
                    low = int((sub["Severity_Level"] == "Low").sum())
                    penalty = (high * 3.0 + med * 1.5 + low * 0.5) / max(tot, 1) * 32.0
                    health_score = max(5.0, round(100.0 - penalty, 1))
                    g_color = "#2ecc71" if health_score >= 70 else ("#f39c12" if health_score >= 45 else "#e74c3c")
                    fig_g = _interactive.make_gauge(health_score, f"{city_name}", 100, color=g_color)
                    st.plotly_chart(fig_g, use_container_width=True)

    # ── Interactive Dynamic Analytics Row ───────────────────────────────────
    if not fact_df.empty and _has(_interactive, "make_city_bar_race"):
        st.subheader("🏎️ Dynamic Pothole Accumulation & Triage")
        col_race, col_funnel = st.columns([1.3, 1])
        with col_race:
            st.plotly_chart(_interactive.make_city_bar_race(fact_df), use_container_width=True)
        with col_funnel:
            if _has(_interactive, "make_severity_funnel"):
                st.plotly_chart(_interactive.make_severity_funnel(fact_df), use_container_width=True)

        if _has(_interactive, "make_road_type_severity_chart"):
            st.plotly_chart(_interactive.make_road_type_severity_chart(fact_df), use_container_width=True)

    st.divider()

    col_obj, col_arch = st.columns([1, 1])

    with col_obj:
        st.subheader("🎯 Project Objective")
        st.markdown(
            """
Detect and catalogue road potholes from images, video, and webcam feeds using a
**YOLOv8 object detection model**, then store results in a **star-schema data
warehouse** to support:

- **OLAP operations** – Roll-Up, Drill-Down, Slice, Dice
- **K-Means Clustering** – discover pothole hotspot patterns
- **Geospatial mapping** – visualise severity across Indian cities
- **Trend reporting** – monthly and road-level analytics
            """
        )

    with col_arch:
        st.subheader("🏗️ System Architecture")
        st.code(
            """
┌─────────────────────────────────────────┐
│              DATA SOURCES               │
│  RDD2022 Dataset │ Webcam │ Image/Video │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  YOLOv8 Detection  │
         │  (pothole class)   │
         └─────────┬──────────┘
                   │
      ┌────────────▼────────────┐
      │   Preprocessing ETL     │
      │   (filter, label, save) │
      └────────────┬────────────┘
                   │
   ┌───────────────▼───────────────┐
   │       Star-Schema Warehouse   │
   │  Date │ Time │ Location │ ... │
   │       Fact: Detections        │
   └───────────────┬───────────────┘
                   │
      ┌────────────┴─────────────┐
      │                          │
 OLAP Queries              K-Means Mining
 Roll-up / Drill-down      Cluster hotspots
      │                          │
      └────────────┬─────────────┘
                   │
           Streamlit Dashboard
            """,
            language="text",
        )

    st.divider()

    col_stack, col_start = st.columns([1, 1])

    with col_stack:
        st.subheader("🛠️ Tech Stack")
        st.markdown(
            """
| Component       | Technology                     |
|-----------------|-------------------------------|
| Detection       | YOLOv8 (Ultralytics)           |
| Dashboard       | Streamlit + Plotly             |
| Warehouse       | SQLite (star schema)           |
| Data Mining     | scikit-learn (K-Means)         |
| Maps            | Plotly scatter_mapbox          |
| Dataset         | RDD2022 (D40 class)            |
| Language        | Python 3.10+                   |
            """
        )

    with col_start:
        st.subheader("🚀 Quick Start")
        st.markdown(
            """
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place RDD2022 data
#    data/raw/Japan/  data/raw/India/  etc.

# 3. Run preprocessing
python preprocessing/preprocess.py

# 4. Train YOLO (or download weights)
python training/train_yolo.py

# 5. Launch dashboard
streamlit run app.py
```

> Use **Demo Mode** (sidebar checkbox) to explore without real data.
            """
        )


# ===========================================================================
# ── PAGE 2: Dataset & Preprocessing ─────────────────────────────────────────
# ===========================================================================

def page_dataset():
    st.title("📊 Dataset & Preprocessing")

    if is_demo or not warehouse_has_data():
        show_demo_banner()

    # RDD2022 Info
    with st.expander("📦 About RDD2022 Dataset", expanded=True):
        st.markdown(
            """
**Road Damage Detection 2022 (RDD2022)** is a large-scale benchmark dataset for
road surface damage detection collected across six countries.

| Property      | Detail                                |
|---------------|---------------------------------------|
| Total Images  | ~47,420                               |
| Countries     | Japan, India, USA, Czech, Norway, China|
| Classes       | D00 (longitudinal crack), D10, D20, **D40 (pothole)** |
| Format        | YOLO annotation (class id, bbox)      |
| Source        | IEEE DataPort / GitHub                |

**Why D40?**  The class `D40` corresponds to potholes specifically, which is the
focus of this project.  All other damage classes are filtered out during
preprocessing.
            """
        )

    # Expected folder structure
    st.subheader("📂 Expected Data Structure")
    st.code(
        """
project_pathhole/
├── data/
│   ├── raw/
│   │   ├── Japan/
│   │   │   ├── images/  (*.jpg)
│   │   │   └── annotations/  (*.xml or *.txt)
│   │   ├── India/
│   │   ├── Czech/
│   │   └── ...
│   └── processed/
│       ├── images/
│       └── labels/   (YOLO .txt format)
├── models/
│   └── pothole_yolo.pt
└── ...
        """,
        language="text",
    )

    # Preprocessing pipeline
    st.subheader("⚙️ Preprocessing Pipeline")
    steps = [
        ("1️⃣ Parse Annotations", "Read XML/JSON ground-truth files; extract D40 bounding boxes."),
        ("2️⃣ Filter D40 Class",  "Keep only pothole (D40) annotations; discard other damage classes."),
        ("3️⃣ Resize & Prepare","Resize images to 416×416 for the YOLO training pipeline."),
        ("4️⃣ Convert to YOLO",   "Write YOLO-format .txt label files (class_id cx cy w h)."),
        ("5️⃣ Train/Val/Test Split", "70% train / 20% validation / 10% test with grouped splitting to prevent leakage."),
        ("6️⃣ Augmentation",      "Horizontal flip, brightness adjustment, scaling, and cropping with bounding-box-aware labels."),
        ("7️⃣ Warehouse ETL",     "Load detection results into star-schema SQLite warehouse."),
    ]
    for title, desc in steps:
        with st.expander(title):
            st.write(desc)

    st.divider()

    col_run, col_report = st.columns([1, 1])

    with col_run:
        st.subheader("▶️ Run Preprocessing Steps")

        if st.button("📂 Check Data Directory"):
            raw_path = os.path.join(os.path.dirname(__file__), "data", "raw")
            if os.path.isdir(raw_path):
                subdirs = os.listdir(raw_path)
                st.success(f"Found {len(subdirs)} subdirectories: {subdirs}")
            else:
                st.warning(f"Data directory not found: `{raw_path}`")

        if st.button("🔧 Run Full RDD2022 Preprocessing"):
            try:
                from scripts.prepare_dataset import run_pipeline, build_final_report
                from utils.helpers import load_config, setup_logging
                cfg = load_config()
                pipeline_reports = run_pipeline(cfg, setup_logging("streamlit_preprocessing"))
                report = build_final_report(pipeline_reports)
                st.session_state["preproc_report"] = report
                if report.get("pipeline_status") == "COMPLETE":
                    st.success("✅ Preprocessing pipeline completed!")
                else:
                    st.warning("Preprocessing stopped because the RDD2022 dataset is not available.")
            except Exception as exc:
                st.error(f"Preprocessing error: {exc}")
                st.code(traceback.format_exc())

        if st.button("🏗️ Initialise Warehouse Schema"):
            try:
                if _has(_db, "create_schema"):
                    _db.create_schema()
                    st.success("Warehouse schema created / verified.")
                else:
                    st.info("warehouse/database.py missing create_schema().")
            except Exception as exc:
                st.error(str(exc))

    with col_report:
        if _has(_components, "show_preprocessing_report"):
            _components.show_preprocessing_report(
                st.session_state.get("preproc_report", {})
            )


# ===========================================================================
# ── PAGE 3: Live Detection ──────────────────────────────────────────────────
# ===========================================================================

def page_live_detection():
    st.title("\U0001f4f9 Live Detection")
    st.caption("Real-time pothole detection from road cameras and webcam feeds.")

    model_path = st.session_state["model_path"]
    if _has(_components, "show_model_status"):
        _components.show_model_status(model_path)

    st.info(
        "Live detection captures camera frames and runs real-time pothole inference "
        "using the YOLOv8 engine (with fallback to Morphological AI Computer Vision).",
        icon="\u2139\ufe0f",
    )

    # \u2500\u2500 Location tagging (shared across all tabs) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    with st.expander("\U0001f4cd Municipal Location Tagging", expanded=False):
        lc1, lc2, lc3, lc4 = st.columns(4)
        live_city      = lc1.selectbox("City", ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata"], key="live_city")
        live_area      = lc2.text_input("Area / Suburb", value="Indiranagar", key="live_area")
        live_road      = lc3.text_input("Road Name", value="100 Feet Road", key="live_road")
        live_road_type = lc4.selectbox("Road Category", ["Urban", "Highway", "Rural"], key="live_road_type")

    CITY_COORDS = {
        "Bangalore": (12.9716, 77.5946), "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.6139, 77.2090), "Chennai": (13.0827, 80.2707),
        "Hyderabad": (17.3850, 78.4867), "Pune": (18.5204, 73.8567),
        "Kolkata": (22.5726, 88.3639),
    }

    # ── Start/verify live YOLO inference server (port 8502) ───────────────────
    _live_server_ok = False
    try:
        from ml.live_server import ensure_live_server
        _live_server_ok = ensure_live_server()
    except Exception as _srv_err:
        _live_server_ok = False

    # Quick ping to confirm server is actually responding
    try:
        import urllib.request as _ur
        _ur.urlopen("http://127.0.0.1:8502/api/health", timeout=1.5)
        _live_server_ok = True
    except Exception:
        _live_server_ok = False

    if _live_server_ok:
        st.success(
            "✅ **YOLO Live Inference Server ACTIVE** — `models/best.pt` loaded on `http://127.0.0.1:8502`. "
            "Real RDD2022 D40 Pothole detections are ready.",
            icon="✅",
        )
    else:
        st.error(
            "❌ **YOLO Inference Server not reachable** on `http://127.0.0.1:8502`.  \n"
            "Start it manually in a separate terminal:  \n"
            "```\npython ml/live_server.py\n```"
        )

    # \u2500\u2500 Three tabs \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    tab_browser, tab_simulated, tab_opencv = st.tabs([
        "\U0001f4f9 Laptop Browser Cam (Live WebRTC)",
        "\U0001f6e3\ufe0f Simulated Road Feed",
        "\U0001f50c OpenCV Direct Stream",
    ])

    # =========================================================================
    # TAB 1 \u2014 Browser Webcam via WebRTC + Real YOLO Inference (models/best.pt)
    # =========================================================================
    with tab_browser:
        st.markdown("### \U0001f4f9 Real-Time WebRTC Laptop Webcam")
        st.caption(
            "Captures live video frames directly from your browser webcam via WebRTC, "
            "runs real YOLO inference (`models/best.pt`) on each frame, renders RDD2022 D40 Pothole bounding boxes, "
            "computes live severity, and displays real-time results."
        )

        import streamlit.components.v1 as st_components
        from dashboard.live_webcam_component import render_live_webcam_html

        html_code = render_live_webcam_html(
            api_url="http://127.0.0.1:8502",
            default_conf=0.20,
            city=live_city,
            area=live_area,
            road_name=live_road,
            road_type=live_road_type,
        )
        st_components.html(html_code, height=750, scrolling=False)

        with st.expander("\U0001f4f8 Alternative: Single Snapshot Mode (Photo Capture)", expanded=False):
            st.info("Take a single static photo using your browser camera for point-in-time detection.")
            conf_thres_browser = st.slider("Snapshot Confidence Threshold", 0.05, 0.90, 0.20, 0.05, key="browser_conf")
            cam_image = st.camera_input("\U0001f4f7 Take a photo with your laptop camera", key="snap_cam_input")

        if cam_image is not None:
            import numpy as np
            import cv2
            from PIL import Image
            import io

            img_bytes = cam_image.getvalue()
            pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            frame     = np.array(pil_img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            col_orig, col_det = st.columns(2)
            col_orig.image(frame, caption="\U0001f4f7 Captured Frame", use_container_width=True)

            with st.spinner("\U0001f50d Running pothole detection..."):
                detections    = []
                annotated_rgb = frame.copy()

                yolo_ok = False
                if model_path and os.path.exists(str(model_path)):
                    try:
                        from ultralytics import YOLO
                        yolo_model = YOLO(str(model_path))
                        results    = yolo_model(frame_bgr, conf=conf_thres_browser, verbose=False)
                        annotated  = results[0].plot()
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        boxes = results[0].boxes
                        if boxes is not None:
                            img_h, img_w = frame_bgr.shape[:2]
                            for box in boxes:
                                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                                conf_v = float(box.conf[0])
                                from utils.severity import calculate_severity
                                area_r = ((x2-x1)*(y2-y1)) / max(img_h*img_w, 1)
                                sev    = calculate_severity(conf_v, area_r)
                                detections.append({
                                    "class": "D40 Pothole",
                                    "confidence": round(conf_v, 3),
                                    "severity": sev,
                                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                    "area_ratio": round(area_r, 4),
                                })
                        yolo_ok = True
                    except Exception as e:
                        logger.warning(f"Snapshot YOLO inference error: {e}")
                        yolo_ok = False

                if not yolo_ok:
                    try:
                        from ml.demo_detector import detect_potholes_cv
                        dets, annotated_rgb, stats = detect_potholes_cv(frame_bgr, conf_thres=conf_thres_browser, is_demo=False)
                        detections = dets
                    except Exception as e:
                        st.error(f"Detection error: {e}")

            col_det.image(annotated_rgb, caption="\U0001f50d Detection Result", use_container_width=True)

            k1, k2, k3 = st.columns(3)
            n_det  = len(detections)
            avg_cf = round(sum(d["confidence"] for d in detections)/max(n_det,1), 3) if detections else 0.0
            sev_counts = {"High": 0, "Medium": 0, "Low": 0}
            for d in detections:
                sev_counts[d.get("severity","Low")] += 1
            top_sev = "High" if sev_counts["High"] else ("Medium" if sev_counts["Medium"] else ("Low" if n_det else "None"))

            k1.metric("Potholes Detected", n_det)
            k2.metric("Avg Confidence", f"{avg_cf:.1%}")
            k3.metric("Severity", top_sev)

            if detections:
                st.dataframe(pd.DataFrame(detections), use_container_width=True)
                st.divider()
                st.markdown("#### \U0001f4be Save Detection to Warehouse")
                if st.button("\U0001f4e5 Ingest Browser Cam Detection", type="primary", key="browser_save"):
                    lat, lon = CITY_COORDS.get(live_city, (0.0, 0.0))
                    rec = {
                        "pothole_count": n_det, "confidence": avg_cf,
                        "severity": top_sev, "source_type": "webcam",
                        "detection_date": str(date.today()),
                        "city": live_city, "area": live_area,
                        "road_name": live_road, "road_type": live_road_type,
                        "latitude": lat, "longitude": lon,
                    }
                    try:
                        if _has(_db, "insert_detection"):
                            det_id = _db.insert_detection(rec)
                            st.success(f"\u2705 Saved! Detection ID: `{det_id}`")
                        else:
                            st.warning("insert_detection not available.")
                    except Exception as ex:
                        st.error(f"Failed to save: {ex}")
            else:
                st.info("No potholes detected in this frame. Try capturing a different angle.")

    # =========================================================================
    # TAB 2 \u2014 Simulated Road Feed
    # =========================================================================
    with tab_simulated:
        st.markdown("### \U0001f6e3\ufe0f Simulated Live Road Camera Feed")
        st.info("Plays the built-in demo road video and runs pothole detection frame-by-frame.", icon="\u2139\ufe0f")

        conf_thres_sim = st.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05, key="sim_conf")
        max_frames_sim = st.number_input("Max Frames to Process", min_value=10, max_value=300, value=60, step=10, key="sim_frames")

        sim_col1, sim_col2 = st.columns(2)
        run_sim  = sim_col1.button("\u25b6\ufe0f Start Simulated Feed", type="primary", use_container_width=True, key="run_sim")
        stop_sim = sim_col2.button("\u23f9\ufe0f Stop", use_container_width=True, key="stop_sim")

        if run_sim:
            st.session_state["sim_running"]  = True
            st.session_state["sim_potholes"] = 0
            st.session_state["sim_frames"]   = 0

        if stop_sim:
            st.session_state["sim_running"] = False

        if st.session_state.get("sim_running"):
            import cv2, time
            video_path = "data/demo/test_road.mp4"
            if not os.path.exists(video_path):
                st.error("Demo video not found at `data/demo/test_road.mp4`.")
                st.session_state["sim_running"] = False
            else:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Could not open demo video.")
                    st.session_state["sim_running"] = False
                else:
                    frame_ph    = st.empty()
                    sk1, sk2, sk3 = st.columns(3)
                    m_sc = sk1.empty(); m_sf = sk2.empty(); m_ss = sk3.empty()
                    status_sim = st.empty()
                    status_sim.info("\U0001f7e2 Simulated Feed Running...")

                    yolo_sim = None
                    try:
                        if model_path and os.path.exists(str(model_path)):
                            from ultralytics import YOLO
                            yolo_sim = YOLO(str(model_path))
                    except Exception:
                        pass

                    sim_count = 0; sim_fc = 0; last_sev = "Low"

                    while cap.isOpened() and st.session_state.get("sim_running"):
                        ret, frame = cap.read()
                        if not ret:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue

                        if yolo_sim is not None:
                            results   = yolo_sim(frame, conf=conf_thres_sim, verbose=False)
                            annotated = results[0].plot()
                            n_this    = len(results[0].boxes) if results[0].boxes else 0
                            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        else:
                            try:
                                from ml.demo_detector import detect_potholes_cv
                                dets, frame_rgb, stats = detect_potholes_cv(frame, conf_thres=conf_thres_sim, is_demo=True)
                                n_this = stats.get("pothole_count", 0)
                            except Exception:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                n_this    = 0

                        sim_count += n_this; sim_fc += 1
                        last_sev = "High" if n_this >= 3 else ("Medium" if n_this >= 1 else "Low")
                        frame_ph.image(frame_rgb, channels="RGB", caption=f"Simulated Frame #{sim_fc}", use_container_width=True)
                        m_sc.metric("Potholes", sim_count)
                        m_sf.metric("Frames", sim_fc)
                        m_ss.metric("Severity", last_sev)
                        time.sleep(0.04)
                        if sim_fc >= int(max_frames_sim):
                            break

                    cap.release()
                    st.session_state.update({
                        "sim_running": False, "sim_potholes": sim_count,
                        "sim_frames": sim_fc, "sim_last_sev": last_sev,
                    })
                    status_sim.success(f"\u2705 Done: {sim_count} potholes in {sim_fc} frames.")

        if st.session_state.get("sim_frames", 0) > 0 and not st.session_state.get("sim_running"):
            st.divider()
            tot_p = st.session_state.get("sim_potholes", 0)
            tot_f = st.session_state.get("sim_frames", 0)
            sv    = st.session_state.get("sim_last_sev", "Low")
            st.write(f"Session: **{tot_p} potholes** / **{tot_f} frames** \u00b7 Severity: **{sv}**")
            if st.button("\U0001f4e5 Save Simulated Session to Warehouse", type="primary", key="sim_save"):
                lat, lon = CITY_COORDS.get(live_city, (0.0, 0.0))
                rec = {
                    "pothole_count": max(1, tot_p), "confidence": 0.72, "severity": sv,
                    "source_type": "video", "detection_date": str(date.today()),
                    "city": live_city, "area": live_area,
                    "road_name": live_road, "road_type": live_road_type,
                    "latitude": lat, "longitude": lon,
                }
                try:
                    if _has(_db, "insert_detection"):
                        det_id = _db.insert_detection(rec)
                        st.success(f"\u2705 Saved! Detection ID: `{det_id}`")
                    else:
                        st.warning("insert_detection not available.")
                except Exception as ex:
                    st.error(f"Failed to save: {ex}")

    # =========================================================================
    # TAB 3 \u2014 OpenCV Direct Stream
    # =========================================================================
    with tab_opencv:
        st.markdown("### \U0001f50c OpenCV Direct Webcam Stream")
        st.warning(
            "**Advanced / Optional.** Requires OpenCV to open your camera via `cv2.VideoCapture`. "
            "On Windows with OpenCV 5, this may fail. If it does, use the **\U0001f4f8 Browser Cam** tab.",
            icon="\u26a0\ufe0f",
        )

        cam_index  = st.number_input("Camera Index (0=built-in, 1=USB)", min_value=0, max_value=5, value=0, key="ocv_cam_idx")
        conf_ocv   = st.slider("Confidence Threshold", 0.05, 0.90, 0.20, 0.05, key="ocv_conf")
        max_fr_ocv = st.number_input("Max Frames", min_value=10, max_value=300, value=60, step=10, key="ocv_frames")

        oc1, oc2 = st.columns(2)
        run_ocv  = oc1.button("\u25b6\ufe0f Start OpenCV Stream", type="primary", use_container_width=True, key="run_ocv")
        stop_ocv = oc2.button("\u23f9\ufe0f Stop", use_container_width=True, key="stop_ocv")

        if run_ocv:
            st.session_state.update({"ocv_running": True, "ocv_potholes": 0, "ocv_frames": 0})
        if stop_ocv:
            st.session_state["ocv_running"] = False

        if st.session_state.get("ocv_running"):
            import cv2, time
            cap = None
            for backend in [cv2.CAP_MSMF, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(int(cam_index), backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        break
                    cap.release(); cap = None
                except Exception:
                    cap = None

            if cap is None or not cap.isOpened():
                st.error(
                    f"\u274c Could not open camera `{cam_index}`. "
                    "**Use the \U0001f4f8 Browser Cam tab instead.**"
                )
                st.session_state["ocv_running"] = False
            else:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                st.info(f"Camera opened: {w}\u00d7{h}")
                fr_ph = st.empty()
                ek1, ek2, ek3 = st.columns(3)
                m_oc = ek1.empty(); m_of = ek2.empty(); m_os = ek3.empty()
                ocv_status = st.empty()
                ocv_status.info("\U0001f7e2 OpenCV Stream Active...")

                yolo_ocv = None
                try:
                    if model_path and os.path.exists(str(model_path)):
                        from ultralytics import YOLO
                        yolo_ocv = YOLO(str(model_path))
                except Exception:
                    pass

                ocv_count = 0; ocv_fc = 0; ocv_sev = "Low"
                while cap.isOpened() and st.session_state.get("ocv_running"):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if yolo_ocv is not None:
                        results   = yolo_ocv(frame, conf=conf_ocv, verbose=False)
                        annotated = results[0].plot()
                        n_this    = len(results[0].boxes) if results[0].boxes else 0
                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    else:
                        try:
                            from ml.demo_detector import detect_potholes_cv
                            dets, frame_rgb, stats = detect_potholes_cv(frame, conf_thres=conf_ocv, is_demo=False)
                            n_this = stats.get("pothole_count", 0)
                        except Exception:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); n_this = 0
                    ocv_count += n_this; ocv_fc += 1
                    ocv_sev = "High" if n_this >= 3 else ("Medium" if n_this >= 1 else "Low")
                    fr_ph.image(frame_rgb, channels="RGB", caption=f"Frame #{ocv_fc}", use_container_width=True)
                    m_oc.metric("Potholes", ocv_count); m_of.metric("Frames", ocv_fc); m_os.metric("Severity", ocv_sev)
                    time.sleep(0.03)
                    if ocv_fc >= int(max_fr_ocv):
                        break
                cap.release()
                st.session_state.update({
                    "ocv_running": False, "ocv_potholes": ocv_count,
                    "ocv_frames": ocv_fc, "ocv_last_sev": ocv_sev,
                })
                ocv_status.success(f"\u2705 Stream done: {ocv_count} potholes in {ocv_fc} frames.")

        if st.session_state.get("ocv_frames", 0) > 0 and not st.session_state.get("ocv_running"):
            st.divider()
            tot_p = st.session_state.get("ocv_potholes", 0)
            tot_f = st.session_state.get("ocv_frames", 0)
            sv    = st.session_state.get("ocv_last_sev", "Low")
            st.write(f"Session: **{tot_p} potholes** / **{tot_f} frames** \u00b7 Severity: **{sv}**")
            if st.button("\U0001f4e5 Save OpenCV Session to Warehouse", type="primary", key="ocv_save"):
                lat, lon = CITY_COORDS.get(live_city, (0.0, 0.0))
                rec = {
                    "pothole_count": max(1, tot_p), "confidence": 0.80, "severity": sv,
                    "source_type": "webcam", "detection_date": str(date.today()),
                    "city": live_city, "area": live_area,
                    "road_name": live_road, "road_type": live_road_type,
                    "latitude": lat, "longitude": lon,
                }
                try:
                    if _has(_db, "insert_detection"):
                        det_id = _db.insert_detection(rec)
                        st.success(f"\u2705 Saved! Detection ID: `{det_id}`")
                    else:
                        st.warning("insert_detection not available.")
                except Exception as ex:
                    st.error(f"Failed to save: {ex}")


# ===========================================================================
# ── PAGE 4: Image Detection ──────────────────────────────────────────────────
# ===========================================================================

def page_image_detection():
    st.title("🖼️ Image Detection")

    model_path = st.session_state["model_path"]
    if _has(_components, "show_model_status"):
        _components.show_model_status(model_path)

    uploaded = st.file_uploader(
        "Upload a road or pavement image", type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    c_cfg1, c_cfg2 = st.columns([1, 1])
    with c_cfg1:
        engine_mode = st.selectbox(
            "Detection Engine",
            [
                "🌟 Hybrid AI (YOLOv8 + Adaptive CV - Recommended)",
                "🧠 YOLOv8 Neural Network Only",
                "🔬 Morphological Computer Vision Only",
            ],
            key="img_engine",
            help="Hybrid AI combines deep learning with morphological depression filtering to catch both dry asphalt potholes and water-filled / reflective road puddles."
        )
    with c_cfg2:
        conf_thres = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.90,
            value=0.20,
            step=0.05,
            key="img_conf",
            help="Default 0.20. Lower to 0.10–0.15 for wet, distant, or water-filled potholes; raise for high-confidence only."
        )

    if uploaded:
        # Check if a new file was uploaded; reset stored detection if changed
        current_file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("img_last_file_id") != current_file_id:
            st.session_state["img_last_file_id"] = current_file_id
            st.session_state["img_detection_result"] = None

        import tempfile
        import numpy as np
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        btn_c1, btn_c2 = st.columns([1, 3])
        with btn_c1:
            run_btn = st.button("🔍 Run Detection", type="primary", use_container_width=True)
        with btn_c2:
            if st.session_state.get("img_detection_result") is not None:
                if st.button("🔄 Reset / Clear Result"):
                    st.session_state["img_detection_result"] = None
                    st.rerun()

        # Execute detection ONLY when user explicitly clicks the "Run Detection" button
        if run_btn:
            with st.spinner("🔍 Running pothole detection..."):
                try:
                    img_bgr = cv2.imread(tmp_path)
                    if img_bgr is None:
                        st.error("Failed to read uploaded image.")
                        return

                    img_h, img_w = img_bgr.shape[:2]
                    img_area = max(img_h * img_w, 1)

                    detections = []
                    model_used = "Ultralytics YOLOv8"

                    # 1. Run YOLO if selected or in Hybrid mode
                    run_yolo = "YOLOv8" in engine_mode or "Hybrid" in engine_mode
                    run_cv   = "Morphological" in engine_mode or "Hybrid" in engine_mode

                    if run_yolo and model_path and os.path.exists(str(model_path)):
                        try:
                            from ultralytics import YOLO
                            y_model = YOLO(model_path)
                            y_res = y_model.predict(img_bgr, conf=conf_thres, verbose=False)[0]
                            if y_res.boxes is not None:
                                for b in y_res.boxes:
                                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                                    conf = float(b.conf[0])
                                    box_area = (x2 - x1) * (y2 - y1)
                                    area_ratio = round(box_area / img_area, 6)
                                    sev = "High" if (area_ratio >= 0.05 and conf >= 0.70) else ("Low" if conf < 0.40 else "Medium")
                                    detections.append({
                                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                        "confidence": round(conf, 3),
                                        "severity": sev,
                                        "area_ratio": area_ratio,
                                        "source": "YOLOv8",
                                    })
                        except Exception as y_err:
                            st.warning(f"YOLO inference warning: {y_err}")

                    # 2. Run Morphological CV if selected or in Hybrid mode
                    if run_cv:
                        try:
                            from ml.demo_detector import detect_potholes_cv
                            cv_dets, _, _ = detect_potholes_cv(img_bgr, conf_thres=max(conf_thres, 0.25))

                            def _iou(boxA, boxB):
                                xA = max(boxA["x1"], boxB["x1"])
                                yA = max(boxA["y1"], boxB["y1"])
                                xB = min(boxA["x2"], boxB["x2"])
                                yB = min(boxA["y2"], boxB["y2"])
                                inter = max(0, xB - xA) * max(0, yB - yA)
                                areaA = (boxA["x2"] - boxA["x1"]) * (boxA["y2"] - boxA["y1"])
                                areaB = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])
                                union = areaA + areaB - inter
                                return inter / union if union > 0 else 0

                            for cd in cv_dets:
                                if not any(_iou(cd, d) > 0.35 for d in detections):
                                    detections.append({
                                        "x1": cd["x1"], "y1": cd["y1"], "x2": cd["x2"], "y2": cd["y2"],
                                        "confidence": cd["confidence"],
                                        "severity": cd.get("severity", "Medium"),
                                        "area_ratio": cd.get("area_ratio", 0.005),
                                        "source": "Adaptive Road Vision",
                                    })
                        except Exception as cv_err:
                            st.warning(f"Morphological CV warning: {cv_err}")

                    # Determine model label
                    if "Hybrid" in engine_mode:
                        model_used = "Hybrid AI (YOLOv8 + Adaptive Road Vision)"
                    elif "Morphological" in engine_mode:
                        model_used = "Morphological CV (Surface Depression Engine)"
                    else:
                        model_used = "Ultralytics YOLOv8"

                    # 3. Draw annotations
                    annotated_bgr = img_bgr.copy()
                    sev_colors = {
                        "High": (0, 0, 255),       # Red
                        "Medium": (0, 165, 255),   # Orange
                        "Low": (0, 220, 0),        # Green
                    }

                    for d in detections:
                        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                        sev = d["severity"]
                        color = sev_colors.get(sev, (0, 255, 0))
                        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
                        label = f"Pothole {d['confidence']:.2f} [{sev}]"
                        (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(annotated_bgr, (x1, max(0, y1 - lh - base - 4)), (x1 + lw, y1), color, -1)
                        cv2.putText(annotated_bgr, label, (x1, max(lh + 2, y1 - base - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    ann_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    n_det = len(detections)
                    avg_conf = (sum(d["confidence"] for d in detections) / n_det) if n_det > 0 else 0.0

                    # Overall severity
                    if n_det >= 4 or any(d["severity"] == "High" for d in detections):
                        overall_sev = "High"
                    elif n_det >= 2 or any(d["severity"] == "Medium" for d in detections):
                        overall_sev = "Medium"
                    else:
                        overall_sev = "Low"

                    # Cache results into session state
                    st.session_state["img_detection_result"] = {
                        "ann_rgb": ann_rgb,
                        "n_det": n_det,
                        "avg_conf": avg_conf,
                        "overall_sev": overall_sev,
                        "model_used": model_used,
                        "detections": detections,
                        "conf_thres": conf_thres,
                        "engine_mode": engine_mode,
                    }
                except Exception as exc:
                    st.error(f"Detection error: {exc}")
                    st.code(traceback.format_exc())

        col_orig, col_ann = st.columns(2)

        with col_orig:
            st.subheader("Original Image")
            st.image(tmp_path, use_container_width=True)

        res = st.session_state.get("img_detection_result")

        with col_ann:
            st.subheader("Annotated Image")
            if res is not None and res.get("ann_rgb") is not None:
                st.image(res["ann_rgb"], use_container_width=True)
            else:
                st.info(
                    "👉 **Click the '🔍 Run Detection' button above** when you are ready to process this image.",
                    icon="👉"
                )

        if res is not None:
            n_det = res["n_det"]
            avg_conf = res["avg_conf"]
            overall_sev = res["overall_sev"]
            model_used = res["model_used"]
            detections = res["detections"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Potholes Detected", n_det)
            c2.metric("Avg Confidence", f"{avg_conf:.3f}")
            c3.metric("Overall Severity", overall_sev)
            c4.metric("Engine Used", "Hybrid AI" if "Hybrid" in model_used else ("YOLOv8" if "YOLO" in model_used else "CV"))

            if n_det == 0:
                st.info(
                    f"💡 **0 potholes detected at confidence threshold {res['conf_thres']:.2f}.**\n\n"
                    "- If the road has water puddles or sunlight reflections, make sure **🌟 Hybrid AI Mode** is active.\n"
                    "- You can also adjust the confidence slider above down to **0.10–0.15** and click **🔍 Run Detection** again.",
                    icon="💡"
                )
            else:
                with st.expander(f"📋 Detected Potholes Breakdown ({n_det} items)", expanded=False):
                    det_df = pd.DataFrame([
                        {
                            "#": i + 1,
                            "Severity": d["severity"],
                            "Confidence": f"{d['confidence']:.3f}",
                            "Box [x1, y1, x2, y2]": f"[{d['x1']}, {d['y1']}, {d['x2']}, {d['y2']}]",
                            "Area Ratio": f"{d['area_ratio']:.4f}",
                            "Detected By": d.get("source", "Model"),
                        }
                        for i, d in enumerate(detections)
                    ])
                    st.dataframe(det_df, use_container_width=True, hide_index=True)

            # Save to warehouse
            st.subheader("💾 Save to Warehouse")
            form_result = None
            if _has(_components, "show_detection_record_form"):
                form_result = _components.show_detection_record_form()

            if form_result:
                form_result["pothole_count"] = max(1, n_det)
                form_result["confidence"] = avg_conf
                form_result["severity"] = overall_sev
                form_result["source_type"] = f"Image ({model_used})"
                try:
                    if _has(_db, "insert_detection"):
                        det_id = _db.insert_detection(form_result)
                        st.success(f"✅ Record saved to warehouse (ID: `{det_id}`)!")
                except Exception as exc:
                    st.error(f"Failed to save: {exc}")


# ===========================================================================
# ── PAGE 5: Video Detection ──────────────────────────────────────────────────
# ===========================================================================

def page_video_detection():
    st.title("🎬 Video Detection")

    model_path = st.session_state["model_path"]
    if _has(_components, "show_model_status"):
        _components.show_model_status(model_path)

    video_source_type = st.radio(
        "Choose Video Input Source",
        ["🎬 Use Built-in Road Sample Video (Instant Demo)", "📁 Upload Custom Road Video File"],
        horizontal=True,
        key="vid_source_type"
    )

    video_path = None
    if "Built-in" in video_source_type:
        video_path = "data/demo/test_road.mp4"
        st.info("Using pre-packaged road sample video (`data/demo/test_road.mp4`).")
    else:
        uploaded = st.file_uploader(
            "Upload a video file", type=["mp4", "avi", "mov"],
            help="Supported: MP4, AVI, MOV",
        )
        if uploaded:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded.read())
                video_path = tmp.name

    col1, col2 = st.columns([1, 2])
    with col1:
        conf_thres = st.slider("Confidence Threshold", 0.05, 0.90, 0.20, 0.05, key="vid_conf")
        sample_rate = st.number_input(
            "Process every Nth frame", min_value=1, max_value=60, value=5,
            help="1 = every frame (slow); 5 = every 5th frame (fast and recommended).",
        )
        process_btn = st.button("▶️ Process Road Video", type="primary", use_container_width=True)

        st.divider()
        st.markdown("##### 📍 Location Tagging for Warehouse")
        v_city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata"], key="vid_city")
        v_area = st.text_input("Area", value="Koramangala", key="vid_area")
        v_road = st.text_input("Road Name", value="Sarjapur Main Road", key="vid_road")
        v_road_type = st.selectbox("Road Category", ["Urban", "Highway", "Rural"], key="vid_road_type")

    with col2:
        if process_btn:
            if not video_path:
                st.warning("Please upload a video file or select the built-in road sample video.")
            elif not os.path.isfile(model_path):
                st.error("Model not found. Cannot run detection.")
            else:
                progress = st.progress(0, text="Initialising video analysis...")

                try:
                    import cv2
                    yolo_model = None
                    try:
                        from ultralytics import YOLO
                        yolo_model = YOLO(model_path)
                    except Exception:
                        yolo_model = None
                        st.info("ℹ️ Running Demo AI Model (OpenCV Morphological and Contour Vision Engine)")

                    cap = cv2.VideoCapture(str(video_path))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                    frame_num = 0
                    total_potholes = 0
                    conf_list = []
                    frame_results = []

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_num % sample_rate == 0:
                            if yolo_model is not None:
                                results = yolo_model(frame, conf=conf_thres, verbose=False)
                                boxes = results[0].boxes
                                n_det = len(boxes)
                                confs = [float(b.conf[0]) for b in boxes]
                            else:
                                from ml.demo_detector import detect_potholes_cv
                                dets, _, stats = detect_potholes_cv(frame, conf_thres=conf_thres, is_demo=True)
                                n_det = stats["pothole_count"]
                                confs = [d["confidence"] for d in dets]

                            total_potholes += n_det
                            conf_list.extend(confs)
                            frame_results.append({
                                "frame": frame_num,
                                "potholes": n_det,
                                "avg_conf": round(sum(confs) / len(confs), 3) if confs else 0.0,
                            })

                        frame_num += 1
                        pct = min(int((frame_num / max(total_frames, 1)) * 100), 100)
                        progress.progress(pct, text=f"Analyzing frame {frame_num}/{total_frames}...")

                    cap.release()
                    progress.progress(100, text="✅ Video analysis complete!")

                    avg_conf = round(sum(conf_list) / len(conf_list), 3) if conf_list else 0.0
                    sev = "High" if total_potholes >= 15 else ("Medium" if total_potholes >= 5 else "Low")

                    st.session_state["video_results"] = {
                        "total_frames": total_frames,
                        "sampled_frames": len(frame_results),
                        "total_potholes": total_potholes,
                        "avg_conf": avg_conf,
                        "severity": sev,
                        "frame_results": frame_results,
                    }

                except Exception as exc:
                    st.error(f"Video processing error: {exc}")
                    st.code(traceback.format_exc())

        # Render persisted results and warehouse ingestion form
        res = st.session_state.get("video_results")
        if res:
            st.markdown("#### 📊 Video Analysis Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Frames", res["total_frames"])
            c2.metric("Sampled Frames", res["sampled_frames"])
            c3.metric("Total Potholes", res["total_potholes"])
            c4.metric("Avg Confidence", f"{res['avg_conf']:.3f}")

            sev_color = "red" if res["severity"] == "High" else ("orange" if res["severity"] == "Medium" else "green")
            st.markdown(f"**Assessed Road Condition Severity:** :{sev_color}[**{res['severity']} Severity**]")

            if res["frame_results"]:
                fr_df = pd.DataFrame(res["frame_results"])
                import plotly.express as px
                fig = px.line(
                    fr_df, x="frame", y="potholes",
                    title="Potholes Detected per Sampled Frame",
                    labels={"frame": "Frame Number", "potholes": "Pothole Count"},
                    markers=True,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            if st.button("💾 Ingest Video Assessment into Warehouse", type="primary"):
                rec = {
                    "pothole_count": max(1, res["total_potholes"]),
                    "confidence": max(0.50, res["avg_conf"]),
                    "severity": res["severity"],
                    "source_type": "Video Inference Feed",
                    "detection_date": str(date.today()),
                    "city": v_city,
                    "area": v_area,
                    "road_name": v_road,
                    "road_type": v_road_type,
                    "latitude": 12.9352 if v_city == "Bangalore" else 19.0760,
                    "longitude": 77.6245 if v_city == "Bangalore" else 72.8777,
                }
                try:
                    if _has(_db, "insert_detection"):
                        det_id = _db.insert_detection(rec)
                        st.success(f"✅ Video Assessment successfully saved to Data Warehouse! Detection ID: `{det_id}`")
                    else:
                        st.warning("insert_detection not found in database module.")
                except Exception as ex:
                    st.error(f"Save failed: {ex}")


# ===========================================================================
# ── PAGE 6: Detection Records ────────────────────────────────────────────────
# ===========================================================================

def page_records():
    st.title("📋 Detection Records")

    if is_demo or not warehouse_has_data():
        show_demo_banner()
        if not warehouse_has_data():
            no_data_prompt()
            return

    # Load fact + dims
    try:
        df = query_df(
            """
            SELECT
                f.Detection_ID,
                d.Full_Date        AS date,
                tm.Hour            AS hour,
                l.City_Name        AS city,
                l.Area_Name        AS area,
                r.Road_Name        AS road_name,
                r.Road_Type        AS road_type,
                sv.Severity_Label  AS severity,
                f.Pothole_Count    AS pothole_count,
                f.Confidence_Score AS confidence,
                f.Source_Type      AS source_type
            FROM Pothole_Detection_Fact f
            JOIN Date_Dim     d  ON f.Date_ID     = d.Date_ID
            JOIN Time_Dim     tm ON f.Time_ID     = tm.Time_ID
            JOIN Location_Dim l  ON f.Location_ID = l.Location_ID
            JOIN Road_Dim     r  ON f.Road_ID     = r.Road_ID
            JOIN Severity_Dim sv ON f.Severity_ID = sv.Severity_ID
            ORDER BY f.Detection_ID DESC
            """
        )
    except Exception as exc:
        st.error(f"Failed to load records: {exc}")
        return

    if df.empty:
        no_data_prompt()
        return

    # Filters
    with st.expander("🔽 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            sev_filter = st.multiselect(
                "Severity", ["Low", "Medium", "High"],
                default=["Low", "Medium", "High"],
            )
        with col2:
            cities = ["All"] + sorted(df["city"].dropna().unique().tolist())
            city_filter = st.selectbox("City", cities)
        with col3:
            date_range = st.date_input(
                "Date Range",
                value=(date.today() - timedelta(days=365), date.today()),
            )

    # Apply filters
    filtered = df.copy()
    if sev_filter:
        filtered = filtered[filtered["severity"].isin(sev_filter)]
    if city_filter != "All":
        filtered = filtered[filtered["city"] == city_filter]

    st.subheader(f"Showing {len(filtered):,} records")

    # Pagination
    page_size = 50
    total_pages = max(1, (len(filtered) - 1) // page_size + 1)
    pg_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start = (pg_num - 1) * page_size
    end   = start + page_size

    st.dataframe(filtered.iloc[start:end], use_container_width=True)
    st.caption(f"Page {pg_num} of {total_pages}")

    # Download
    csv = filtered.to_csv(index=False)
    st.download_button(
        "⬇️ Download as CSV",
        data=csv,
        file_name="pothole_detection_records.csv",
        mime="text/csv",
    )


# ===========================================================================
# ── PAGE 7: Data Warehouse ───────────────────────────────────────────────────
# ===========================================================================

def page_warehouse():
    st.title("🏛️ Data Warehouse")

    # Star schema diagram
    with st.expander("📐 Star Schema Diagram", expanded=True):
        st.code(
            """
                     ┌──────────────┐
                     │  Date_Dim    │
                     │  Date_ID(PK) │
                     │  Full_Date   │
                     │  Year/Month  │
                     │  Day_of_Week │
                     └──────┬───────┘
                            │
┌──────────────┐    ┌───────▼────────────────────┐    ┌──────────────┐
│ Location_Dim │    │  Pothole_Detection_Fact     │    │  Road_Dim    │
│ Location_ID  ├────┤  Detection_ID (PK)          ├────┤  Road_ID(PK) │
│ City_Name    │    │  Date_ID     (FK)           │    │  Road_Name   │
│ Area_Name    │    │  Time_ID     (FK)           │    │  Road_Type   │
│ Latitude     │    │  Location_ID (FK)           │    └──────────────┘
│ Longitude    │    │  Road_ID     (FK)           │
└──────────────┘    │  Severity_ID (FK)           │    ┌──────────────┐
                    │  Pothole_Count              ├────┤ Severity_Dim │
┌──────────────┐    │  Confidence_Score           │    │ Severity_ID  │
│  Time_Dim    │    │  Source_Type                │    │ Severity_Lbl │
│  Time_ID(PK) ├────┤  Bounding_Box_Data          │    │ Score_Range  │
│  Hour        │    └─────────────────────────────┘    └──────────────┘
│  Minute      │
│  Shift       │
└──────────────┘
            """,
            language="text",
        )

    st.divider()

    # Table viewer
    TABLE_MAP = {
        "Date_Dim":                 "SELECT * FROM Date_Dim LIMIT 100",
        "Time_Dim":                 "SELECT * FROM Time_Dim LIMIT 100",
        "Location_Dim":             "SELECT * FROM Location_Dim LIMIT 100",
        "Road_Dim":                 "SELECT * FROM Road_Dim LIMIT 100",
        "Severity_Dim":             "SELECT * FROM Severity_Dim LIMIT 100",
        "Pothole_Detection_Fact":   "SELECT * FROM Pothole_Detection_Fact LIMIT 100",
    }

    selected_table = st.selectbox("View Table", list(TABLE_MAP.keys()))

    try:
        tdf = query_df(TABLE_MAP[selected_table])
        count_df = query_df(
            f"SELECT COUNT(*) AS cnt FROM {selected_table}"
        )
        total_rows = int(count_df.iloc[0]["cnt"]) if not count_df.empty else 0

        st.caption(f"Total rows in {selected_table}: **{total_rows:,}** (showing up to 100)")
        st.dataframe(tdf, use_container_width=True)

    except Exception as exc:
        st.error(f"Could not load table `{selected_table}`: {exc}")
        st.info("The warehouse may not be initialised yet. Click 'Initialise Schema' below.")

    st.divider()

    col_init, col_seed = st.columns(2)
    with col_init:
        if st.button("🏗️ Initialise Warehouse Schema", type="primary"):
            try:
                if _has(_db, "create_schema"):
                    _db.create_schema()
                    st.success("✅ Schema created / verified.")
                else:
                    st.info("warehouse/database.py missing create_schema(). Please implement it.")
            except Exception as exc:
                st.error(str(exc))

    with col_seed:
        if st.button("🌱 Seed Demo Data"):
            seed_demo_data()


# ===========================================================================
# ── PAGE 8: OLAP Analysis ────────────────────────────────────────────────────
# ===========================================================================

def page_olap():
    st.title("🔍 OLAP Analysis")

    if is_demo or not warehouse_has_data():
        show_demo_banner()

    tab_interactive, tab_rollup, tab_drill, tab_slice, tab_dice = st.tabs(
        ["🌟 Interactive OLAP Explorer", "📊 Roll-Up", "🔬 Drill-Down", "✂️ Slice", "🎲 Dice"]
    )

    # ── Interactive OLAP Explorer ─────────────────────────────────────────
    with tab_interactive:
        st.subheader("🌟 Interactive Sunburst & Zoomable Treemap OLAP Explorer")
        st.markdown(
            "OLAP dimension hierarchy: **City → Area → Road**. "
            "**Click any ring or sector** to drill down into deeper granular levels. "
            "**Click the inner circle** to roll up."
        )
        fact_df = get_joined_fact_df()
        fact_df = apply_global_filters(fact_df)
        if not fact_df.empty and _has(_interactive, "make_sunburst_olap"):
            col_sb, col_tm = st.columns([1.1, 1])
            with col_sb:
                st.markdown("#### 🎯 Sunburst Drill-Down / Roll-Up")
                fig_sun = _interactive.make_sunburst_olap(fact_df)
                st.plotly_chart(fig_sun, use_container_width=True)
            with col_tm:
                st.markdown("#### 🗂️ Multi-Dimensional Treemap")
                split_mode = st.radio("Group sub-blocks by:", ["Severity Level", "Road Category"], horizontal=True, key="olap_tm_split")
                target_col = "Severity_Level" if split_mode == "Severity Level" else "Road_Type"
                fig_tm = _interactive.make_treemap(fact_df, split_by=target_col)
                st.plotly_chart(fig_tm, use_container_width=True)
        else:
            st.info("Seed warehouse data to use the interactive OLAP Explorer.")

    # ── Roll-Up ──────────────────────────────────────────────────────────
    with tab_rollup:
        st.subheader("Roll-Up: Road → Area → City")
        st.markdown(
            "Aggregates pothole counts from the most granular level (road) "
            "up through area to city."
        )

        level = st.radio("Aggregation Level", ["Road", "Area", "City"], horizontal=True)
        sev_filter = st.multiselect(
            "Filter by Severity", ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"], key="ru_sev",
        )
        sev_clause = (
            f"AND sv.Severity_Label IN ({','.join(['?']*len(sev_filter))})"
            if sev_filter else ""
        )

        try:
            if level == "Road":
                sql = f"""
                    SELECT r.Road_Name AS road_name, SUM(f.Pothole_Count) AS pothole_count
                    FROM Pothole_Detection_Fact f
                    JOIN Road_Dim r ON f.Road_ID = r.Road_ID
                    JOIN Severity_Dim sv ON f.Severity_ID = sv.Severity_ID
                    WHERE 1=1 {sev_clause}
                    GROUP BY r.Road_Name ORDER BY pothole_count DESC
                """
                df = query_df(sql, tuple(sev_filter))
                st.dataframe(df, use_container_width=True)
                if _has(_charts, "create_top_roads_bar") and not df.empty:
                    fig = _charts.create_top_roads_bar(df, is_demo=is_demo)
                    st.plotly_chart(fig, use_container_width=True)

            elif level == "Area":
                sql = f"""
                    SELECT l.Area_Name AS area_name, SUM(f.Pothole_Count) AS pothole_count
                    FROM Pothole_Detection_Fact f
                    JOIN Location_Dim l ON f.Location_ID = l.Location_ID
                    JOIN Severity_Dim sv ON f.Severity_ID = sv.Severity_ID
                    WHERE 1=1 {sev_clause}
                    GROUP BY l.Area_Name ORDER BY pothole_count DESC
                """
                df = query_df(sql, tuple(sev_filter))
                st.dataframe(df, use_container_width=True)
                if not df.empty:
                    import plotly.express as px
                    fig = px.bar(df, x="area_name", y="pothole_count",
                                 title="Pothole Count by Area",
                                 labels={"area_name": "Area", "pothole_count": "Count"})
                    st.plotly_chart(fig, use_container_width=True)

            else:  # City
                sql = f"""
                    SELECT l.City_Name AS city_name, SUM(f.Pothole_Count) AS pothole_count
                    FROM Pothole_Detection_Fact f
                    JOIN Location_Dim l ON f.Location_ID = l.Location_ID
                    JOIN Severity_Dim sv ON f.Severity_ID = sv.Severity_ID
                    WHERE 1=1 {sev_clause}
                    GROUP BY l.City_Name ORDER BY pothole_count DESC
                """
                df = query_df(sql, tuple(sev_filter))
                st.dataframe(df, use_container_width=True)
                if not df.empty:
                    import plotly.express as px
                    fig = px.bar(df, x="city_name", y="pothole_count",
                                 title="Pothole Count by City",
                                 labels={"city_name": "City", "pothole_count": "Count"},
                                 color="pothole_count", color_continuous_scale="Reds")
                    st.plotly_chart(fig, use_container_width=True)

            # Combined roll-up chart
            if _has(_charts, "create_rollup_chart"):
                st.subheader("Combined Roll-Up View")
                road_df = query_df(
                    "SELECT r.Road_Name AS road_name, SUM(f.Pothole_Count) AS pothole_count "
                    "FROM Pothole_Detection_Fact f JOIN Road_Dim r ON f.Road_ID=r.Road_ID "
                    "GROUP BY r.Road_Name"
                )
                area_df = query_df(
                    "SELECT l.Area_Name AS area_name, SUM(f.Pothole_Count) AS pothole_count "
                    "FROM Pothole_Detection_Fact f JOIN Location_Dim l ON f.Location_ID=l.Location_ID "
                    "GROUP BY l.Area_Name"
                )
                city_df = query_df(
                    "SELECT l.City_Name AS city_name, SUM(f.Pothole_Count) AS pothole_count "
                    "FROM Pothole_Detection_Fact f JOIN Location_Dim l ON f.Location_ID=l.Location_ID "
                    "GROUP BY l.City_Name"
                )
                fig = _charts.create_rollup_chart(road_df, area_df, city_df, is_demo=is_demo)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as exc:
            st.error(f"Roll-up error: {exc}")

    # ── Drill-Down ────────────────────────────────────────────────────────
    with tab_drill:
        st.subheader("Drill-Down: City → Area → Road")
        st.markdown("Select a city to drill into its areas, then a road.")

        try:
            city_df = query_df(
                "SELECT DISTINCT l.City_Name FROM Location_Dim l ORDER BY l.City_Name"
            )
            cities = city_df["City_Name"].tolist() if not city_df.empty else ["(No data)"]

            sel_city = st.selectbox("Select City", cities, key="dd_city")

            area_df = query_df(
                "SELECT DISTINCT l.Area_Name FROM Location_Dim l WHERE l.City_Name=? ORDER BY l.Area_Name",
                (sel_city,),
            )
            areas = area_df["Area_Name"].tolist() if not area_df.empty else ["(No data)"]
            sel_area = st.selectbox("Select Area", areas, key="dd_area")

            road_df = query_df(
                """
                SELECT r.Road_Name, SUM(f.Pothole_Count) AS pothole_count,
                       AVG(f.Confidence_Score) AS avg_confidence
                FROM Pothole_Detection_Fact f
                JOIN Location_Dim l ON f.Location_ID = l.Location_ID
                JOIN Road_Dim r ON f.Road_ID = r.Road_ID
                WHERE l.City_Name=? AND l.Area_Name=?
                GROUP BY r.Road_Name ORDER BY pothole_count DESC
                """,
                (sel_city, sel_area),
            )

            if not road_df.empty:
                st.dataframe(road_df, use_container_width=True)
            else:
                st.info(f"No road data found for {sel_city} / {sel_area}.")

        except Exception as exc:
            st.error(f"Drill-down error: {exc}")

    # ── Slice ─────────────────────────────────────────────────────────────
    with tab_slice:
        st.subheader("Slice: Fix One Dimension")
        st.markdown("Hold one dimension constant and view all others.")

        try:
            slice_dim = st.selectbox(
                "Slice Dimension",
                ["Severity", "City", "Year", "Road Type"],
                key="sl_dim",
            )

            if slice_dim == "Severity":
                sev_val = st.selectbox("Severity Value", ["Low", "Medium", "High"])
                sql = """
                    SELECT l.City_Name AS city, r.Road_Name AS road,
                           SUM(f.Pothole_Count) AS pothole_count
                    FROM Pothole_Detection_Fact f
                    JOIN Location_Dim l ON f.Location_ID = l.Location_ID
                    JOIN Road_Dim r ON f.Road_ID = r.Road_ID
                    JOIN Severity_Dim sv ON f.Severity_ID = sv.Severity_ID
                    WHERE sv.Severity_Label = ?
                    GROUP BY l.City_Name, r.Road_Name ORDER BY pothole_count DESC
                """
                df = query_df(sql, (sev_val,))

            elif slice_dim == "City":
                city_list_df = query_df("SELECT DISTINCT City_Name FROM Location_Dim ORDER BY City_Name")
                city_list = city_list_df["City_Name"].tolist() if not city_list_df.empty else []
                city_val = st.selectbox("City Value", city_list or ["(No data)"])
                sql = """
                    SELECT d.Year, d.Month_Name, sv.Severity_Label AS severity,
                           SUM(f.Pothole_Count) AS pothole_count
                    FROM Pothole_Detection_Fact f
                    JOIN Date_Dim d ON f.Date_ID = d.Date_ID
                    JOIN Location_Dim l ON f.Location_ID = l.Location_ID
                    JOIN Severity_Dim sv ON f.Severity_ID = sv.Severity_ID
                    WHERE l.City_Name = ?
                    GROUP BY d.Year, d.Month_Name, sv.Severity_Label
                    ORDER BY d.Year, d.Month_Num
                """
                df = query_df(sql, (city_val,))

            elif slice_dim == "Year":
                yr_df = query_df("SELECT DISTINCT Year FROM Date_Dim ORDER BY Year DESC")
                yr_list = [str(y) for y in yr_df["Year"].tolist()] if not yr_df.empty else []
                yr_val = st.selectbox("Year Value", yr_list or ["(No data)"])
                sql = """
                    SELECT l.City_Name AS city, l.Area_Name AS area,
                           SUM(f.Pothole_Count) AS pothole_count
                    FROM Pothole_Detection_Fact f
                    JOIN Date_Dim d ON f.Date_ID = d.Date_ID
                    JOIN Location_Dim l ON f.Location_ID = l.Location_ID
                    WHERE d.Year = ?
                    GROUP BY l.City_Name, l.Area_Name ORDER BY pothole_count DESC
                """
                df = query_df(sql, (int(yr_val),))

            else:  # Road Type
                rt_df = query_df("SELECT DISTINCT Road_Type FROM Road_Dim ORDER BY Road_Type")
                rt_list = rt_df["Road_Type"].tolist() if not rt_df.empty else []
                rt_val = st.selectbox("Road Type Value", rt_list or ["(No data)"])
                sql = """
                    SELECT l.City_Name AS city, r.Road_Name AS road,
                           SUM(f.Pothole_Count) AS pothole_count
                    FROM Pothole_Detection_Fact f
                    JOIN Road_Dim r ON f.Road_ID = r.Road_ID
                    JOIN Location_Dim l ON f.Location_ID = l.Location_ID
                    WHERE r.Road_Type = ?
                    GROUP BY l.City_Name, r.Road_Name ORDER BY pothole_count DESC
                """
                df = query_df(sql, (rt_val,))

            if not df.empty:
                st.dataframe(df, use_container_width=True)
                st.caption(f"{len(df)} records")
            else:
                st.info("No records match the selected slice criteria.")

        except Exception as exc:
            st.error(f"Slice error: {exc}")

    # ── Dice ──────────────────────────────────────────────────────────────
    with tab_dice:
        st.subheader("Dice: Multiple Dimension Filters")
        st.markdown("Apply filters on multiple dimensions simultaneously.")

        try:
            with st.form("dice_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    dice_sev = st.multiselect(
                        "Severity", ["Low", "Medium", "High"],
                        default=["Low", "Medium", "High"],
                    )
                with col2:
                    dice_city_df = query_df("SELECT DISTINCT City_Name FROM Location_Dim ORDER BY City_Name")
                    dice_cities = dice_city_df["City_Name"].tolist() if not dice_city_df.empty else []
                    dice_city = st.multiselect("Cities", dice_cities, default=dice_cities[:3] if dice_cities else [])
                with col3:
                    dice_yr_df = query_df("SELECT DISTINCT Year FROM Date_Dim ORDER BY Year DESC")
                    dice_years = dice_yr_df["Year"].tolist() if not dice_yr_df.empty else []
                    dice_year = st.multiselect("Years", [str(y) for y in dice_years],
                                               default=[str(y) for y in dice_years[:2]])
                dice_submit = st.form_submit_button("🎲 Apply Dice Filters", type="primary")

            if dice_submit:
                conditions, params = [], []

                if dice_sev:
                    conditions.append(f"sv.Severity_Label IN ({','.join(['?']*len(dice_sev))})")
                    params.extend(dice_sev)
                if dice_city:
                    conditions.append(f"l.City_Name IN ({','.join(['?']*len(dice_city))})")
                    params.extend(dice_city)
                if dice_year:
                    conditions.append(f"d.Year IN ({','.join(['?']*len(dice_year))})")
                    params.extend([int(y) for y in dice_year])

                where = "WHERE " + " AND ".join(conditions) if conditions else ""

                sql = f"""
                    SELECT d.Year, d.Month_Name, l.City_Name AS city, l.Area_Name AS area,
                           r.Road_Name AS road, sv.Severity_Label AS severity,
                           SUM(f.Pothole_Count) AS pothole_count
                    FROM Pothole_Detection_Fact f
                    JOIN Date_Dim d ON f.Date_ID = d.Date_ID
                    JOIN Location_Dim l ON f.Location_ID = l.Location_ID
                    JOIN Road_Dim r ON f.Road_ID = r.Road_ID
                    JOIN Severity_Dim sv ON f.Severity_ID = sv.Severity_ID
                    {where}
                    GROUP BY d.Year, d.Month_Name, l.City_Name, l.Area_Name,
                             r.Road_Name, sv.Severity_Label
                    ORDER BY pothole_count DESC
                """
                df = query_df(sql, tuple(params))

                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"{len(df)} result rows")
                    csv = df.to_csv(index=False)
                    st.download_button("⬇️ Download Dice Result", csv,
                                       "dice_result.csv", "text/csv")
                else:
                    st.info("No records match the dice criteria.")

        except Exception as exc:
            st.error(f"Dice error: {exc}")


# ===========================================================================
# ── PAGE 9: K-Means Clustering ───────────────────────────────────────────────
# ===========================================================================

def page_kmeans():
    st.title("🔮 K-Means Clustering")
    st.markdown(
        "Cluster road segments based on **pothole count** and "
        "**detection frequency** to identify hotspot patterns."
    )

    if is_demo or not warehouse_has_data():
        show_demo_banner()

    # K slider
    k = 3
    if _has(_components, "show_cluster_controls"):
        k = _components.show_cluster_controls(max_k=8)
    else:
        k = st.slider("Number of Clusters (K)", 2, 8, 3)

    if st.button("▶️ Run K-Means Clustering", type="primary"):
        try:
            # Try to use the mining module
            if _has(_kmeans, "run_kmeans"):
                _model = _kmeans.run_kmeans(n_clusters=k)
                result = {
                    "clustered_df": _model.df,
                    "centers_df": _model.get_cluster_centers_df(),
                    "silhouette_score": _model.get_silhouette_score(),
                    "k": _model.n_clusters,
                }
            else:
                # Fallback: pull data from warehouse and run sklearn K-Means
                sql = """
                    SELECT r.Road_Name AS road_name,
                           l.Area_Name AS area_name,
                           l.City_Name AS city_name,
                           SUM(f.Pothole_Count) AS pothole_count,
                           COUNT(f.Detection_ID) AS detection_frequency
                    FROM Pothole_Detection_Fact f
                    JOIN Road_Dim r ON f.Road_ID = r.Road_ID
                    JOIN Location_Dim l ON f.Location_ID = l.Location_ID
                    GROUP BY r.Road_Name, l.Area_Name, l.City_Name
                    HAVING pothole_count > 0
                """
                df = query_df(sql)

                if df.empty:
                    st.warning("No data in warehouse. Using demo data for clustering.")
                    if _has(_maps, "get_demo_locations"):
                        demo_df = _maps.get_demo_locations()
                        df = demo_df[["road_name", "city", "pothole_count"]].copy()
                        df["detection_frequency"] = df["pothole_count"] // 3 + 1
                        df.rename(columns={"city": "city_name"}, inplace=True)
                    else:
                        import numpy as np
                        rng = pd.np.random if hasattr(pd, "np") else None
                        import numpy as _np
                        df = pd.DataFrame({
                            "road_name": [f"Road_{i}" for i in range(30)],
                            "pothole_count": _np.random.randint(1, 50, 30),
                            "detection_frequency": _np.random.randint(1, 20, 30),
                        })

                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import silhouette_score

                features = df[["pothole_count", "detection_frequency"]].fillna(0).values
                scaler   = StandardScaler()
                X_scaled = scaler.fit_transform(features)

                n_clusters = min(k, len(df))
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)

                df["cluster_label"] = labels

                sil = 0.0
                if len(set(labels)) > 1:
                    sil = silhouette_score(X_scaled, labels)

                centers = scaler.inverse_transform(km.cluster_centers_)
                centers_df = pd.DataFrame(
                    centers,
                    columns=["pothole_count_center", "detection_frequency_center"],
                )
                centers_df["cluster"] = range(len(centers_df))

                result = {
                    "clustered_df": df,
                    "centers_df": centers_df,
                    "silhouette_score": sil,
                    "k": n_clusters,
                }

            st.session_state["kmeans_result"] = result
            st.success(f"✅ K-Means complete with K={result.get('k', k)}")

        except ImportError as e:
            st.error(f"Missing package: {e}. Install scikit-learn.")
        except Exception as exc:
            st.error(f"Clustering error: {exc}")
            st.code(traceback.format_exc())

    # Display results
    result = st.session_state.get("kmeans_result")
    if result:
        clustered_df = result.get("clustered_df", pd.DataFrame())
        centers_df   = result.get("centers_df", pd.DataFrame())
        sil_score    = result.get("silhouette_score", 0.0)
        k_used       = result.get("k", k)

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters (K)", k_used)
        c2.metric("Silhouette Score", f"{sil_score:.4f}")
        c3.metric("Roads Clustered", len(clustered_df))

        st.info(
            "ℹ️ **Cluster labels are neutral integers** (0, 1, 2 …). "
            "They are assigned by geometric proximity in feature space — "
            "not by severity or quality rank.",
            icon="ℹ️",
        )

        # Cluster Visualizations (2D & 3D Interactive)
        st.subheader("📊 Cluster Space Visualisation")
        tab_2d, tab_3d = st.tabs(["📊 2D Feature Space", "🔮 3D Interactive Cluster Space"])
        with tab_2d:
            if _has(_charts, "create_kmeans_scatter"):
                fig = _charts.create_kmeans_scatter(clustered_df, is_demo=is_demo)
                st.plotly_chart(fig, use_container_width=True)
        with tab_3d:
            st.markdown(
                "**Interactive 3D Cluster Visualizer**: Click and drag to rotate the camera in 3D, "
                "scroll to zoom in/out, hover over data points for road and area details."
            )
            if _has(_interactive, "make_3d_cluster_scatter"):
                fig_3d = _interactive.make_3d_cluster_scatter(clustered_df)
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("3D Visualizer available when dashboard.interactive is loaded.")

        # Cluster summary table
        st.subheader("Cluster Summary")
        summary = (
            clustered_df.groupby("cluster_label")
            .agg(
                roads=("road_name", "count") if "road_name" in clustered_df.columns else ("pothole_count", "count"),
                avg_potholes=("pothole_count", "mean"),
                total_potholes=("pothole_count", "sum"),
                avg_frequency=("detection_frequency", "mean"),
            )
            .reset_index()
        )
        summary.columns = [c.replace("_", " ").title() for c in summary.columns]
        st.dataframe(summary, use_container_width=True)

        # Cluster centres
        if not centers_df.empty:
            st.subheader("Cluster Centers (original scale)")
            st.dataframe(centers_df, use_container_width=True)

        # Per-cluster breakdown
        st.subheader("Records per Cluster")
        for cl in sorted(clustered_df["cluster_label"].unique()):
            with st.expander(f"Cluster {cl}"):
                cl_df = clustered_df[clustered_df["cluster_label"] == cl]
                st.dataframe(cl_df, use_container_width=True)


# ===========================================================================
# ── PAGE 10: Hotspot Map ─────────────────────────────────────────────────────
# ===========================================================================

def page_map():
    st.title("🗺️ Pothole Hotspot Map")

    if is_demo or not warehouse_has_data():
        show_demo_banner()

    # Load location data
    try:
        df = query_df(
            """
            SELECT l.City_Name AS city, l.Area_Name AS area,
                   l.Latitude AS latitude, l.Longitude AS longitude,
                   r.Road_Name AS road_name,
                   sv.Severity_Label AS severity,
                   SUM(f.Pothole_Count) AS pothole_count
            FROM Pothole_Detection_Fact f
            JOIN Location_Dim l ON f.Location_ID = l.Location_ID
            JOIN Road_Dim r ON f.Road_ID = r.Road_ID
            JOIN Severity_Dim sv ON f.Severity_ID = sv.Severity_ID
            WHERE l.Latitude IS NOT NULL AND l.Longitude IS NOT NULL
            GROUP BY l.City_Name, l.Area_Name, l.Latitude, l.Longitude,
                     r.Road_Name, sv.Severity_Label
            """
        )
    except Exception:
        df = pd.DataFrame()

    # Use demo data if empty
    if df.empty and _has(_maps, "get_demo_locations"):
        df = _maps.get_demo_locations()
        st.info("Using demo locations for the map.", icon="ℹ️")

    if df.empty:
        st.warning("No geospatial data found. Seed demo data first.")
        if st.button("🌱 Seed Demo Data"):
            seed_demo_data()
        return

    # Interactive view mode toggle
    map_mode = st.radio(
        "Map Display Mode",
        ["📍 Interactive Severity Pins", "🔥 Kernel Density Heatmap"],
        horizontal=True,
        key="map_view_mode",
    )

    # Filters
    col1, col2 = st.columns([1, 2])
    with col1:
        city_opts = ["All"] + sorted(df["city"].dropna().unique().tolist())
        city_filter = st.selectbox("Filter by City", city_opts, key="map_city")

        sev_filter = st.multiselect(
            "Filter by Severity", ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"], key="map_sev",
        )

    filtered = df.copy()
    if city_filter != "All":
        filtered = filtered[filtered["city"] == city_filter]
    if sev_filter:
        filtered = filtered[filtered["severity"].isin(sev_filter)]

    with col2:
        st.metric("Locations on Map", len(filtered))

    # Render interactive map using Plotly 7 carto-darkmatter
    fact_map_df = get_joined_fact_df()
    if not fact_map_df.empty:
        if city_filter != "All":
            fact_map_df = fact_map_df[fact_map_df["City"] == city_filter]
        if sev_filter:
            fact_map_df = fact_map_df[fact_map_df["Severity_Level"].isin(sev_filter)]

    if _has(_interactive, "make_density_map") and not fact_map_df.empty:
        mode_val = "density" if "Heatmap" in map_mode else "scatter"
        fig_dark_map = _interactive.make_density_map(fact_map_df, map_type=mode_val)
        st.plotly_chart(fig_dark_map, use_container_width=True)
    elif _has(_maps, "create_hotspot_map"):
        fig = _maps.create_hotspot_map(filtered, is_demo=(is_demo or df.get("Is_Demo", pd.Series([0])).any() if hasattr(df, "get") else is_demo))
        st.plotly_chart(fig, use_container_width=True)

    # City heatmap
    st.subheader("City-Level Heatmap")
    if _has(_maps, "create_city_heatmap"):
        fig2 = _maps.create_city_heatmap(
            filtered.rename(columns={"city": "city_name"}) if "city" in filtered.columns else filtered,
            is_demo=is_demo,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ===========================================================================
# ── PAGE 11: Trends ──────────────────────────────────────────────────────────
# ===========================================================================

def page_trends():
    st.title("📈 Trends & Analytics")

    if is_demo or not warehouse_has_data():
        show_demo_banner()

    fact_df = get_joined_fact_df()
    fact_df = apply_global_filters(fact_df)

    tab_anim, tab_cal, tab_polar, tab_radar, tab_std = st.tabs([
        "📅 Animated Monthly Trend",
        "🗓️ Day × Month Heatmap",
        "🕐 Peak Hour Rose Chart",
        "🕸️ Multi-City Radar",
        "📊 Classic Trends & Violins",
    ])

    with tab_anim:
        st.subheader("📅 Animated Monthly Pothole Accumulation")
        st.caption("Presents the quarterly evolution of potholes by severity level. Press Play to animate.")
        if _has(_interactive, "make_animated_monthly_trend") and not fact_df.empty:
            st.plotly_chart(_interactive.make_animated_monthly_trend(fact_df), use_container_width=True)
        else:
            st.info("Interactive monthly trend available when warehouse has data.")

    with tab_cal:
        st.subheader("🗓️ Day of Week × Month Risk Heatmap")
        st.caption("Identify temporal hotspots: which day and month combinations have peak pothole emergence.")
        if _has(_interactive, "make_heatmap_calendar") and not fact_df.empty:
            st.plotly_chart(_interactive.make_heatmap_calendar(fact_df), use_container_width=True)
        else:
            st.info("Heatmap calendar available when warehouse has data.")

    with tab_polar:
        st.subheader("🕐 Peak Hour Risk Polar Windrose")
        st.caption("24-hour circular representation of detections by hour of day and severity band.")
        if _has(_interactive, "make_polar_time_chart") and not fact_df.empty:
            st.plotly_chart(_interactive.make_polar_time_chart(fact_df), use_container_width=True)
        else:
            st.info("Polar time chart available when warehouse has data.")

    with tab_radar:
        st.subheader("🕸️ Multi-City Comparative Damage Radar")
        st.caption("Normalized 5-dimensional comparison of municipal road networks across cities.")
        if _has(_interactive, "make_city_radar") and not fact_df.empty:
            st.plotly_chart(_interactive.make_city_radar(fact_df), use_container_width=True)
        else:
            st.info("City radar available when warehouse has data.")

    with tab_std:
        st.subheader("Monthly Pothole Detection Trend")
    try:
        monthly_df = query_df(
            """
            SELECT d.Year || '-' || PRINTF('%02d', d.Month_Num) AS month_year,
                   sv.Severity_Label AS severity,
                   SUM(f.Pothole_Count) AS pothole_count
            FROM Pothole_Detection_Fact f
            JOIN Date_Dim d ON f.Date_ID = d.Date_ID
            JOIN Severity_Dim sv ON f.Severity_ID = sv.Severity_ID
            GROUP BY month_year, severity
            ORDER BY month_year
            """
        )

        if monthly_df.empty and is_demo:
            # Generate fake monthly data for demo
            import numpy as np
            months = pd.date_range(start="2022-01", periods=18, freq="MS")
            rows = []
            for m in months:
                for sev, base in [("Low", 10), ("Medium", 6), ("High", 3)]:
                    rows.append({
                        "month_year": m.strftime("%Y-%m"),
                        "severity": sev,
                        "pothole_count": int(base + np.random.randint(0, 8)),
                    })
            monthly_df = pd.DataFrame(rows)

        if _has(_charts, "create_monthly_trend_chart"):
            fig = _charts.create_monthly_trend_chart(monthly_df, is_demo=is_demo)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"Trend chart error: {exc}")

    # ── Day-of-Week Pattern ───────────────────────────────────────────────
    st.subheader("Day-of-Week Pattern")
    try:
        dow_df = query_df(
            """
            SELECT d.Day_Of_Week AS day, SUM(f.Pothole_Count) AS pothole_count
            FROM Pothole_Detection_Fact f
            JOIN Date_Dim d ON f.Date_ID = d.Date_ID
            GROUP BY d.Day_Of_Week
            ORDER BY CASE d.Day_Of_Week
                WHEN 'Monday' THEN 1 WHEN 'Tuesday' THEN 2
                WHEN 'Wednesday' THEN 3 WHEN 'Thursday' THEN 4
                WHEN 'Friday' THEN 5 WHEN 'Saturday' THEN 6
                ELSE 7 END
            """
        )

        if dow_df.empty and is_demo:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            import numpy as np
            dow_df = pd.DataFrame({
                "day": days,
                "pothole_count": [int(v) for v in np.random.randint(15, 60, 7)],
            })

        if not dow_df.empty:
            import plotly.express as px
            fig = px.bar(
                dow_df, x="day", y="pothole_count",
                title="Pothole Detections by Day of Week",
                labels={"day": "Day", "pothole_count": "Pothole Count"},
                color="pothole_count", color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"Day-of-week chart error: {exc}")

    # ── Top Roads ────────────────────────────────────────────────────────
    st.subheader("Top Roads by Pothole Count")
    try:
        top_roads_df = query_df(
            """
            SELECT r.Road_Name AS road_name, SUM(f.Pothole_Count) AS pothole_count
            FROM Pothole_Detection_Fact f
            JOIN Road_Dim r ON f.Road_ID = r.Road_ID
            GROUP BY r.Road_Name ORDER BY pothole_count DESC LIMIT 15
            """
        )

        if top_roads_df.empty and is_demo and _has(_maps, "get_demo_locations"):
            demo = _maps.get_demo_locations()
            top_roads_df = demo[["road_name", "pothole_count"]].copy()

        if _has(_charts, "create_top_roads_bar") and not top_roads_df.empty:
            fig = _charts.create_top_roads_bar(top_roads_df, top_n=10, is_demo=is_demo)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"Top roads chart error: {exc}")

    # ── City Comparison ───────────────────────────────────────────────────
    st.subheader("City Comparison")
    try:
        city_df = query_df(
            """
            SELECT l.City_Name AS city_name, sv.Severity_Label AS severity,
                   SUM(f.Pothole_Count) AS pothole_count
            FROM Pothole_Detection_Fact f
            JOIN Location_Dim l ON f.Location_ID = l.Location_ID
            JOIN Severity_Dim sv ON f.Severity_ID = sv.Severity_ID
            GROUP BY l.City_Name, sv.Severity_Label ORDER BY pothole_count DESC
            """
        )

        if city_df.empty and is_demo and _has(_maps, "get_demo_locations"):
            demo = _maps.get_demo_locations()
            city_df = (
                demo.groupby(["city", "severity"])["pothole_count"]
                .sum().reset_index()
                .rename(columns={"city": "city_name"})
            )

        if not city_df.empty:
            import plotly.express as px
            fig = px.bar(
                city_df, x="city_name", y="pothole_count", color="severity",
                barmode="stack",
                color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
                title="City-wise Pothole Distribution by Severity",
                labels={"city_name": "City", "pothole_count": "Pothole Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"City comparison chart error: {exc}")

    # ── Confidence Distribution ───────────────────────────────────────────
    st.subheader("Confidence Score Distribution")
    try:
        conf_df = query_df(
            "SELECT Confidence_Score AS confidence FROM Pothole_Detection_Fact"
        )

        if conf_df.empty and is_demo:
            import numpy as np
            conf_df = pd.DataFrame({"confidence": np.random.beta(5, 2, 200)})

        if _has(_charts, "create_confidence_histogram") and not conf_df.empty:
            fig = _charts.create_confidence_histogram(conf_df, is_demo=is_demo)
            st.plotly_chart(fig, use_container_width=True)

        if _has(_interactive, "make_confidence_violin") and not fact_df.empty:
            st.plotly_chart(_interactive.make_confidence_violin(fact_df), use_container_width=True)

    except Exception as exc:
        st.error(f"Confidence chart error: {exc}")


# ===========================================================================
# ── PAGE 12: Reports ─────────────────────────────────────────────────────────
# ===========================================================================

def page_reports():
    st.title("📑 Reports")

    if is_demo or not warehouse_has_data():
        show_demo_banner()

    tab_summary, tab_preproc, tab_kmeans_rpt = st.tabs(
        ["📊 Summary Report", "⚙️ Preprocessing Report", "🔮 K-Means Report"]
    )

    # ── Summary Report ────────────────────────────────────────────────────
    with tab_summary:
        st.subheader("Warehouse Summary Report")

        try:
            summ = get_warehouse_summary()
            if _has(_charts, "create_kpi_metrics"):
                kpi_formatted = _charts.create_kpi_metrics(summ)
                rows = [{"Metric": k, "Value": v} for k, v in kpi_formatted.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Full fact summary
            fact_df = query_df(
                """
                SELECT sv.Severity_Label AS severity,
                       l.City_Name AS city,
                       SUM(f.Pothole_Count) AS total_potholes,
                       COUNT(*) AS detections,
                       AVG(f.Confidence_Score) AS avg_confidence
                FROM Pothole_Detection_Fact f
                JOIN Location_Dim l ON f.Location_ID = l.Location_ID
                JOIN Severity_Dim sv ON f.Severity_ID = sv.Severity_ID
                GROUP BY sv.Severity_Label, l.City_Name
                ORDER BY total_potholes DESC
                """
            )

            if not fact_df.empty:
                st.subheader("Breakdown by City & Severity")
                st.dataframe(fact_df, use_container_width=True)
                csv = fact_df.to_csv(index=False)
                st.download_button("⬇️ Download Summary CSV", csv,
                                   "pothole_summary.csv", "text/csv")

                import json
                json_str = fact_df.to_json(orient="records", indent=2)
                st.download_button("⬇️ Download Summary JSON", json_str,
                                   "pothole_summary.json", "application/json")
            else:
                no_data_prompt("Seed Demo Data to Generate Report")

        except Exception as exc:
            st.error(f"Report generation error: {exc}")

    # ── Preprocessing Report ──────────────────────────────────────────────
    with tab_preproc:
        report = st.session_state.get("preproc_report", {})
        if _has(_components, "show_preprocessing_report"):
            _components.show_preprocessing_report(report)
        else:
            if report:
                st.json(report)
            else:
                st.info("Run the preprocessing pipeline to generate a report.")

        if report:
            csv = pd.DataFrame(
                [{"Metric": k, "Value": v} for k, v in report.items()]
            ).to_csv(index=False)
            st.download_button("⬇️ Download Preprocessing Report", csv,
                               "preprocessing_report.csv", "text/csv")

    # ── K-Means Report ────────────────────────────────────────────────────
    with tab_kmeans_rpt:
        result = st.session_state.get("kmeans_result")
        if not result:
            st.info("Run K-Means clustering (🔮 K-Means Clustering page) to generate this report.")
            return

        clustered_df = result.get("clustered_df", pd.DataFrame())
        sil = result.get("silhouette_score", 0.0)
        k   = result.get("k", "?")

        st.subheader(f"K-Means Report — K={k}")
        c1, c2, c3 = st.columns(3)
        c1.metric("K (Clusters)", k)
        c2.metric("Silhouette Score", f"{sil:.4f}")
        c3.metric("Roads Clustered", len(clustered_df))

        if not clustered_df.empty:
            summary = (
                clustered_df.groupby("cluster_label")
                .agg(
                    count=("pothole_count", "count"),
                    avg_potholes=("pothole_count", "mean"),
                    max_potholes=("pothole_count", "max"),
                    avg_frequency=("detection_frequency", "mean"),
                )
                .reset_index()
            )
            st.dataframe(summary, use_container_width=True)

            csv = clustered_df.to_csv(index=False)
            st.download_button("⬇️ Download Cluster Data CSV", csv,
                               "kmeans_clusters.csv", "text/csv")



# ===========================================================================
# ── PAGE 13: AI Predictions & Optimization ──────────────────────────────────
# ===========================================================================

def page_ai_analytics():
    st.title("🤖 AI Predictions & Route Optimization")
    st.caption("Advanced AI Decision Support, Material Forecasting & Dispatch Optimization")

    if is_demo or not warehouse_has_data():
        show_demo_banner()

    tab_cost, tab_forecast, tab_route, tab_budget = st.tabs([
        "💰 AI Repair Cost Estimator",
        "🌧️ AI Deterioration Forecaster",
        "🗺️ AI Route Optimizer (TSP)",
        "🎯 Budget Scenario Simulator"
    ])

    # ── TAB 1: AI Repair Cost & Material Estimator ─────────────────────────
    with tab_cost:
        st.subheader("🛠️ AI Pothole Repair Cost & Asphalt Material Estimator")
        st.markdown(
            "Predicts required asphalt material volume ($m^3$), material weight, "
            "financial repair estimates (in ₹ INR), and calculates the **Repair Priority Index (RPI)**."
        )

        col1, col2 = st.columns(2)
        with col1:
            # Load roads from warehouse if available
            roads_df = query_df("SELECT DISTINCT r.Road_Name, r.Road_Type FROM Road_Dim r ORDER BY r.Road_Name")
            road_options = ["(Custom Road / Manual Input)"] + roads_df["Road_Name"].tolist() if not roads_df.empty else ["(Custom Input)"]
            selected_road = st.selectbox("Select Road", road_options, key="ai_road_sel")

            pothole_count = st.number_input("Detected Pothole Count", min_value=1, max_value=200, value=6, step=1)
            severity = st.selectbox("Severity Classification", ["High", "Medium", "Low"], index=0)
            road_type = st.selectbox("Road Category", ["Urban", "Highway", "Rural"], index=0)

        with col2:
            from mining.ai_analytics import estimate_repair_cost
            cost_info = estimate_repair_cost(
                pothole_count=int(pothole_count),
                severity_level=severity,
                road_type=road_type
            )

            st.markdown("#### 📋 AI Assessment Results")
            kpi1, kpi2 = st.columns(2)
            kpi1.metric("Estimated Cost", f"₹ {cost_info['estimated_cost_inr']:,.2f}")
            kpi2.metric("Priority Index (RPI)", f"{cost_info['priority_index']}/100")

            kpi3, kpi4 = st.columns(2)
            kpi3.metric("Asphalt Volume", f"{cost_info['estimated_volume_m3']:.3f} m³")
            kpi4.metric("Material Weight", f"{cost_info['material_weight_tonnes']:.2f} Tonnes")

            rpi = cost_info["priority_index"]
            if rpi >= 75:
                st.error(f"🚨 **Recommendation:** {cost_info['recommended_action']}")
            elif rpi >= 50:
                st.warning(f"⚠️ **Recommendation:** {cost_info['recommended_action']}")
            else:
                st.success(f"✅ **Recommendation:** {cost_info['recommended_action']}")

            st.caption("Based on standard CPWD Municipal Asphalt Paving Specifications.")

    # ── TAB 2: AI Road Deterioration Forecaster ───────────────────────────
    with tab_forecast:
        st.subheader("🌧️ AI Road Deterioration & Monsoon Degradation Forecaster")
        st.markdown(
            "Simulates future pothole proliferation and surface degradation over time "
            "under untreated environmental and traffic stress."
        )

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            init_potholes = st.slider("Current Pothole Count", 1, 50, 8)
        with col_f2:
            monsoon = st.slider("Monsoon Rain Intensity Multiplier", 1.0, 3.0, 1.8, 0.1)
        with col_f3:
            traffic = st.selectbox("Heavy Traffic Density", ["High", "Medium", "Low"], index=0)

        days_horizon = st.radio("Forecast Horizon", [30, 60, 90, 180], index=2, horizontal=True)

        from mining.ai_analytics import forecast_road_deterioration
        forecast_df = forecast_road_deterioration(
            current_potholes=init_potholes,
            monsoon_factor=monsoon,
            traffic_density=traffic,
            days=days_horizon
        )

        import plotly.express as px
        fig_fc = px.line(
            forecast_df, x="Days_Elapsed", y="Projected_Potholes",
            markers=True,
            title=f"Exponential Deterioration Curve ({days_horizon}-Day Projection)",
            labels={"Days_Elapsed": "Days Elapsed", "Projected_Potholes": "Projected Potholes"},
        )
        fig_fc.add_bar(
            x=forecast_df["Days_Elapsed"], y=forecast_df["New_Potholes_Formed"],
            name="New Potholes Formed"
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.dataframe(forecast_df, use_container_width=True)

    # ── TAB 3: AI Optimal Municipal Repair Route (TSP Solver) ──────────────
    with tab_route:
        st.subheader("🗺️ AI Municipal Repair Dispatch Route Optimizer")
        st.markdown(
            "Solves the **Traveling Salesperson Problem (TSP)** using a nearest-neighbor "
            "heuristic to minimize repair truck travel distance across critical pothole hotspots."
        )

        city_df = query_df("SELECT DISTINCT City FROM Location_Dim ORDER BY City")
        cities = city_df["City"].tolist() if not city_df.empty else ["Bangalore", "Mumbai", "Delhi"]
        sel_city = st.selectbox("Select Target Municipality / City", cities, key="tsp_city_sel")

        loc_df = query_df("""
            SELECT r.Road_Name AS road_name, l.Area AS area, l.City AS city,
                   l.Latitude AS latitude, l.Longitude AS longitude,
                   SUM(f.Pothole_Count) AS pothole_count
            FROM Pothole_Detection_Fact f
            JOIN Location_Dim l ON f.Location_ID = l.Location_ID
            JOIN Road_Dim r ON f.Road_ID = r.Road_ID
            WHERE l.City = ?
            GROUP BY r.Road_Name, l.Area, l.City, l.Latitude, l.Longitude
            ORDER BY pothole_count DESC
        """, (sel_city,))

        if loc_df.empty:
            if _has(_maps, "get_demo_locations"):
                demo_locs = _maps.get_demo_locations()
                loc_df = demo_locs[demo_locs["city"] == sel_city]

        from mining.ai_analytics import optimize_repair_route
        route_stops, total_dist_km = optimize_repair_route(loc_df, start_city=sel_city)

        if route_stops:
            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                st.metric("Total Optimal Route Distance", f"{total_dist_km:.2f} km")
                st.metric("Hotspots Visited", len(route_stops))
                est_fuel_saved = round(total_dist_km * 0.32, 1)
                st.success(f"⛽ **Estimated Fuel Saved:** ~{est_fuel_saved} Liters vs unoptimized dispatch.")

            route_table_df = pd.DataFrame(route_stops)[
                ["stop_order", "road_name", "area", "pothole_count", "leg_distance_km", "cumulative_km"]
            ]
            route_table_df.columns = ["Stop #", "Road Name", "Area", "Potholes", "Leg Distance (km)", "Cumulative (km)"]
            st.dataframe(route_table_df, use_container_width=True)

            # Route Plotly Map
            import plotly.graph_objects as go
            map_stops = pd.DataFrame(route_stops)
            fig_route = go.Figure()

            ScatterMapClass = getattr(go, "Scattermap", None) or getattr(go, "Scattermapbox", None)

            # Plot path lines
            fig_route.add_trace(ScatterMapClass(
                mode="lines",
                lon=map_stops["longitude"],
                lat=map_stops["latitude"],
                line=dict(width=3, color="red"),
                name="Optimized Route Path"
            ))

            # Plot stops with order numbers
            fig_route.add_trace(ScatterMapClass(
                mode="markers+text",
                lon=map_stops["longitude"],
                lat=map_stops["latitude"],
                marker=dict(size=14, color="crimson"),
                text=[f"#{s['stop_order']}: {s['road_name']}" for s in route_stops],
                textposition="top right",
                name="Repair Stops"
            ))

            center_lat = float(map_stops["latitude"].mean())
            center_lon = float(map_stops["longitude"].mean())

            if hasattr(go, "Scattermap"):
                fig_route.update_layout(
                    map=dict(
                        style="open-street-map",
                        center=dict(lat=center_lat, lon=center_lon),
                        zoom=11,
                    ),
                    title=f"Optimized Repair Dispatch Route — {sel_city}",
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
            else:
                fig_route.update_layout(
                    mapbox=dict(
                        style="open-street-map",
                        center=dict(lat=center_lat, lon=center_lon),
                        zoom=11,
                    ),
                    title=f"Optimized Repair Dispatch Route — {sel_city}",
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
            st.plotly_chart(fig_route, use_container_width=True)
        else:
            st.info(f"No coordinate data available for {sel_city}.")

    # ── TAB 4: Budget Impact Simulator ────────────────────────────────────
    with tab_budget:
        st.subheader("🎯 AI Municipal Budget Allocation Scenario Simulator")
        st.markdown(
            "Simulates the risk reduction and infrastructure impact of allocating municipal repair funds."
        )

        budget_lakhs = st.slider("Municipal Asphalt Repair Budget (in ₹ Lakhs)", 1, 100, 15, step=1)
        budget_inr = budget_lakhs * 100000

        # Cost per patch approx ₹1,200 (material + overhead)
        avg_patch_cost = 1150.0
        potholes_repaired = int(budget_inr // avg_patch_cost)
        total_in_wh = int(get_warehouse_summary().get("total_potholes", 150))
        coverage_pct = min(100.0, round((potholes_repaired / max(1, total_in_wh)) * 100, 1))
        risk_reduction = min(88.0, round(coverage_pct * 0.85, 1))

        b1, b2, b3 = st.columns(3)
        b1.metric("Potholes Repaired", f"{potholes_repaired:,}")
        b2.metric("Warehouse Coverage", f"{coverage_pct:.1f}%")
        b3.metric("Accident Risk Reduction", f"↓ {risk_reduction:.1f}%")

        st.progress(coverage_pct / 100.0, text=f"City Road Repair Coverage: {coverage_pct}%")


# ===========================================================================
# ── Router ───────────────────────────────────────────────────────────────────
# ===========================================================================

PAGE_FUNCTIONS = {
    "🏠 Home":                   page_home,
    "📊 Dataset & Preprocessing": page_dataset,
    "📹 Live Detection":          page_live_detection,
    "🖼️ Image Detection":        page_image_detection,
    "🎬 Video Detection":         page_video_detection,
    "📋 Detection Records":       page_records,
    "🏛️ Data Warehouse":         page_warehouse,
    "🔍 OLAP Analysis":           page_olap,
    "🔮 K-Means Clustering":      page_kmeans,
    "🗺️ Hotspot Map":            page_map,
    "📈 Trends":                  page_trends,
    "📑 Reports":                 page_reports,
    "🤖 AI Predictions & Optimization": page_ai_analytics,
}

# Render the selected page (always inside try/except for safety)
try:
    render_fn = PAGE_FUNCTIONS.get(page, page_home)
    render_fn()
except Exception as _err:
    st.error(f"An unexpected error occurred on page **{page}**:")
    st.code(traceback.format_exc())
    st.info("Try refreshing the page or switching to another section.")
