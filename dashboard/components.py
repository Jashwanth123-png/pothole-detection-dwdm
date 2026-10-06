"""
dashboard/components.py
------------------------
Reusable Streamlit UI component functions for the Pothole Detection DWDM app.

Each function renders one logical UI block.  They use st.* calls directly and
return data (dicts / values) only where the caller needs to act on user input.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import date, datetime
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Demo Banner
# ---------------------------------------------------------------------------

def show_demo_banner() -> None:
    """
    Display a status banner.
    - If models/best.pt exists (>5 MB), show a SUCCESS banner: real RDD2022 model active.
    - Otherwise show a WARNING banner: demo data mode, real model required.
    """
    import os
    from pathlib import Path

    # Locate models/best.pt relative to the dashboard package
    _here = Path(__file__).resolve().parent.parent  # project root
    best_pt = _here / "models" / "best.pt"
    real_model_ready = best_pt.exists() and best_pt.stat().st_size > 5_000_000

    if real_model_ready:
        sz_mb = best_pt.stat().st_size / (1024 * 1024)
        st.success(
            f"""
✅ **REAL RDD2022 MODEL ACTIVE**

Running with the **trained YOLOv8n model** (`models/best.pt`, {sz_mb:.0f} MB)
trained on the **RDD2022 D40 Pothole** dataset.

| Property | Value |
|----------|-------|
| Model | YOLOv8n (YOLOv8 Nano) |
| Dataset | RDD2022 — D40 Pothole class |
| Class mapping | 0 = D40 Pothole |
| Model file | `models/best.pt` |
            """,
            icon="✅",
        )
    else:
        st.warning(
            """
⚠️ **DEMO DATA MODE — NOT RDD2022 RESULTS**

This dashboard is running with **synthetic demo data** for demonstration purposes.
Real results require the **RDD2022 dataset** and a **trained YOLO model**.

To use real data:
1. Ensure RDD2022 data is prepared in `data/processed/split/`
2. Run training: `python ml/train_rdd2022.py --epochs 50`
3. The app will automatically switch to real model once `models/best.pt` is ready
            """,
            icon="⚠️",
        )



# ---------------------------------------------------------------------------
# 2. KPI Cards
# ---------------------------------------------------------------------------

def show_kpi_cards(summary: dict) -> None:
    """
    Render five metric cards in a single row using st.columns.

    Parameters
    ----------
    summary : dict
        Keys (all optional — defaults to 0/N/A):
            total_detections, total_potholes, high_severity,
            unique_roads, unique_cities
    """
    if not summary:
        summary = {}

    def _safe_int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    total_det   = _safe_int(summary.get("total_detections", 0))
    total_pot   = _safe_int(summary.get("total_potholes", 0))
    high_sev    = _safe_int(summary.get("high_severity", 0))
    unique_roads = _safe_int(summary.get("unique_roads", 0))
    unique_cities = _safe_int(summary.get("unique_cities", 0))

    # Deltas (percentage change) — optional
    det_delta = summary.get("detections_delta")
    pot_delta = summary.get("potholes_delta")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="📷 Total Detections",
            value=f"{total_det:,}",
            delta=f"{det_delta:+,}" if det_delta is not None else None,
        )
    with col2:
        st.metric(
            label="🕳️ Total Potholes",
            value=f"{total_pot:,}",
            delta=f"{pot_delta:+,}" if pot_delta is not None else None,
        )
    with col3:
        pct = (high_sev / total_pot * 100) if total_pot > 0 else 0
        st.metric(
            label="🔴 High Severity",
            value=f"{high_sev:,}",
            delta=f"{pct:.1f}% of total",
            delta_color="inverse",
        )
    with col4:
        st.metric(label="🛣️ Roads Monitored", value=f"{unique_roads:,}")
    with col5:
        st.metric(label="🏙️ Cities Covered", value=f"{unique_cities:,}")


# ---------------------------------------------------------------------------
# 3. Detection Record Form
# ---------------------------------------------------------------------------

def show_detection_record_form() -> Optional[dict]:
    """
    Render a manual detection-record entry form.

    Returns
    -------
    dict  – form values when the user clicks Submit, else None.
    """
    st.subheader("➕ Add Manual Detection Record")

    with st.form("detection_record_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        # ── Location ──────────────────────────────────────────────────────
        with col1:
            city = st.text_input("City *", placeholder="e.g. Mumbai")
            area = st.text_input("Area", placeholder="e.g. Andheri")
            latitude = st.number_input(
                "Latitude", min_value=-90.0, max_value=90.0,
                value=19.076, format="%.6f",
            )
            longitude = st.number_input(
                "Longitude", min_value=-180.0, max_value=180.0,
                value=72.877, format="%.6f",
            )

        # ── Road ──────────────────────────────────────────────────────────
        with col2:
            road_name = st.text_input("Road Name *", placeholder="e.g. SV Road")
            road_type = st.selectbox(
                "Road Type",
                ["National Highway", "State Highway", "District Road",
                 "Urban Road", "Rural Road", "Other"],
            )
            severity = st.selectbox(
                "Severity *", ["Low", "Medium", "High"]
            )

        # ── Detection ─────────────────────────────────────────────────────
        with col3:
            confidence = st.slider(
                "Confidence Score", min_value=0.0, max_value=1.0,
                value=0.80, step=0.01,
            )
            pothole_count = st.number_input(
                "Pothole Count *", min_value=1, max_value=9999, value=1
            )
            source_type = st.selectbox(
                "Source Type",
                ["YOLO Image", "YOLO Video", "YOLO Webcam", "Manual Entry"],
            )
            detection_date = st.date_input("Detection Date", value=date.today())

        submitted = st.form_submit_button("💾 Save Record", type="primary")

        if submitted:
            if not city.strip():
                st.error("City is required.")
                return None
            if not road_name.strip():
                st.error("Road Name is required.")
                return None

            return {
                "city": city.strip(),
                "area": area.strip() or "Unknown",
                "latitude": float(latitude),
                "longitude": float(longitude),
                "road_name": road_name.strip(),
                "road_type": road_type,
                "severity": severity,
                "confidence": float(confidence),
                "pothole_count": int(pothole_count),
                "source_type": source_type,
                "detection_date": str(detection_date),
            }

    return None


# ---------------------------------------------------------------------------
# 4. Model Status
# ---------------------------------------------------------------------------

def show_model_status(model_path: str) -> None:
    """
    Display a coloured indicator showing whether the YOLO model file exists.

    Parameters
    ----------
    model_path : str
        Absolute or relative path to the model weights file (.pt).
    """
    import os

    st.subheader("🤖 YOLO Model Status")

    if model_path and os.path.isfile(model_path):
        st.success(f"✅ Model found: `{model_path}`")
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        st.caption(f"File size: {size_mb:.1f} MB")
    else:
        st.error(
            f"""
❌ **Model not found** at: `{model_path}`

**How to get a model:**
1. **Train your own** – run `python training/train_yolo.py` after placing RDD2022 data in `data/raw/`
2. **Download pre-trained** – place a YOLOv8 `.pt` file (e.g. `yolov8n.pt`) in the `models/` directory
3. **Use a checkpoint** – copy any existing `.pt` file to `{model_path}`

Detection pages will be **disabled** until a model is available.
            """
        )


# ---------------------------------------------------------------------------
# 5. Preprocessing Report
# ---------------------------------------------------------------------------

def show_preprocessing_report(report_dict: dict) -> None:
    """
    Display preprocessing statistics in a formatted table.

    Parameters
    ----------
    report_dict : dict
        Arbitrary key→value pairs produced by the preprocessing pipeline,
        e.g. { 'total_images': 5000, 'filtered_images': 4800, ... }
    """
    st.subheader("📋 Preprocessing Report")

    if not report_dict:
        st.info("No preprocessing report available yet. Run the pipeline first.")
        return

    # Build a tidy two-column table
    rows = []
    for key, value in report_dict.items():
        label = key.replace("_", " ").title()
        rows.append({"Metric": label, "Value": value})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Highlight success / failure counts if present
    success_keys = [k for k in report_dict if "success" in k.lower()]
    failure_keys = [k for k in report_dict if "fail" in k.lower() or "error" in k.lower()]

    if success_keys:
        total_success = sum(
            int(report_dict[k]) for k in success_keys
            if str(report_dict[k]).isdigit()
        )
        st.metric("✅ Success Records", total_success)

    if failure_keys:
        total_fail = sum(
            int(report_dict[k]) for k in failure_keys
            if str(report_dict[k]).isdigit()
        )
        st.metric("❌ Failed Records", total_fail, delta_color="inverse")


# ---------------------------------------------------------------------------
# 6. OLAP Controls
# ---------------------------------------------------------------------------

def show_olap_controls(
    cities: Optional[list] = None,
    years: Optional[list] = None,
    roads: Optional[list] = None,
) -> dict:
    """
    Render OLAP filter controls in a sidebar expander.

    Parameters
    ----------
    cities : list[str], optional   – available city names for the dropdown
    years  : list[int], optional   – available years for the dropdown
    roads  : list[str], optional   – available road names for the dropdown

    Returns
    -------
    dict with keys:
        operation  – 'Roll-Up' | 'Drill-Down' | 'Slice' | 'Dice'
        severity   – list[str]
        city       – str or 'All'
        year       – int or 'All'
        road_name  – str or 'All'
    """
    cities = cities or []
    years  = years  or []
    roads  = roads  or []

    with st.sidebar.expander("🔍 OLAP Filters", expanded=True):
        operation = st.selectbox(
            "OLAP Operation",
            ["Roll-Up", "Drill-Down", "Slice", "Dice"],
            key="olap_operation",
        )

        severity_opts = st.multiselect(
            "Severity",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            key="olap_severity",
        )

        city_opts = ["All"] + sorted(cities) if cities else ["All"]
        city = st.selectbox("City", city_opts, key="olap_city")

        year_opts = ["All"] + sorted([str(y) for y in years], reverse=True) if years else ["All"]
        year = st.selectbox("Year", year_opts, key="olap_year")

        road_opts = ["All"] + sorted(roads) if roads else ["All"]
        road = st.selectbox("Road", road_opts, key="olap_road")

    return {
        "operation": operation,
        "severity": severity_opts,
        "city": city,
        "year": year if year == "All" else int(year),
        "road_name": road,
    }


# ---------------------------------------------------------------------------
# 7. Cluster Controls
# ---------------------------------------------------------------------------

def show_cluster_controls(max_k: int = 10) -> int:
    """
    Render K-Means parameter controls (K slider).

    Parameters
    ----------
    max_k : int
        Maximum number of clusters (upper bound of the slider).

    Returns
    -------
    int  – selected K value
    """
    st.subheader("⚙️ K-Means Parameters")

    col1, col2 = st.columns([2, 3])
    with col1:
        k = st.slider(
            "Number of Clusters (K)",
            min_value=2,
            max_value=max(max_k, 2),
            value=3,
            step=1,
            help="Choose how many clusters the algorithm should find.",
        )
    with col2:
        st.markdown(
            f"""
**Selected K = {k}**

The algorithm will partition roads into **{k} distinct clusters** based on:
- `pothole_count` — number of potholes detected
- `detection_frequency` — how often this road is monitored

> Cluster labels are **neutral integers** (0, 1, 2 …) and do not imply any
> value judgment — they are assigned purely by proximity in feature space.
            """
        )

    return k
