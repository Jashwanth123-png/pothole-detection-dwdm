"""
dashboard/maps.py
-----------------
Geospatial mapping functions for the Pothole Detection DWDM dashboard.

Functions
---------
create_hotspot_map  : Plotly scatter_mapbox of pothole hotspots.
create_city_heatmap : Bar-chart fallback for city-level severity heat.
get_demo_locations  : Returns a DataFrame of demo Indian city locations.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Colour constants (consistent with charts.py)
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e74c3c",
}

# Default size scaling for map markers
_BASE_MARKER_SIZE = 8
_MAX_MARKER_SIZE = 40


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _empty_figure(message: str = "No data available") -> go.Figure:
    """Return a blank figure with a centred status message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="grey"),
    )
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _scale_sizes(series: pd.Series, base: int = _BASE_MARKER_SIZE,
                 max_size: int = _MAX_MARKER_SIZE) -> pd.Series:
    """Linearly scale a numeric series to [base, max_size] for marker sizing."""
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return pd.Series([base + (max_size - base) // 2] * len(series), index=series.index)
    scaled = base + (series - min_val) / (max_val - min_val) * (max_size - base)
    return scaled.fillna(base)


# ---------------------------------------------------------------------------
# 1. Hotspot Map
# ---------------------------------------------------------------------------

def create_hotspot_map(df: pd.DataFrame, is_demo: bool = False) -> go.Figure:
    """
    Scatter mapbox showing pothole hotspots across India.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns:
            city       (str)   – city name
            area       (str)   – area / neighbourhood name
            latitude   (float) – WGS-84 latitude
            longitude  (float) – WGS-84 longitude
            pothole_count (int)
            severity   (str)   – 'Low' | 'Medium' | 'High'
    is_demo : bool
        If True, overlays a DEMO watermark annotation.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None or df.empty:
        return _empty_figure("No location data available for the map.")

    required = {"latitude", "longitude", "pothole_count"}
    missing = required - set(df.columns)
    if missing:
        return _empty_figure(f"Map data missing columns: {missing}")

    df = df.copy()
    df["pothole_count"] = pd.to_numeric(df["pothole_count"], errors="coerce").fillna(1)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    if df.empty:
        return _empty_figure("No valid coordinate data found.")

    # Ensure severity column exists
    if "severity" not in df.columns:
        df["severity"] = "Medium"

    # Build hover text
    hover_parts = []
    for col in ["city", "area", "road_name"]:
        if col in df.columns:
            hover_parts.append(col)

    # Colour list matched to severity
    color_sequence = [SEVERITY_COLORS.get(s, "#95a5a6") for s in df["severity"]]

    # We use scatter_mapbox with explicit marker sizing
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
        size="pothole_count",
        size_max=_MAX_MARKER_SIZE,
        hover_name="city" if "city" in df.columns else None,
        hover_data={
            col: True
            for col in ["area", "pothole_count", "severity", "road_name"]
            if col in df.columns
        },
        zoom=4,
        center={"lat": 20.5937, "lon": 78.9629},   # centre of India
        mapbox_style="open-street-map",
        title="🗺️ Pothole Hotspot Map",
        labels={"severity": "Severity", "pothole_count": "Potholes"},
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="Severity",
        paper_bgcolor="white",
    )

    if is_demo:
        fig.add_annotation(
            text="⚠️ DEMO DATA — NOT RDD2022 RESULTS",
            xref="paper", yref="paper",
            x=0.5, y=0.02,
            showarrow=False,
            bgcolor="rgba(255,200,0,0.75)",
            bordercolor="orange",
            borderwidth=1,
            font=dict(size=13, color="darkred"),
        )

    return fig


# ---------------------------------------------------------------------------
# 2. City Heatmap (bar chart fallback)
# ---------------------------------------------------------------------------

def create_city_heatmap(df: pd.DataFrame, is_demo: bool = False) -> go.Figure:
    """
    Stacked bar chart showing pothole counts per city broken down by severity.

    Plotly does not provide a built-in choropleth for Indian cities without a
    custom GeoJSON, so a stacked bar chart is used as a practical alternative.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns: city_name (str), severity (str), pothole_count (int).
    is_demo : bool

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None or df.empty:
        return _empty_figure("No city data available.")

    df = df.copy()
    df["pothole_count"] = pd.to_numeric(df["pothole_count"], errors="coerce").fillna(0)

    # Pivot so each severity becomes a bar segment
    if "severity" in df.columns:
        pivot = (
            df.groupby(["city_name", "severity"])["pothole_count"]
            .sum()
            .reset_index()
        )
        fig = px.bar(
            pivot,
            x="city_name",
            y="pothole_count",
            color="severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            title="City-wise Pothole Heatmap (Stacked by Severity)",
            labels={"city_name": "City", "pothole_count": "Pothole Count"},
        )
    else:
        city_totals = df.groupby("city_name")["pothole_count"].sum().reset_index()
        fig = px.bar(
            city_totals,
            x="city_name",
            y="pothole_count",
            title="City-wise Pothole Heatmap",
            labels={"city_name": "City", "pothole_count": "Pothole Count"},
            color="pothole_count",
            color_continuous_scale="Reds",
        )

    fig.update_layout(
        xaxis_title="City",
        yaxis_title="Pothole Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_tickangle=-30,
        legend_title="Severity",
    )

    if is_demo:
        fig.add_annotation(
            text="DEMO DATA",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=30, color="rgba(200,0,0,0.2)"),
            textangle=-30,
        )

    return fig


# ---------------------------------------------------------------------------
# 3. Demo Locations DataFrame
# ---------------------------------------------------------------------------

def get_demo_locations() -> pd.DataFrame:
    """
    Return a DataFrame of synthetic Indian city/area/road pothole records
    for demonstration purposes when no real data is available.

    Each row represents one road segment with pothole information.
    The Is_Demo flag is always 1.

    Returns
    -------
    pd.DataFrame
        Columns: city, area, road_name, latitude, longitude,
                 pothole_count, severity, confidence, Is_Demo
    """
    records = [
        # ── Mumbai ──────────────────────────────────────────────────────────
        {
            "city": "Mumbai", "area": "Andheri",
            "road_name": "Andheri-Kurla Road",
            "latitude": 19.1136, "longitude": 72.8697,
            "pothole_count": 34, "severity": "High",
            "confidence": 0.91,
        },
        {
            "city": "Mumbai", "area": "Bandra",
            "road_name": "Linking Road",
            "latitude": 19.0544, "longitude": 72.8404,
            "pothole_count": 18, "severity": "Medium",
            "confidence": 0.78,
        },
        {
            "city": "Mumbai", "area": "Dadar",
            "road_name": "Tilak Bridge Road",
            "latitude": 19.0178, "longitude": 72.8478,
            "pothole_count": 9, "severity": "Low",
            "confidence": 0.65,
        },
        # ── Delhi ───────────────────────────────────────────────────────────
        {
            "city": "Delhi", "area": "Rohini",
            "road_name": "Rohini Sector 3 Road",
            "latitude": 28.7041, "longitude": 77.1025,
            "pothole_count": 27, "severity": "High",
            "confidence": 0.88,
        },
        {
            "city": "Delhi", "area": "Dwarka",
            "road_name": "Dwarka Sector 10 Marg",
            "latitude": 28.5921, "longitude": 77.0460,
            "pothole_count": 15, "severity": "Medium",
            "confidence": 0.74,
        },
        {
            "city": "Delhi", "area": "Saket",
            "road_name": "Press Enclave Road",
            "latitude": 28.5272, "longitude": 77.2090,
            "pothole_count": 6, "severity": "Low",
            "confidence": 0.61,
        },
        # ── Bangalore ───────────────────────────────────────────────────────
        {
            "city": "Bangalore", "area": "Koramangala",
            "road_name": "Koramangala 4th Block Road",
            "latitude": 12.9352, "longitude": 77.6245,
            "pothole_count": 22, "severity": "High",
            "confidence": 0.85,
        },
        {
            "city": "Bangalore", "area": "Whitefield",
            "road_name": "Hope Farm Junction Road",
            "latitude": 12.9698, "longitude": 77.7499,
            "pothole_count": 11, "severity": "Medium",
            "confidence": 0.72,
        },
        {
            "city": "Bangalore", "area": "Indiranagar",
            "road_name": "100 Feet Road",
            "latitude": 12.9784, "longitude": 77.6408,
            "pothole_count": 5, "severity": "Low",
            "confidence": 0.59,
        },
        # ── Chennai ─────────────────────────────────────────────────────────
        {
            "city": "Chennai", "area": "Tambaram",
            "road_name": "GST Road",
            "latitude": 12.9249, "longitude": 80.1000,
            "pothole_count": 19, "severity": "High",
            "confidence": 0.87,
        },
        {
            "city": "Chennai", "area": "Anna Nagar",
            "road_name": "Shanthi Colony Road",
            "latitude": 13.0850, "longitude": 80.2101,
            "pothole_count": 8, "severity": "Medium",
            "confidence": 0.69,
        },
        # ── Kolkata ─────────────────────────────────────────────────────────
        {
            "city": "Kolkata", "area": "Salt Lake",
            "road_name": "Sector V IT Road",
            "latitude": 22.5726, "longitude": 88.4308,
            "pothole_count": 13, "severity": "Medium",
            "confidence": 0.76,
        },
        {
            "city": "Kolkata", "area": "Park Street",
            "road_name": "Park Street",
            "latitude": 22.5510, "longitude": 88.3525,
            "pothole_count": 7, "severity": "Low",
            "confidence": 0.63,
        },
        # ── Hyderabad ────────────────────────────────────────────────────────
        {
            "city": "Hyderabad", "area": "Hitech City",
            "road_name": "Cyber Towers Road",
            "latitude": 17.4435, "longitude": 78.3772,
            "pothole_count": 25, "severity": "High",
            "confidence": 0.90,
        },
        {
            "city": "Hyderabad", "area": "Gachibowli",
            "road_name": "Financial District Road",
            "latitude": 17.4401, "longitude": 78.3489,
            "pothole_count": 12, "severity": "Medium",
            "confidence": 0.77,
        },
        {
            "city": "Hyderabad", "area": "Banjara Hills",
            "road_name": "Road No. 12",
            "latitude": 17.4126, "longitude": 78.4483,
            "pothole_count": 4, "severity": "Low",
            "confidence": 0.58,
        },
        # ── Pune ─────────────────────────────────────────────────────────────
        {
            "city": "Pune", "area": "Hinjewadi",
            "road_name": "Hinjewadi Phase 1 Road",
            "latitude": 18.5913, "longitude": 73.7389,
            "pothole_count": 30, "severity": "High",
            "confidence": 0.92,
        },
        {
            "city": "Pune", "area": "Kothrud",
            "road_name": "Karve Road",
            "latitude": 18.5074, "longitude": 73.8077,
            "pothole_count": 14, "severity": "Medium",
            "confidence": 0.73,
        },
        {
            "city": "Pune", "area": "Shivajinagar",
            "road_name": "FC Road",
            "latitude": 18.5308, "longitude": 73.8474,
            "pothole_count": 6, "severity": "Low",
            "confidence": 0.60,
        },
    ]

    df = pd.DataFrame(records)
    df["Is_Demo"] = 1
    return df
