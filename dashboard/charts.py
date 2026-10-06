"""
dashboard/charts.py
-------------------
Plotly chart factory functions for the Pothole Detection DWDM dashboard.

Each function returns a Plotly figure object that can be displayed with
st.plotly_chart(fig, use_container_width=True) in Streamlit.

All charts handle empty DataFrames gracefully and support a demo watermark.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Colour palette constants
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "Low": "#2ecc71",      # green
    "Medium": "#f39c12",   # orange
    "High": "#e74c3c",     # red
}

CLUSTER_SYMBOLS = ["circle", "square", "diamond", "cross", "x", "triangle-up",
                   "triangle-down", "pentagon", "hexagon", "star"]

DEMO_ANNOTATION = dict(
    text="DEMO DATA — NOT RDD2022 RESULTS",
    xref="paper", yref="paper",
    x=0.5, y=0.5,
    showarrow=False,
    font=dict(size=24, color="rgba(200,0,0,0.25)"),
    textangle=-30,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _empty_figure(message: str = "No data available") -> go.Figure:
    """Return a blank figure with a centred message — used when df is empty."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="grey"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
    )
    return fig


def _add_demo_watermark(fig: go.Figure) -> go.Figure:
    """Overlay a semi-transparent DEMO watermark annotation on any figure."""
    fig.add_annotation(**DEMO_ANNOTATION)
    return fig


# ---------------------------------------------------------------------------
# 1. Monthly Trend Chart
# ---------------------------------------------------------------------------

def create_monthly_trend_chart(df: pd.DataFrame, is_demo: bool = False) -> go.Figure:
    """
    Line chart of pothole detections grouped by Month-Year.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns: month_year (str), pothole_count (int),
        optionally severity (str).
    is_demo : bool
        If True, add a watermark annotation.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None or df.empty:
        return _empty_figure("No trend data available yet.")

    # Ensure pothole_count is numeric
    df = df.copy()
    df["pothole_count"] = pd.to_numeric(df["pothole_count"], errors="coerce").fillna(0)

    # Build figure — colour by severity if the column is present
    if "severity" in df.columns and df["severity"].nunique() > 0:
        fig = px.line(
            df,
            x="month_year",
            y="pothole_count",
            color="severity",
            color_discrete_map=SEVERITY_COLORS,
            markers=True,
            title="Monthly Pothole Detection Trend",
            labels={
                "month_year": "Month-Year",
                "pothole_count": "Pothole Count",
                "severity": "Severity",
            },
        )
    else:
        fig = px.line(
            df,
            x="month_year",
            y="pothole_count",
            markers=True,
            title="Monthly Pothole Detection Trend",
            labels={"month_year": "Month-Year", "pothole_count": "Pothole Count"},
            color_discrete_sequence=["#3498db"],
        )

    fig.update_layout(
        xaxis_title="Month-Year",
        yaxis_title="Pothole Count",
        legend_title="Severity",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(tickangle=-45)

    if is_demo:
        _add_demo_watermark(fig)

    return fig


# ---------------------------------------------------------------------------
# 2. Severity Donut Chart
# ---------------------------------------------------------------------------

def create_severity_donut(df: pd.DataFrame, is_demo: bool = False) -> go.Figure:
    """
    Donut chart showing the proportion of potholes by severity.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns: severity (str), pothole_count (int).
    is_demo : bool

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None or df.empty:
        return _empty_figure("No severity data available.")

    df = df.copy()
    df["pothole_count"] = pd.to_numeric(df["pothole_count"], errors="coerce").fillna(0)

    # Build colour list matched to the severity labels present
    colors = [SEVERITY_COLORS.get(s, "#95a5a6") for s in df["severity"]]

    fig = go.Figure(
        go.Pie(
            labels=df["severity"],
            values=df["pothole_count"],
            hole=0.45,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Severity Distribution",
        legend_title="Severity",
        showlegend=True,
    )

    if is_demo:
        _add_demo_watermark(fig)

    return fig


# ---------------------------------------------------------------------------
# 3. Top Roads Horizontal Bar Chart
# ---------------------------------------------------------------------------

def create_top_roads_bar(df: pd.DataFrame, top_n: int = 10, is_demo: bool = False) -> go.Figure:
    """
    Horizontal bar chart of the top N roads by pothole count.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns: road_name (str), pothole_count (int).
    top_n : int
        Number of roads to display (default 10).
    is_demo : bool

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None or df.empty:
        return _empty_figure("No road data available.")

    df = df.copy()
    df["pothole_count"] = pd.to_numeric(df["pothole_count"], errors="coerce").fillna(0)
    df = df.nlargest(top_n, "pothole_count")

    fig = px.bar(
        df,
        x="pothole_count",
        y="road_name",
        orientation="h",
        title=f"Top {top_n} Roads by Pothole Count",
        labels={"pothole_count": "Pothole Count", "road_name": "Road Name"},
        color="pothole_count",
        color_continuous_scale="Reds",
        text="pothole_count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Pothole Count",
        yaxis_title="Road Name",
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    if is_demo:
        _add_demo_watermark(fig)

    return fig


# ---------------------------------------------------------------------------
# 4. OLAP Roll-Up Grouped Bar Chart
# ---------------------------------------------------------------------------

def create_rollup_chart(
    road_df: pd.DataFrame,
    area_df: pd.DataFrame,
    city_df: pd.DataFrame,
    is_demo: bool = False,
) -> go.Figure:
    """
    Grouped bar chart showing three OLAP roll-up levels:
    Road → Area → City.

    Parameters
    ----------
    road_df : pd.DataFrame   columns: road_name, pothole_count
    area_df : pd.DataFrame   columns: area_name, pothole_count
    city_df : pd.DataFrame   columns: city_name, pothole_count
    is_demo : bool

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Validate inputs
    all_empty = all(
        (d is None or d.empty) for d in [road_df, area_df, city_df]
    )
    if all_empty:
        return _empty_figure("No OLAP roll-up data available.")

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Road Level", "Area Level", "City Level"],
        shared_yaxes=False,
    )

    # --- Road level ---
    if road_df is not None and not road_df.empty:
        road_df = road_df.copy()
        road_df["pothole_count"] = pd.to_numeric(road_df["pothole_count"], errors="coerce").fillna(0)
        top_roads = road_df.nlargest(8, "pothole_count")
        fig.add_trace(
            go.Bar(
                x=top_roads["road_name"],
                y=top_roads["pothole_count"],
                name="Road",
                marker_color="#3498db",
                showlegend=True,
            ),
            row=1, col=1,
        )

    # --- Area level ---
    if area_df is not None and not area_df.empty:
        area_df = area_df.copy()
        area_df["pothole_count"] = pd.to_numeric(area_df["pothole_count"], errors="coerce").fillna(0)
        fig.add_trace(
            go.Bar(
                x=area_df["area_name"],
                y=area_df["pothole_count"],
                name="Area",
                marker_color="#e67e22",
                showlegend=True,
            ),
            row=1, col=2,
        )

    # --- City level ---
    if city_df is not None and not city_df.empty:
        city_df = city_df.copy()
        city_df["pothole_count"] = pd.to_numeric(city_df["pothole_count"], errors="coerce").fillna(0)
        fig.add_trace(
            go.Bar(
                x=city_df["city_name"],
                y=city_df["pothole_count"],
                name="City",
                marker_color="#e74c3c",
                showlegend=True,
            ),
            row=1, col=3,
        )

    fig.update_layout(
        title_text="OLAP Roll-Up: Road → Area → City",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
    )
    fig.update_xaxes(tickangle=-35)

    if is_demo:
        _add_demo_watermark(fig)

    return fig


# ---------------------------------------------------------------------------
# 5. K-Means Cluster Scatter Plot
# ---------------------------------------------------------------------------

def create_kmeans_scatter(df: pd.DataFrame, is_demo: bool = False) -> go.Figure:
    """
    Scatter plot of K-Means clustering results.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns: pothole_count (int), detection_frequency (int/float),
        cluster_label (int or str).
    is_demo : bool

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None or df.empty:
        return _empty_figure("Run K-Means clustering to see results here.")

    df = df.copy()
    df["cluster_label"] = df["cluster_label"].astype(str)
    df["pothole_count"] = pd.to_numeric(df["pothole_count"], errors="coerce").fillna(0)
    df["detection_frequency"] = pd.to_numeric(df["detection_frequency"], errors="coerce").fillna(0)

    unique_clusters = sorted(df["cluster_label"].unique())
    symbol_map = {
        cl: CLUSTER_SYMBOLS[i % len(CLUSTER_SYMBOLS)]
        for i, cl in enumerate(unique_clusters)
    }

    fig = px.scatter(
        df,
        x="pothole_count",
        y="detection_frequency",
        color="cluster_label",
        symbol="cluster_label",
        symbol_map=symbol_map,
        hover_data=[col for col in ["road_name", "area_name", "city_name"] if col in df.columns],
        title="K-Means Clustering Results",
        labels={
            "pothole_count": "Pothole Count",
            "detection_frequency": "Detection Frequency",
            "cluster_label": "Cluster",
        },
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title="Cluster",
    )

    if is_demo:
        _add_demo_watermark(fig)

    return fig


# ---------------------------------------------------------------------------
# 6. Detection Timeline Scatter
# ---------------------------------------------------------------------------

def create_detection_timeline(df: pd.DataFrame, is_demo: bool = False) -> go.Figure:
    """
    Timeline scatter of detections over time, coloured by severity.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns: date (str/datetime), pothole_count (int),
        severity (str).
    is_demo : bool

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None or df.empty:
        return _empty_figure("No timeline data available.")

    df = df.copy()
    df["pothole_count"] = pd.to_numeric(df["pothole_count"], errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        return _empty_figure("No valid date data available for timeline.")

    color_col = "severity" if "severity" in df.columns else None

    fig = px.scatter(
        df,
        x="date",
        y="pothole_count",
        color=color_col,
        color_discrete_map=SEVERITY_COLORS if color_col else None,
        size="pothole_count",
        size_max=20,
        title="Detection Timeline",
        labels={
            "date": "Date",
            "pothole_count": "Pothole Count",
            "severity": "Severity",
        },
        hover_data=[c for c in ["road_name", "area_name", "city_name"] if c in df.columns],
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Date",
        yaxis_title="Pothole Count",
    )

    if is_demo:
        _add_demo_watermark(fig)

    return fig


# ---------------------------------------------------------------------------
# 7. Confidence Score Histogram
# ---------------------------------------------------------------------------

def create_confidence_histogram(df: pd.DataFrame, is_demo: bool = False) -> go.Figure:
    """
    Histogram of YOLO detection confidence scores.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns: confidence (float, 0–1).
    is_demo : bool

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None or df.empty:
        return _empty_figure("No confidence data available.")

    df = df.copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").dropna()

    if df["confidence"].empty:
        return _empty_figure("No valid confidence values found.")

    fig = px.histogram(
        df,
        x="confidence",
        nbins=20,
        title="Confidence Score Distribution",
        labels={"confidence": "Confidence Score", "count": "Frequency"},
        color_discrete_sequence=["#9b59b6"],
        range_x=[0, 1],
    )
    fig.update_layout(
        xaxis_title="Confidence Score (0–1)",
        yaxis_title="Frequency",
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.05,
    )
    # Add a vertical reference line at 0.5
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="0.5 threshold",
        annotation_position="top right",
    )

    if is_demo:
        _add_demo_watermark(fig)

    return fig


# ---------------------------------------------------------------------------
# 8. KPI Metrics Formatter
# ---------------------------------------------------------------------------

def create_kpi_metrics(summary_dict: dict) -> dict:
    """
    Format raw summary statistics into display-ready strings for KPI cards.

    Parameters
    ----------
    summary_dict : dict
        Keys (all optional):
            total_detections, total_potholes, high_severity,
            medium_severity, low_severity, unique_roads,
            unique_cities, avg_confidence, date_range

    Returns
    -------
    dict of str  — formatted values keyed by metric name
    """
    if not summary_dict:
        summary_dict = {}

    def _fmt_int(val, default="0"):
        try:
            return f"{int(val):,}"
        except (TypeError, ValueError):
            return default

    def _fmt_pct(val, total, default="N/A"):
        try:
            pct = (int(val) / int(total)) * 100
            return f"{int(val):,} ({pct:.1f}%)"
        except (TypeError, ValueError, ZeroDivisionError):
            return default

    def _fmt_float(val, decimals=3, default="N/A"):
        try:
            return f"{float(val):.{decimals}f}"
        except (TypeError, ValueError):
            return default

    total_detections = summary_dict.get("total_detections", 0)
    total_potholes = summary_dict.get("total_potholes", 0)
    high_sev = summary_dict.get("high_severity", 0)
    medium_sev = summary_dict.get("medium_severity", 0)
    low_sev = summary_dict.get("low_severity", 0)

    return {
        "Total Detections": _fmt_int(total_detections),
        "Total Potholes": _fmt_int(total_potholes),
        "High Severity": _fmt_pct(high_sev, total_potholes),
        "Medium Severity": _fmt_pct(medium_sev, total_potholes),
        "Low Severity": _fmt_pct(low_sev, total_potholes),
        "Unique Roads": _fmt_int(summary_dict.get("unique_roads", 0)),
        "Unique Cities": _fmt_int(summary_dict.get("unique_cities", 0)),
        "Avg Confidence": _fmt_float(summary_dict.get("avg_confidence", None)),
        "Date Range": str(summary_dict.get("date_range", "N/A")),
    }
