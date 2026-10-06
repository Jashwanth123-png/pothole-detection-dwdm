"""
dashboard/interactive.py
========================
Advanced interactive chart builders for the Pothole Detection DWDM Dashboard.

All functions return Plotly figure objects that are rendered via st.plotly_chart().
Charts use the full Pothole_Detection_Fact flat DataFrame as input (already joined).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Color palette ─────────────────────────────────────────────────────────────
SEV_COLORS = {
    "High": "#e74c3c",
    "Medium": "#f39c12",
    "Low": "#27ae60",
}
CITY_PALETTE = px.colors.qualitative.Set2
ROAD_TYPE_COLORS = {"Highway": "#3498db", "Urban": "#9b59b6", "Rural": "#1abc9c"}


# ── 1. KPI GAUGE (single value) ───────────────────────────────────────────────
def make_gauge(value: float, title: str, max_val: float,
               color: str = "#e74c3c") -> go.Figure:
    """Speedometer-style KPI gauge with a needle."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": title, "font": {"size": 14}},
        delta={"reference": max_val * 0.5, "increasing": {"color": color}},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, max_val * 0.33], "color": "#2ecc71"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "#f39c12"},
                {"range": [max_val * 0.66, max_val], "color": "#e74c3c"},
            ],
            "threshold": {
                "line": {"color": "#c0392b", "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=200)
    return fig


# ── 2. ANIMATED MONTHLY BAR RACE ─────────────────────────────────────────────
def make_animated_monthly_trend(df: pd.DataFrame) -> go.Figure:
    """
    Animated stacked-bar chart showing potholes per month, animated
    quarter by quarter so viewers see the trend build up over time.
    """
    monthly = (
        df.groupby(["Quarter", "Month", "Month_Name", "Severity_Level"], as_index=False)
        .agg(pothole_count=("Pothole_Count", "sum"))
    )
    monthly["Label"] = "Q" + monthly["Quarter"].astype(str) + " – " + monthly["Month_Name"]

    fig = px.bar(
        monthly,
        x="Month_Name", y="pothole_count",
        color="Severity_Level",
        animation_frame="Quarter",
        animation_group="Month_Name",
        barmode="stack",
        color_discrete_map=SEV_COLORS,
        title="📅 Monthly Pothole Trend — Animated by Quarter",
        labels={"pothole_count": "Potholes", "Month_Name": "Month"},
        text="pothole_count",
        range_y=[0, monthly["pothole_count"].max() * 2.5],
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        legend_title="Severity",
        xaxis=dict(categoryorder="array",
                   categoryarray=["January","February","March","April","May","June",
                                  "July","August","September","October","November","December"]),
        height=400,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1a2e",
        font_color="white",
    )
    return fig


# ── 3. SUNBURST OLAP DRILLDOWN ────────────────────────────────────────────────
def make_sunburst_olap(df: pd.DataFrame) -> go.Figure:
    """
    Interactive sunburst: click a City to drill into Areas,
    click an Area to drill into Roads. Click center to roll up.
    """
    agg = (
        df.groupby(["City", "Area", "Road_Name"], as_index=False)
        .agg(pothole_count=("Pothole_Count", "sum"))
    )

    fig = px.sunburst(
        agg,
        path=["City", "Area", "Road_Name"],
        values="pothole_count",
        color="pothole_count",
        color_continuous_scale="Oranges",
        title="🔍 OLAP Sunburst — Click to Drill-Down / Centre to Roll-Up",
        branchvalues="total",
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Potholes: %{value}<br>%{percentParent:.1%} of parent<extra></extra>",
        textinfo="label+percent parent",
        insidetextorientation="radial",
    )
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#0f1117",
        font_color="white",
        coloraxis_showscale=False,
    )
    return fig


# ── 4. TREEMAP ────────────────────────────────────────────────────────────────
def make_treemap(df: pd.DataFrame, split_by: str = "Severity_Level") -> go.Figure:
    """Zoomable treemap with either severity or road-type colour split."""
    agg = (
        df.groupby(["City", "Area", split_by], as_index=False)
        .agg(pothole_count=("Pothole_Count", "sum"))
    )
    color_map = SEV_COLORS if split_by == "Severity_Level" else ROAD_TYPE_COLORS

    fig = px.treemap(
        agg,
        path=["City", "Area", split_by],
        values="pothole_count",
        color=split_by,
        color_discrete_map=color_map,
        title=f"🗂️ Treemap — City → Area → {split_by.replace('_', ' ')} (Click to Zoom)",
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Potholes: %{value}<extra></extra>",
        textinfo="label+value",
    )
    fig.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#0f1117",
        font_color="white",
    )
    return fig


# ── 5. 3-D SCATTER (K-MEANS CLUSTER SPACE) ───────────────────────────────────
def make_3d_cluster_scatter(clustered_df: pd.DataFrame) -> go.Figure:
    """
    3-D interactive scatter plot of K-Means clustering results.
    Users can rotate, zoom, and hover for full road details.
    """
    required = {"pothole_count", "detection_frequency", "avg_confidence", "cluster_label"}
    if not required.issubset(clustered_df.columns):
        return go.Figure().add_annotation(text="Run K-Means first.", showarrow=False)

    severity_col = "severity_score" if "severity_score" in clustered_df.columns else "pothole_count"
    hover_cols = [c for c in ["road_name", "city", "area"] if c in clustered_df.columns]

    fig = px.scatter_3d(
        clustered_df,
        x="pothole_count",
        y="detection_frequency",
        z=severity_col,
        color=clustered_df["cluster_label"].astype(str),
        size="pothole_count",
        size_max=18,
        opacity=0.82,
        hover_name=hover_cols[0] if hover_cols else None,
        hover_data={c: True for c in hover_cols[1:]},
        title="🔮 3D K-Means Cluster Space (Drag to Rotate · Scroll to Zoom)",
        labels={
            "pothole_count": "Pothole Count",
            "detection_frequency": "Detection Freq",
            severity_col: "Severity Score",
            "color": "Cluster",
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(
        height=550,
        paper_bgcolor="#0f1117",
        scene=dict(
            bgcolor="#1a1a2e",
            xaxis=dict(backgroundcolor="#1a1a2e", color="white", gridcolor="#333"),
            yaxis=dict(backgroundcolor="#1a1a2e", color="white", gridcolor="#333"),
            zaxis=dict(backgroundcolor="#1a1a2e", color="white", gridcolor="#333"),
        ),
        font_color="white",
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ── 6. HEATMAP CALENDAR ───────────────────────────────────────────────────────
def make_heatmap_calendar(df: pd.DataFrame) -> go.Figure:
    """
    Day-of-week × Month heatmap showing detection density —
    reveals which days / months are highest risk.
    """
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    valid_months = [m for m in month_order if m in df["Month_Name"].values]
    pivot = (
        df.groupby(["Day_Name", "Month_Name"], as_index=False)
        .agg(potholes=("Pothole_Count", "sum"))
        .pivot(index="Day_Name", columns="Month_Name", values="potholes")
        .reindex(index=day_order)
        .reindex(columns=valid_months)
        .fillna(0)
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        showscale=True,
        hoverongaps=False,
        hovertemplate="<b>%{y} · %{x}</b><br>Potholes: %{z}<extra></extra>",
        colorbar=dict(title="Potholes", tickfont=dict(color="white")),
    ))
    fig.update_layout(
        title="🗓️ Detection Heatmap — Day of Week × Month",
        xaxis_title="Month", yaxis_title="Day of Week",
        height=380,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1a2e",
        font_color="white",
        xaxis=dict(color="white"),
        yaxis=dict(color="white"),
        margin=dict(l=80, r=20, t=60, b=40),
    )
    return fig


# ── 7. TIME-OF-DAY POLAR ROSE CHART ──────────────────────────────────────────
def make_polar_time_chart(df: pd.DataFrame) -> go.Figure:
    """
    Polar / windrose chart showing pothole detections by hour of day.
    Reveals peak risk windows (morning commute, night, etc.).
    """
    hourly = (
        df.groupby(["Hour", "Severity_Level"], as_index=False)
        .agg(count=("Pothole_Count", "sum"))
    )
    hourly["angle"] = hourly["Hour"] * 15  # 360 / 24 = 15 degrees per hour

    fig = go.Figure()
    for sev, color in SEV_COLORS.items():
        sub = hourly[hourly["Severity_Level"] == sev]
        fig.add_trace(go.Barpolar(
            r=sub["count"],
            theta=sub["Hour"].astype(str) + ":00",
            name=sev,
            marker_color=color,
            opacity=0.85,
            hovertemplate=f"<b>{sev}</b><br>Hour: %{{theta}}<br>Potholes: %{{r}}<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#1a1a2e",
            radialaxis=dict(visible=True, color="white", gridcolor="#444"),
            angularaxis=dict(
                tickvals=list(range(24)),
                ticktext=[f"{h}:00" for h in range(24)],
                color="white",
                direction="clockwise",
                gridcolor="#444",
            ),
        ),
        showlegend=True,
        legend_title="Severity",
        title="🕐 Hourly Risk Polar Chart — Peak Detection Times",
        height=500,
        paper_bgcolor="#0f1117",
        font_color="white",
        margin=dict(l=50, r=50, t=70, b=50),
    )
    return fig


# ── 8. CITY COMPARISON RADAR ──────────────────────────────────────────────────
def make_city_radar(df: pd.DataFrame) -> go.Figure:
    """
    Multi-axis radar / spider chart comparing cities on 5 dimensions:
    Total Potholes, Avg Confidence, High Severity %, Unique Roads, Unique Areas.
    """
    cities = df["City"].unique().tolist()

    city_stats = []
    for city in cities:
        sub = df[df["City"] == city]
        city_stats.append({
            "City": city,
            "Total Potholes": sub["Pothole_Count"].sum(),
            "Avg Confidence": sub["Confidence"].mean() * 100,
            "High Sev %": (sub["Severity_Level"] == "High").mean() * 100,
            "Unique Roads": sub["Road_Name"].nunique(),
            "Unique Areas": sub["Area"].nunique(),
        })
    stats_df = pd.DataFrame(city_stats)

    # Normalize each metric 0-100
    metrics = ["Total Potholes", "Avg Confidence", "High Sev %", "Unique Roads", "Unique Areas"]
    for m in metrics:
        col_max = stats_df[m].max()
        if col_max > 0:
            stats_df[m + "_n"] = (stats_df[m] / col_max) * 100
        else:
            stats_df[m + "_n"] = 0

    norm_metrics = [m + "_n" for m in metrics]

    fig = go.Figure()
    for i, row in stats_df.iterrows():
        vals = [row[m] for m in norm_metrics]
        vals += [vals[0]]  # close the polygon

        hover_text = "<br>".join(
            [f"{m}: {row[m]:.1f}" for m in metrics]
        )

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=row["City"],
            opacity=0.6,
            hovertemplate=f"<b>{row['City']}</b><br>{hover_text}<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#1a1a2e",
            radialaxis=dict(visible=True, range=[0, 110], color="white", gridcolor="#444"),
            angularaxis=dict(color="white", gridcolor="#444"),
        ),
        showlegend=True,
        legend_title="City",
        title="🕸️ Multi-City Radar — Normalized Damage Profiles",
        height=500,
        paper_bgcolor="#0f1117",
        font_color="white",
        margin=dict(l=60, r=60, t=70, b=60),
    )
    return fig


# ── 9. INTERACTIVE GEOSPATIAL DENSITY + PIN MAP ───────────────────────────────
def make_density_map(df: pd.DataFrame, map_type: str = "scatter") -> go.Figure:
    """
    Geospatial map with toggle between density heatmap and pin scatter modes.
    map_type: 'scatter' | 'density'
    """
    geo = (
        df.groupby(["City", "Area", "Road_Name", "Latitude", "Longitude", "Severity_Level"], as_index=False)
        .agg(pothole_count=("Pothole_Count", "sum"), detections=("Pothole_Count", "count"))
        .dropna(subset=["Latitude", "Longitude"])
    )

    if geo.empty:
        fig = go.Figure()
        fig.add_annotation(text="No geospatial data in warehouse.", showarrow=False)
        return fig

    title = "📍 Pothole Hotspot Pin Map — Hover for Details" if map_type == "scatter" else "🔥 Pothole Density Heatmap (Kernel Density Estimation)"

    fig = go.Figure()
    if map_type == "density":
        fig.add_trace(go.Densitymap(
            lat=geo["Latitude"],
            lon=geo["Longitude"],
            z=geo["pothole_count"],
            radius=35,
            colorscale="YlOrRd",
            showscale=True,
            hovertemplate="Potholes: %{z}<extra></extra>",
        ))
    else:
        sev_color_map = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
        for sev in ["High", "Medium", "Low"]:
            sub = geo[geo["Severity_Level"] == sev]
            if sub.empty:
                continue
            fig.add_trace(go.Scattermap(
                lat=sub["Latitude"],
                lon=sub["Longitude"],
                mode="markers",
                marker=dict(
                    size=(sub["pothole_count"] / geo["pothole_count"].max() * 20 + 8).tolist(),
                    color=sev_color_map[sev],
                    opacity=0.85,
                ),
                text=sub.apply(
                    lambda r: f"<b>{r['Road_Name']}</b><br>{r['City']} / {r['Area']}<br>Potholes: {r['pothole_count']}", axis=1
                ),
                hoverinfo="text",
                name=sev,
            ))

    fig.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=20.5, lon=78.9),
            zoom=4,
        ),
        title=title,
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#0f1117",
        font_color="white",
        legend_title="Severity",
    )
    return fig


# ── 10. SEVERITY FUNNEL CHART ─────────────────────────────────────────────────
def make_severity_funnel(df: pd.DataFrame) -> go.Figure:
    """
    Funnel showing how potholes taper from Low → Medium → High priority.
    """
    sev_order = ["Low", "Medium", "High"]
    sev_counts = df.groupby("Severity_Level")["Pothole_Count"].sum().reindex(sev_order, fill_value=0)

    fig = go.Figure(go.Funnel(
        y=sev_order,
        x=sev_counts.values,
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.88,
        marker=dict(color=[SEV_COLORS[s] for s in sev_order]),
        connector=dict(line=dict(color="#555", width=2)),
        hovertemplate="<b>%{y}</b><br>Potholes: %{x}<br>%{percentInitial:.1%} of total<extra></extra>",
    ))
    fig.update_layout(
        title="🔽 Severity Funnel — Detection Triage",
        height=350,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1a2e",
        font_color="white",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ── 11. ROAD TYPE vs SEVERITY GROUPED BAR ─────────────────────────────────────
def make_road_type_severity_chart(df: pd.DataFrame) -> go.Figure:
    """
    Interactive grouped bar chart: Road Type × Severity.
    Click legend items to show/hide severity bands.
    """
    agg = (
        df.groupby(["Road_Type", "Severity_Level"], as_index=False)
        .agg(pothole_count=("Pothole_Count", "sum"))
    )

    fig = px.bar(
        agg,
        x="Road_Type", y="pothole_count",
        color="Severity_Level",
        barmode="group",
        color_discrete_map=SEV_COLORS,
        text="pothole_count",
        title="🛣️ Road Type vs Severity (Click Legend to Filter)",
        labels={"pothole_count": "Potholes", "Road_Type": "Road Category"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=380,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1a2e",
        font_color="white",
        xaxis=dict(color="white"),
        yaxis=dict(color="white", gridcolor="#333"),
        legend_title="Severity",
    )
    return fig


# ── 12. CONFIDENCE DISTRIBUTION VIOLIN ───────────────────────────────────────
def make_confidence_violin(df: pd.DataFrame) -> go.Figure:
    """
    Violin + box + scatter jitter plot for model confidence per severity.
    Shows distribution shape better than a simple boxplot.
    """
    fig = go.Figure()
    for sev, color in SEV_COLORS.items():
        sub = df[df["Severity_Level"] == sev]["Confidence"]
        fig.add_trace(go.Violin(
            x=[sev] * len(sub),
            y=sub,
            name=sev,
            box_visible=True,
            meanline_visible=True,
            points="outliers",
            line_color=color,
            fillcolor=color,
            opacity=0.6,
            hovertemplate=f"<b>{sev}</b><br>Confidence: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title="🎻 Detection Confidence Distribution by Severity",
        yaxis_title="Confidence Score",
        showlegend=False,
        height=380,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1a2e",
        font_color="white",
        xaxis=dict(color="white"),
        yaxis=dict(color="white", gridcolor="#333", range=[0, 1]),
    )
    return fig


# ── 13. CITY BAR RACE (TOP CITIES ANIMATED) ───────────────────────────────────
def make_city_bar_race(df: pd.DataFrame) -> go.Figure:
    """
    Monthly cumulative pothole accumulation race across cities —
    animated bar chart showing which city accumulates the most damage fastest.
    """
    cumulative = []
    for month_num in sorted(df["Month"].unique()):
        sub = df[df["Month"] <= month_num]
        city_totals = sub.groupby("City")["Pothole_Count"].sum().reset_index()
        month_name = df[df["Month"] == month_num]["Month_Name"].iloc[0]
        city_totals["Month"] = month_num
        city_totals["Month_Name"] = month_name
        cumulative.append(city_totals)

    race_df = pd.concat(cumulative, ignore_index=True)

    fig = px.bar(
        race_df,
        x="Pothole_Count", y="City",
        animation_frame="Month_Name",
        orientation="h",
        color="City",
        color_discrete_sequence=CITY_PALETTE,
        text="Pothole_Count",
        range_x=[0, race_df["Pothole_Count"].max() * 1.1],
        title="🏎️ City Pothole Accumulation Race (Animated by Month)",
        labels={"Pothole_Count": "Cumulative Potholes", "City": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=420,
        showlegend=False,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1a2e",
        font_color="white",
        xaxis=dict(color="white", gridcolor="#333"),
        yaxis=dict(color="white"),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 700, "redraw": True},
                                  "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]),
            ],
            x=0.05, y=-0.12, showactive=True,
        )],
    )
    return fig
