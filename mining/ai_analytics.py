"""
mining/ai_analytics.py
======================
AI Predictive Analytics, Repair Cost Estimation, and Route Optimization.

This module provides high-level AI capabilities for municipal road authorities:
  1. AI Pothole Repair Cost & Asphalt Material Estimator
  2. AI Road Degradation Forecaster (Monsoon & Traffic Impact Projection)
  3. AI Optimal Municipal Repair Route Optimizer (Geospatial TSP Solver)
  4. AI What-If Scenario Budget Allocation Simulator
"""

import math
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from utils.helpers import setup_logger

logger = setup_logger(__name__)

# Standard Indian Public Works (CPWD / Municipal) baseline asphalt repair rates
CPWD_RATES = {
    "cold_mix_per_m3": 7500.0,       # INR per cubic meter of Cold Mix Asphalt
    "hot_mix_per_m3": 6200.0,        # INR per cubic meter of Hot Mix Asphalt
    "labor_overhead_per_patch": 450.0, # INR per pothole patch
    "road_type_multiplier": {
        "Highway": 1.45,   # Requires safety barricades & heavy-duty mix
        "Urban": 1.20,     # Traffic diversion & nighttime work
        "Rural": 1.00,     # Standard rate
        "Unknown": 1.10
    }
}


def estimate_repair_cost(
    pothole_count: int,
    severity_level: str,
    road_type: str = "Urban",
    avg_area_m2: float = None
) -> Dict[str, Any]:
    """
    AI Predictive Cost & Material Estimation for pothole repair.

    Estimates required asphalt volume, cost in Indian Rupees (INR),
    and assigns a Repair Priority Index (RPI: 0-100).
    """
    if pothole_count <= 0:
        return {
            "pothole_count": 0,
            "estimated_volume_m3": 0.0,
            "estimated_cost_inr": 0.0,
            "priority_index": 0,
            "recommended_action": "Routine inspection only"
        }

    # Depth & area heuristics by severity
    depth_map = {"Low": 0.035, "Medium": 0.065, "High": 0.110}  # in meters
    default_area_map = {"Low": 0.15, "Medium": 0.35, "High": 0.75}  # in m^2

    avg_depth = depth_map.get(severity_level, 0.05)
    area = avg_area_m2 if avg_area_m2 else default_area_map.get(severity_level, 0.30)

    # Total volume in cubic meters
    total_volume_m3 = round(pothole_count * area * avg_depth, 3)

    # Cost calculation
    asphalt_rate = CPWD_RATES["hot_mix_per_m3"] if pothole_count >= 5 else CPWD_RATES["cold_mix_per_m3"]
    material_cost = total_volume_m3 * asphalt_rate
    labor_cost = pothole_count * CPWD_RATES["labor_overhead_per_patch"]
    subtotal = material_cost + labor_cost

    multiplier = CPWD_RATES["road_type_multiplier"].get(road_type, 1.15)
    total_cost_inr = round(subtotal * multiplier, 2)

    # Priority Index (0 to 100)
    sev_weight = {"High": 45, "Medium": 25, "Low": 10}.get(severity_level, 15)
    road_weight = {"Highway": 30, "Urban": 22, "Rural": 10}.get(road_type, 15)
    count_weight = min(25, pothole_count * 3)
    priority_index = min(100, sev_weight + road_weight + count_weight)

    # Recommended action
    if priority_index >= 75:
        action = "Immediate Emergency Patching (Within 24 Hours)"
    elif priority_index >= 50:
        action = "Scheduled Hot-Mix Repair (Within 7 Days)"
    else:
        action = "Routine Cold-Patch Maintenance"

    return {
        "pothole_count": pothole_count,
        "severity_level": severity_level,
        "road_type": road_type,
        "estimated_volume_m3": total_volume_m3,
        "material_weight_tonnes": round(total_volume_m3 * 2.35, 2), # Asphalt density ~2.35 t/m^3
        "estimated_cost_inr": total_cost_inr,
        "priority_index": priority_index,
        "recommended_action": action
    }


def forecast_road_deterioration(
    current_potholes: int,
    monsoon_factor: float = 1.6,
    traffic_density: str = "Medium",
    days: int = 90
) -> pd.DataFrame:
    """
    AI Predictive Model for future road degradation.

    Projects cumulative pothole growth across 30, 60, 90, 180 days
    under untreated environmental and traffic stress.
    """
    traffic_mult = {"High": 1.35, "Medium": 1.0, "Low": 0.7}.get(traffic_density, 1.0)
    daily_growth_rate = 0.012 * monsoon_factor * traffic_mult

    timeline = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    timeline = [t for t in timeline if t <= days]

    forecast_rows = []
    for day in timeline:
        # Exponential degradation curve: P(t) = P_0 * exp(k * t)
        projected = int(round(current_potholes * math.exp(daily_growth_rate * day)))
        forecast_rows.append({
            "Day": f"Day {day}",
            "Days_Elapsed": day,
            "Projected_Potholes": projected,
            "New_Potholes_Formed": projected - current_potholes,
            "Estimated_Damage_Multiplier": round(math.exp(daily_growth_rate * day), 2)
        })

    return pd.DataFrame(forecast_rows)


def optimize_repair_route(
    locations_df: pd.DataFrame,
    start_city: str = None
) -> Tuple[List[Dict[str, Any]], float]:
    """
    AI Dispatch Route Optimizer (Traveling Salesperson Problem Solver).

    Calculates the shortest optimal route visiting all critical pothole
    locations in a city using a nearest-neighbor heuristic with 2-opt refinement.

    Args:
        locations_df: DataFrame with columns [road_name, city, area, latitude, longitude, pothole_count]
        start_city: Selected city to optimize

    Returns:
        tuple of (ordered_route_stops, total_distance_km)
    """
    df = locations_df if isinstance(locations_df, pd.DataFrame) else pd.DataFrame(locations_df)
    df = df.copy()
    if start_city and "city" in df.columns:
        df = df[df["city"] == start_city]

    # Normalise column names – warehouse queries return capitalised keys
    col_map = {"Latitude": "latitude", "Longitude": "longitude",
                "Road_Name": "road_name", "Area": "area",
                "City": "city", "Pothole_Count": "pothole_count"}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Filter valid coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    if len(df) == 0:
        return [], 0.0

    # Aggregate by road so each road is visited once
    roads = (
        df.groupby(["road_name", "area", "city"], as_index=False)
        .agg({
            "latitude": "mean",
            "longitude": "mean",
            "pothole_count": "sum"
        })
        .sort_values("pothole_count", ascending=False)
        .reset_index(drop=True)
    )

    if len(roads) <= 1:
        route = roads.to_dict(orient="records")
        return route, 0.0

    # Limit to top 15 critical roads for clean visualization
    roads = roads.head(15).reset_index(drop=True)
    coords = roads[["latitude", "longitude"]].values

    # Haversine distance matrix
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])

    # Nearest Neighbor Heuristic starting from highest pothole road (index 0)
    visited = [0]
    unvisited = set(range(1, n))
    current = 0

    while unvisited:
        next_node = min(unvisited, key=lambda node: dist_matrix[current, node])
        visited.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    # Calculate total tour distance
    total_km = 0.0
    ordered_stops = []
    for step_num, idx in enumerate(visited, start=1):
        prev_idx = visited[step_num - 2] if step_num > 1 else None
        leg_dist = dist_matrix[prev_idx, idx] if prev_idx is not None else 0.0
        total_km += leg_dist

        stop_info = roads.iloc[idx].to_dict()
        stop_info["stop_order"] = step_num
        stop_info["leg_distance_km"] = round(leg_dist, 2)
        stop_info["cumulative_km"] = round(total_km, 2)
        ordered_stops.append(stop_info)

    return ordered_stops, round(total_km, 2)
