"""
warehouse/queries.py
====================
Analytical query functions for the Pothole Detection Data Warehouse.

All functions accept a live SQLAlchemy session and return plain Python
objects (lists of dicts, or a single dict) so that callers never need to
import SQLAlchemy themselves.

Typical usage::

    from warehouse.database import get_db_session
    from warehouse.queries  import get_detection_summary, get_top_roads

    with get_db_session() as session:
        summary = get_detection_summary(session)
        roads   = get_top_roads(session, n=5)
"""

import logging
from typing import List, Dict, Any

from sqlalchemy import func

from warehouse.models import (
    PotholeDetectionFact,
    DateDim,
    TimeDim,
    LocationDim,
    RoadDim,
    SeverityDim,
)
from utils.helpers import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: convert a SQLAlchemy row to a plain dict
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    """
    Convert a SQLAlchemy KeyedTuple / Row to a plain Python dict.

    Works for both ORM instances (via __table__) and query result rows
    produced by .with_entities() / label().

    Args:
        row: Any SQLAlchemy query result row.

    Returns:
        dict: {column_name: value, …}
    """
    if hasattr(row, "_asdict"):          # KeyedTuple / Row (SQLAlchemy 1.x / 2.x)
        return row._asdict()
    if hasattr(row, "__table__"):        # ORM model instance
        return {c.name: getattr(row, c.name) for c in row.__table__.columns}
    # Fallback: try __dict__
    return {k: v for k, v in row.__dict__.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────────────
# 1. All detections
# ─────────────────────────────────────────────────────────────────────────────

def get_all_detections(session) -> List[Dict[str, Any]]:
    """
    Retrieve every detection record joined with all dimension tables.

    Returns a flat list of dicts containing the most useful columns from
    both the fact table and its dimensions — ready for a DataFrame or table.

    Args:
        session: Active SQLAlchemy session.

    Returns:
        list[dict]: One dict per detection row with keys:
            Detection_ID, Date, Hour, Time_Period, City, Area, Latitude,
            Longitude, Road_Name, Road_Type, Severity_Level, Pothole_Count,
            Confidence, BBox_X1, BBox_Y1, BBox_X2, BBox_Y2,
            Source_Type, Is_Demo, Created_At
    """
    try:
        rows = (
            session.query(
                PotholeDetectionFact.Detection_ID,
                DateDim.Date,
                DateDim.Year,
                DateDim.Month,
                DateDim.Month_Name,
                TimeDim.Hour,
                TimeDim.Time_Period,
                LocationDim.City,
                LocationDim.Area,
                LocationDim.Latitude,
                LocationDim.Longitude,
                RoadDim.Road_Name,
                RoadDim.Road_Type,
                SeverityDim.Severity_Level,
                PotholeDetectionFact.Pothole_Count,
                PotholeDetectionFact.Confidence,
                PotholeDetectionFact.BBox_X1,
                PotholeDetectionFact.BBox_Y1,
                PotholeDetectionFact.BBox_X2,
                PotholeDetectionFact.BBox_Y2,
                PotholeDetectionFact.Source_Type,
                PotholeDetectionFact.Is_Demo,
                PotholeDetectionFact.Created_At,
            )
            .join(DateDim,     PotholeDetectionFact.Date_ID     == DateDim.Date_ID)
            .join(TimeDim,     PotholeDetectionFact.Time_ID     == TimeDim.Time_ID)
            .join(LocationDim, PotholeDetectionFact.Location_ID == LocationDim.Location_ID)
            .join(RoadDim,     PotholeDetectionFact.Road_ID     == RoadDim.Road_ID)
            .join(SeverityDim, PotholeDetectionFact.Severity_ID == SeverityDim.Severity_ID)
            .order_by(DateDim.Date.desc(), TimeDim.Hour.desc())
            .all()
        )
        result = [_row_to_dict(r) for r in rows]
        logger.debug(f"get_all_detections: returned {len(result)} rows.")
        return result

    except Exception as exc:
        logger.error(f"get_all_detections error: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 2. KPI summary
# ─────────────────────────────────────────────────────────────────────────────

def get_detection_summary(session) -> Dict[str, Any]:
    """
    Compute high-level KPI statistics for the dashboard.

    Runs five aggregation queries and returns a single dict with all KPIs.

    Args:
        session: Active SQLAlchemy session.

    Returns:
        dict with keys:
            total_detections  (int) – Number of detection events.
            total_potholes    (int) – Sum of all Pothole_Count values.
            avg_confidence    (float) – Mean confidence across all events.
            high_severity_count (int) – Events with Severity_Level == 'High'.
            road_count        (int) – Distinct roads detected on.
            location_count    (int) – Distinct city/area combinations.
    """
    try:
        # Total detection events
        total_det = session.query(func.count(PotholeDetectionFact.Detection_ID)).scalar() or 0

        # Sum of all pothole counts
        total_ph  = session.query(func.sum(PotholeDetectionFact.Pothole_Count)).scalar() or 0

        # Average model confidence
        avg_conf  = session.query(func.avg(PotholeDetectionFact.Confidence)).scalar() or 0.0

        # High-severity count (join to Severity_Dim)
        high_sev  = (
            session.query(func.count(PotholeDetectionFact.Detection_ID))
            .join(SeverityDim, PotholeDetectionFact.Severity_ID == SeverityDim.Severity_ID)
            .filter(SeverityDim.Severity_Level == "High")
            .scalar() or 0
        )

        # Distinct road count
        road_cnt  = session.query(func.count(func.distinct(PotholeDetectionFact.Road_ID))).scalar() or 0

        # Distinct location count
        loc_cnt   = session.query(func.count(func.distinct(PotholeDetectionFact.Location_ID))).scalar() or 0

        summary = {
            "total_detections"  : int(total_det),
            "total_potholes"    : int(total_ph),
            "avg_confidence"    : round(float(avg_conf), 4),
            "high_severity_count": int(high_sev),
            "road_count"        : int(road_cnt),
            "location_count"    : int(loc_cnt),
        }
        logger.debug(f"get_detection_summary: {summary}")
        return summary

    except Exception as exc:
        logger.error(f"get_detection_summary error: {exc}")
        return {
            "total_detections"  : 0,
            "total_potholes"    : 0,
            "avg_confidence"    : 0.0,
            "high_severity_count": 0,
            "road_count"        : 0,
            "location_count"    : 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Monthly trend
# ─────────────────────────────────────────────────────────────────────────────

def get_monthly_trend(session) -> List[Dict[str, Any]]:
    """
    Return total potholes grouped by calendar month and year.

    Useful for plotting a time-series bar/line chart.

    Args:
        session: Active SQLAlchemy session.

    Returns:
        list[dict]: Sorted chronologically, each dict has:
            Year (int), Month (int), Month_Name (str),
            Total_Potholes (int), Detection_Count (int)
    """
    try:
        rows = (
            session.query(
                DateDim.Year,
                DateDim.Month,
                DateDim.Month_Name,
                func.sum(PotholeDetectionFact.Pothole_Count).label("Total_Potholes"),
                func.count(PotholeDetectionFact.Detection_ID).label("Detection_Count"),
            )
            .join(DateDim, PotholeDetectionFact.Date_ID == DateDim.Date_ID)
            .group_by(DateDim.Year, DateDim.Month, DateDim.Month_Name)
            .order_by(DateDim.Year, DateDim.Month)
            .all()
        )
        result = [_row_to_dict(r) for r in rows]
        logger.debug(f"get_monthly_trend: {len(result)} months.")
        return result

    except Exception as exc:
        logger.error(f"get_monthly_trend error: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 4. Top roads by pothole count
# ─────────────────────────────────────────────────────────────────────────────

def get_top_roads(session, n: int = 10) -> List[Dict[str, Any]]:
    """
    Return the top N roads ranked by cumulative pothole count.

    Args:
        session: Active SQLAlchemy session.
        n       (int): Maximum number of roads to return (default 10).

    Returns:
        list[dict]: Each dict has:
            Road_Name (str), Road_Type (str),
            Total_Potholes (int), Detection_Count (int)
    """
    try:
        rows = (
            session.query(
                RoadDim.Road_Name,
                RoadDim.Road_Type,
                func.sum(PotholeDetectionFact.Pothole_Count).label("Total_Potholes"),
                func.count(PotholeDetectionFact.Detection_ID).label("Detection_Count"),
            )
            .join(RoadDim, PotholeDetectionFact.Road_ID == RoadDim.Road_ID)
            .group_by(RoadDim.Road_Name, RoadDim.Road_Type)
            .order_by(func.sum(PotholeDetectionFact.Pothole_Count).desc())
            .limit(n)
            .all()
        )
        result = [_row_to_dict(r) for r in rows]
        logger.debug(f"get_top_roads (n={n}): {len(result)} rows.")
        return result

    except Exception as exc:
        logger.error(f"get_top_roads error: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 5. Severity distribution
# ─────────────────────────────────────────────────────────────────────────────

def get_severity_distribution(session) -> List[Dict[str, Any]]:
    """
    Return the count of detection events for each severity level.

    Useful for a pie/donut chart.

    Args:
        session: Active SQLAlchemy session.

    Returns:
        list[dict]: Each dict has:
            Severity_Level (str), Detection_Count (int), Total_Potholes (int)
    """
    try:
        rows = (
            session.query(
                SeverityDim.Severity_Level,
                func.count(PotholeDetectionFact.Detection_ID).label("Detection_Count"),
                func.sum(PotholeDetectionFact.Pothole_Count).label("Total_Potholes"),
            )
            .join(SeverityDim, PotholeDetectionFact.Severity_ID == SeverityDim.Severity_ID)
            .group_by(SeverityDim.Severity_Level)
            .order_by(SeverityDim.Severity_Level)
            .all()
        )
        result = [_row_to_dict(r) for r in rows]
        logger.debug(f"get_severity_distribution: {result}")
        return result

    except Exception as exc:
        logger.error(f"get_severity_distribution error: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 6. Detections grouped by location
# ─────────────────────────────────────────────────────────────────────────────

def get_detections_by_location(session) -> List[Dict[str, Any]]:
    """
    Return detection statistics grouped by city and area.

    Includes average lat/lon for map plotting.

    Args:
        session: Active SQLAlchemy session.

    Returns:
        list[dict]: Each dict has:
            City (str), Area (str),
            Latitude (float|None), Longitude (float|None),
            Detection_Count (int), Total_Potholes (int),
            Avg_Confidence (float)
    """
    try:
        rows = (
            session.query(
                LocationDim.City,
                LocationDim.Area,
                LocationDim.Latitude,
                LocationDim.Longitude,
                func.count(PotholeDetectionFact.Detection_ID).label("Detection_Count"),
                func.sum(PotholeDetectionFact.Pothole_Count).label("Total_Potholes"),
                func.avg(PotholeDetectionFact.Confidence).label("Avg_Confidence"),
            )
            .join(LocationDim, PotholeDetectionFact.Location_ID == LocationDim.Location_ID)
            .group_by(
                LocationDim.City,
                LocationDim.Area,
                LocationDim.Latitude,
                LocationDim.Longitude,
            )
            .order_by(LocationDim.City, LocationDim.Area)
            .all()
        )

        result = []
        for r in rows:
            d = _row_to_dict(r)
            # Round avg_confidence for readability
            if d.get("Avg_Confidence") is not None:
                d["Avg_Confidence"] = round(float(d["Avg_Confidence"]), 4)
            result.append(d)

        logger.debug(f"get_detections_by_location: {len(result)} locations.")
        return result

    except Exception as exc:
        logger.error(f"get_detections_by_location error: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 7. Features for K-Means clustering
# ─────────────────────────────────────────────────────────────────────────────

def get_detections_for_kmeans(session) -> List[Dict[str, Any]]:
    """
    Return a feature-rich dataset suitable for K-Means clustering.

    Only returns rows that have valid GPS coordinates (lat/lon NOT NULL),
    since spatial clustering requires numeric coordinates.

    Features returned per row:
      - Latitude, Longitude     : geographic position
      - Pothole_Count           : volume feature
      - Confidence              : model certainty feature
      - Severity_Level          : nominal (must be encoded by caller)
      - Hour                    : time-of-day feature
      - Detection_ID, City, Area: for labelling cluster results

    Args:
        session: Active SQLAlchemy session.

    Returns:
        list[dict]: One dict per detection with the features listed above.
                    Empty list if query fails or no GPS data exists.
    """
    try:
        rows = (
            session.query(
                PotholeDetectionFact.Detection_ID,
                LocationDim.City,
                LocationDim.Area,
                LocationDim.Latitude,
                LocationDim.Longitude,
                PotholeDetectionFact.Pothole_Count,
                PotholeDetectionFact.Confidence,
                SeverityDim.Severity_Level,
                TimeDim.Hour,
            )
            .join(LocationDim, PotholeDetectionFact.Location_ID == LocationDim.Location_ID)
            .join(SeverityDim, PotholeDetectionFact.Severity_ID == SeverityDim.Severity_ID)
            .join(TimeDim,     PotholeDetectionFact.Time_ID     == TimeDim.Time_ID)
            # Only include rows with valid GPS data
            .filter(
                LocationDim.Latitude  != None,  # noqa: E711
                LocationDim.Longitude != None,  # noqa: E711
            )
            .all()
        )
        result = [_row_to_dict(r) for r in rows]
        logger.debug(f"get_detections_for_kmeans: {len(result)} rows with GPS data.")
        return result

    except Exception as exc:
        logger.error(f"get_detections_for_kmeans error: {exc}")
        return []
