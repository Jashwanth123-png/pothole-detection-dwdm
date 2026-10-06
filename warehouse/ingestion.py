"""
warehouse/ingestion.py
======================
ETL ingestion pipeline for the Pothole Detection Data Warehouse.

Transforms raw detection records (dicts) produced by the detection pipeline
into normalised rows across the five dimension tables and the central fact
table.

Typical usage::

    from warehouse.ingestion import ingest_detection_safe

    record = {
        "timestamp"    : datetime(2024, 6, 15, 14, 32, 0),
        "city"         : "Mumbai",
        "area"         : "Andheri West",
        "latitude"     : 19.1362,
        "longitude"    : 72.8296,
        "road_name"    : "Link Road",
        "road_type"    : "Urban",
        "severity"     : "High",
        "pothole_count": 3,
        "confidence"   : 0.87,
        "bbox"         : {"x1": 120, "y1": 80, "x2": 340, "y2": 260},
        "source_type"  : "image",
        "is_demo"      : 0,
    }
    success = ingest_detection_safe(record)

Detection_ID format: DET-YYYYMMDD-HHMMSS-XXXXXX  (XXXXXX = 6 random hex chars)
"""

import uuid
import logging
from datetime import datetime
from typing import Optional

from warehouse.models import (
    DateDim,
    TimeDim,
    LocationDim,
    RoadDim,
    SeverityDim,
    PotholeDetectionFact,
)
from warehouse.database import get_db_session
from utils.helpers import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: classify hour → time period
# ─────────────────────────────────────────────────────────────────────────────

def _get_time_period(hour: int) -> str:
    """
    Map an hour (0-23) to a human-readable time period.

    Periods:
      - Morning   : 06:00 – 11:59
      - Afternoon : 12:00 – 16:59
      - Evening   : 17:00 – 20:59
      - Night     : 21:00 – 05:59

    Args:
        hour (int): Hour of day in 24-hour format (0–23).

    Returns:
        str: One of 'Morning', 'Afternoon', 'Evening', 'Night'.
    """
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: generate Detection_ID
# ─────────────────────────────────────────────────────────────────────────────

def _generate_detection_id(dt: datetime) -> str:
    """
    Generate a unique Detection_ID string.

    Format: ``DET-YYYYMMDD-HHMMSS-XXXXXX``
    where XXXXXX is 6 uppercase hex characters from a UUID.

    Args:
        dt (datetime): Timestamp of the detection event.

    Returns:
        str: Unique detection identifier.
    """
    date_part = dt.strftime("%Y%m%d")
    time_part = dt.strftime("%H%M%S")
    rand_part = uuid.uuid4().hex[:6].upper()
    return f"DET-{date_part}-{time_part}-{rand_part}"


# ─────────────────────────────────────────────────────────────────────────────
# Dimension look-up / create helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_date_dim(session, dt: datetime) -> DateDim:
    """
    Return the DateDim row for the given datetime, creating it if absent.

    Uses the ISO date string (YYYY-MM-DD) as the unique key so the same
    calendar date is never inserted twice.

    Args:
        session: Active SQLAlchemy session.
        dt      : Python datetime object.

    Returns:
        DateDim: ORM instance for the matching date row.
    """
    date_str = dt.strftime("%Y-%m-%d")

    # Try to find an existing row
    row = session.query(DateDim).filter_by(Date=date_str).first()
    if row:
        return row

    # Build a new row from the datetime components
    quarter = (dt.month - 1) // 3 + 1  # Q1-Q4

    row = DateDim(
        Date       = date_str,
        Day        = dt.day,
        Month      = dt.month,
        Year       = dt.year,
        Quarter    = quarter,
        Day_Name   = dt.strftime("%A"),      # e.g. 'Monday'
        Month_Name = dt.strftime("%B"),      # e.g. 'January'
    )
    session.add(row)
    session.flush()   # populate auto-increment ID without full commit
    logger.debug(f"Created DateDim: {date_str}")
    return row


def get_or_create_time_dim(session, dt: datetime) -> TimeDim:
    """
    Return the TimeDim row for the given datetime, creating it if absent.

    The unique key is (Hour, Minute) — sub-minute precision is discarded.

    Args:
        session: Active SQLAlchemy session.
        dt      : Python datetime object.

    Returns:
        TimeDim: ORM instance for the matching time row.
    """
    hour   = dt.hour
    minute = dt.minute
    period = _get_time_period(hour)

    row = session.query(TimeDim).filter_by(Hour=hour, Minute=minute).first()
    if row:
        return row

    row = TimeDim(Hour=hour, Minute=minute, Time_Period=period)
    session.add(row)
    session.flush()
    logger.debug(f"Created TimeDim: {hour:02d}:{minute:02d} ({period})")
    return row


def get_or_create_location_dim(
    session,
    city    : str,
    area    : str,
    lat     : Optional[float],
    lon     : Optional[float],
    is_demo : int = 0,
) -> LocationDim:
    """
    Return the LocationDim row for the given city/area pair, creating if absent.

    The unique key is (City, Area). If the row already exists, lat/lon and
    is_demo are NOT updated to preserve the first-seen values.

    Args:
        session : Active SQLAlchemy session.
        city    : City name (e.g. 'Mumbai').
        area    : Area/neighbourhood name (e.g. 'Andheri West').
        lat     : Latitude in decimal degrees, or None.
        lon     : Longitude in decimal degrees, or None.
        is_demo : 1 if this is synthetic/demo data, else 0.

    Returns:
        LocationDim: ORM instance for the matching location row.
    """
    row = session.query(LocationDim).filter_by(City=city, Area=area).first()
    if row:
        return row

    row = LocationDim(
        City      = city,
        Area      = area,
        Latitude  = lat,
        Longitude = lon,
        Is_Demo   = is_demo,
    )
    session.add(row)
    session.flush()
    logger.debug(f"Created LocationDim: {city}/{area}")
    return row


def get_or_create_road_dim(
    session,
    road_name : str,
    road_type : str,
) -> RoadDim:
    """
    Return the RoadDim row for the given road, creating it if absent.

    The unique key is (Road_Name, Road_Type).

    Args:
        session   : Active SQLAlchemy session.
        road_name : Name of the road (e.g. 'NH-48').
        road_type : One of 'Highway', 'Urban', 'Rural', 'Unknown'.

    Returns:
        RoadDim: ORM instance for the matching road row.
    """
    row = session.query(RoadDim).filter_by(
        Road_Name=road_name, Road_Type=road_type
    ).first()
    if row:
        return row

    row = RoadDim(Road_Name=road_name, Road_Type=road_type)
    session.add(row)
    session.flush()
    logger.debug(f"Created RoadDim: {road_name} ({road_type})")
    return row


def get_severity_id(session, severity_level: str) -> int:
    """
    Look up the Severity_ID for a given severity level string.

    The Severity_Dim is seeded at startup with 'Low', 'Medium', 'High'.
    If an unrecognised level is given, it defaults to 'Low'.

    Args:
        session        : Active SQLAlchemy session.
        severity_level : One of 'Low', 'Medium', 'High'.

    Returns:
        int: The Severity_ID primary key.

    Raises:
        RuntimeError: If Severity_Dim is empty (database not initialised).
    """
    # Normalise to title-case for safety
    level = severity_level.strip().title()
    if level not in ("Low", "Medium", "High"):
        logger.warning(
            f"Unrecognised severity '{severity_level}', defaulting to 'Low'."
        )
        level = "Low"

    row = session.query(SeverityDim).filter_by(Severity_Level=level).first()
    if row is None:
        raise RuntimeError(
            "Severity_Dim is empty — call initialize_database() first."
        )
    return row.Severity_ID


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion function
# ─────────────────────────────────────────────────────────────────────────────

def ingest_detection(session, detection_record: dict) -> str:
    """
    Ingest one detection record into the data warehouse.

    Performs a full ETL on the raw record:
      1. Resolves or creates all dimension rows (date, time, location, road,
         severity).
      2. Inserts a new row into Pothole_Detection_Fact.
      3. Returns the Detection_ID of the inserted row.

    Duplicate Prevention
    --------------------
    If the record already contains a ``detection_id`` key, the function first
    checks whether that ID exists in the fact table. If it does, it skips the
    insert and returns the existing ID.

    Expected ``detection_record`` keys
    -----------------------------------
    detection_id  (str,  optional)  – Pre-assigned ID; auto-generated if absent.
    timestamp     (datetime)        – When the detection occurred.
    city          (str)             – City name.
    area          (str)             – Area/neighbourhood.
    latitude      (float|None)      – GPS latitude.
    longitude     (float|None)      – GPS longitude.
    road_name     (str)             – Road name.
    road_type     (str)             – 'Highway' | 'Urban' | 'Rural' | 'Unknown'.
    severity      (str)             – 'Low' | 'Medium' | 'High'.
    pothole_count (int)             – Number of potholes detected.
    confidence    (float)           – Average model confidence (0–1).
    bbox          (dict|None)       – {'x1':…, 'y1':…, 'x2':…, 'y2':…} or None.
    source_type   (str)             – 'image' | 'video' | 'webcam'.
    is_demo       (int)             – 1 = demo/synthetic data, 0 = real.

    Args:
        session          : Active SQLAlchemy session (within a transaction).
        detection_record : Raw detection dict as described above.

    Returns:
        str: The Detection_ID that was inserted (or already existed).

    Raises:
        KeyError     : If a required key is missing from detection_record.
        RuntimeError : If Severity_Dim is not seeded.
    """
    # ── 1. Extract and validate fields ──────────────────────────────────────
    ts            = detection_record["timestamp"]        # datetime
    city          = detection_record.get("city",         "Unknown")
    area          = detection_record.get("area",         "Unknown")
    lat           = detection_record.get("latitude")
    lon           = detection_record.get("longitude")
    road_name     = detection_record.get("road_name",    "Unknown Road")
    road_type     = detection_record.get("road_type",    "Unknown")
    severity      = detection_record.get("severity",     "Low")
    pothole_count = int(detection_record.get("pothole_count", 0))
    confidence    = float(detection_record.get("confidence",  0.0))
    bbox          = detection_record.get("bbox")          # dict or None
    source_type   = detection_record.get("source_type",  "image")
    is_demo       = int(detection_record.get("is_demo",  0))

    # ── 2. Determine Detection_ID ────────────────────────────────────────────
    det_id = detection_record.get("detection_id") or _generate_detection_id(ts)

    # ── 3. Duplicate check ───────────────────────────────────────────────────
    existing = session.query(PotholeDetectionFact).filter_by(
        Detection_ID=det_id
    ).first()
    if existing:
        logger.debug(f"Duplicate detection skipped: {det_id}")
        return det_id

    # ── 4. Resolve / create dimension rows ───────────────────────────────────
    date_row     = get_or_create_date_dim(session, ts)
    time_row     = get_or_create_time_dim(session, ts)
    location_row = get_or_create_location_dim(session, city, area, lat, lon, is_demo)
    road_row     = get_or_create_road_dim(session, road_name, road_type)
    severity_id  = get_severity_id(session, severity)

    # ── 5. Unpack bounding box ───────────────────────────────────────────────
    bbox_x1 = bbox_y1 = bbox_x2 = bbox_y2 = None
    if isinstance(bbox, dict):
        bbox_x1 = bbox.get("x1")
        bbox_y1 = bbox.get("y1")
        bbox_x2 = bbox.get("x2")
        bbox_y2 = bbox.get("y2")

    # ── 6. Build and add fact row ────────────────────────────────────────────
    fact = PotholeDetectionFact(
        Detection_ID  = det_id,
        Date_ID       = date_row.Date_ID,
        Time_ID       = time_row.Time_ID,
        Location_ID   = location_row.Location_ID,
        Road_ID       = road_row.Road_ID,
        Severity_ID   = severity_id,
        Pothole_Count = pothole_count,
        Confidence    = confidence,
        BBox_X1       = bbox_x1,
        BBox_Y1       = bbox_y1,
        BBox_X2       = bbox_x2,
        BBox_Y2       = bbox_y2,
        Source_Type   = source_type,
        Is_Demo       = is_demo,
        Created_At    = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    session.add(fact)
    logger.debug(f"Ingested detection: {det_id} | {city}/{area} | {severity} | count={pothole_count}")
    return det_id


# ─────────────────────────────────────────────────────────────────────────────
# Safe wrapper (used by UI / batch jobs)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_detection_safe(detection_record: dict) -> bool:
    """
    Wrapper around :func:`ingest_detection` that handles its own session and
    silently catches all exceptions.

    Suitable for use in the Streamlit UI or any caller that should not crash
    on a database error.

    Args:
        detection_record (dict): Raw detection record dict
                                 (see :func:`ingest_detection` for schema).

    Returns:
        bool: True if ingestion succeeded, False if any error occurred.
    """
    try:
        with get_db_session() as session:
            det_id = ingest_detection(session, detection_record)
        logger.debug(f"ingest_detection_safe: OK → {det_id}")
        return True
    except Exception as exc:
        logger.error(f"ingest_detection_safe: FAILED — {exc}")
        return False
