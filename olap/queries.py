"""
olap/queries.py
===============
Raw SQL Query Strings and Helper Functions for OLAP Operations

This module centralises all the SQL queries used across the OLAP operations
(Roll-Up, Drill-Down, Slice, Dice).  Each query is stored as a module-level
constant so that individual operation modules can import and reuse them
without duplicating SQL.

OLAP Operations covered:
  - ROLLUP   : aggregate data from fine grain (Road) to coarse grain (City)
  - DRILLDOWN: navigate from coarse grain (City) down to individual records
  - SLICE    : filter on ONE dimension (e.g. Severity = 'High')
  - DICE     : filter on MULTIPLE dimensions simultaneously

Schema assumed:
  Pothole_Detection_Fact (fact table)
    -> Date_Dim, Time_Dim, Location_Dim, Road_Dim, Severity_Dim (dimension tables)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the project root is on the Python path so warehouse/utils are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from utils.helpers import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# ROLL-UP Queries
# ---------------------------------------------------------------------------

ROLLUP_ROAD_LEVEL = """
-- ROLL-UP Level 1 (most granular): aggregate potholes per Road.
-- Joins fact table with Road_Dim, Location_Dim, and Date_Dim to produce
-- human-readable road names, area, and city alongside pothole statistics.
SELECT
    r.Road_Name                 AS road_name,
    r.Road_Type                 AS road_type,
    l.Area                      AS area,
    l.City                      AS city,
    SUM(f.Pothole_Count)        AS pothole_count,
    COUNT(f.Detection_ID)       AS detection_count,
    AVG(f.Confidence)           AS avg_confidence
FROM Pothole_Detection_Fact f
JOIN Road_Dim     r ON f.Road_ID     = r.Road_ID
JOIN Location_Dim l ON f.Location_ID = l.Location_ID
JOIN Date_Dim     d ON f.Date_ID     = d.Date_ID
GROUP BY r.Road_ID, r.Road_Name, r.Road_Type, l.Area, l.City
ORDER BY pothole_count DESC
"""

ROLLUP_AREA_LEVEL = """
-- ROLL-UP Level 2 (intermediate): aggregate potholes per Area.
-- Rolls up the road-level data one step to the area level.
-- road_count shows how many distinct roads contribute to each area total.
SELECT
    l.Area                          AS area,
    l.City                          AS city,
    SUM(f.Pothole_Count)            AS pothole_count,
    COUNT(f.Detection_ID)           AS detection_count,
    COUNT(DISTINCT f.Road_ID)       AS road_count
FROM Pothole_Detection_Fact f
JOIN Location_Dim l ON f.Location_ID = l.Location_ID
GROUP BY l.Area, l.City
ORDER BY pothole_count DESC
"""

ROLLUP_CITY_LEVEL = """
-- ROLL-UP Level 3 (most aggregated): aggregate potholes per City.
-- Rolls up the area-level data one final step to the city level.
-- area_count and road_count show the breadth of coverage per city.
SELECT
    l.City                          AS city,
    SUM(f.Pothole_Count)            AS pothole_count,
    COUNT(f.Detection_ID)           AS detection_count,
    COUNT(DISTINCT l.Area)          AS area_count,
    COUNT(DISTINCT f.Road_ID)       AS road_count
FROM Pothole_Detection_Fact f
JOIN Location_Dim l ON f.Location_ID = l.Location_ID
GROUP BY l.City
ORDER BY pothole_count DESC
"""


# ---------------------------------------------------------------------------
# DRILL-DOWN Queries
# ---------------------------------------------------------------------------

DRILLDOWN_CITY = """
-- DRILL-DOWN starting point: list all cities with aggregate counts.
-- Use this to choose which city to drill into (next: DRILLDOWN_AREA).
SELECT
    l.City                          AS city,
    SUM(f.Pothole_Count)            AS pothole_count,
    COUNT(f.Detection_ID)           AS detection_count,
    COUNT(DISTINCT l.Area)          AS area_count,
    COUNT(DISTINCT f.Road_ID)       AS road_count
FROM Pothole_Detection_Fact f
JOIN Location_Dim l ON f.Location_ID = l.Location_ID
GROUP BY l.City
ORDER BY pothole_count DESC
"""

DRILLDOWN_AREA = """
-- DRILL-DOWN Level 2: list all Areas within a given City.
-- :city parameter must be supplied.  Returns area-level aggregates.
SELECT
    l.Area                          AS area,
    l.City                          AS city,
    SUM(f.Pothole_Count)            AS pothole_count,
    COUNT(f.Detection_ID)           AS detection_count,
    COUNT(DISTINCT f.Road_ID)       AS road_count
FROM Pothole_Detection_Fact f
JOIN Location_Dim l ON f.Location_ID = l.Location_ID
WHERE l.City = :city
GROUP BY l.Area, l.City
ORDER BY pothole_count DESC
"""

DRILLDOWN_ROAD = """
-- DRILL-DOWN Level 3: list all Roads within a given City + Area.
-- :city and :area parameters must be supplied.
SELECT
    r.Road_Name                     AS road_name,
    r.Road_Type                     AS road_type,
    l.Area                          AS area,
    l.City                          AS city,
    SUM(f.Pothole_Count)            AS pothole_count,
    COUNT(f.Detection_ID)           AS detection_count,
    AVG(f.Confidence)               AS avg_confidence
FROM Pothole_Detection_Fact f
JOIN Road_Dim     r ON f.Road_ID     = r.Road_ID
JOIN Location_Dim l ON f.Location_ID = l.Location_ID
WHERE l.City = :city
  AND l.Area = :area
GROUP BY r.Road_ID, r.Road_Name, r.Road_Type, l.Area, l.City
ORDER BY pothole_count DESC
"""

DRILLDOWN_RECORDS = """
-- DRILL-DOWN Level 4 (most granular): individual detection records on a Road.
-- :road_name parameter must be supplied (supports LIKE wildcards via %road_name%).
-- Returns every row from the fact table for the matched road.
SELECT
    f.Detection_ID,
    d.Date                          AS date,
    t.Hour || ':' || printf('%02d', t.Minute) AS time,
    l.City                          AS city,
    l.Area                          AS area,
    r.Road_Name                     AS road_name,
    r.Road_Type                     AS road_type,
    s.Severity_Level                AS severity,
    f.Pothole_Count,
    f.Confidence,
    f.Source_Type,
    f.Is_Demo
FROM Pothole_Detection_Fact f
JOIN Date_Dim     d ON f.Date_ID     = d.Date_ID
JOIN Time_Dim     t ON f.Time_ID     = t.Time_ID
JOIN Location_Dim l ON f.Location_ID = l.Location_ID
JOIN Road_Dim     r ON f.Road_ID     = r.Road_ID
JOIN Severity_Dim s ON f.Severity_ID = s.Severity_ID
WHERE r.Road_Name LIKE :road_name
ORDER BY d.Date DESC, t.Hour DESC
"""


# ---------------------------------------------------------------------------
# SLICE Query
# ---------------------------------------------------------------------------

SLICE_HIGH_SEVERITY = """
-- SLICE: Fix the Severity dimension to 'High'.
-- Returns all detection records where severity is High, including full
-- dimensional context (date, time, location, road) for analysis.
SELECT
    f.Detection_ID,
    d.Date                          AS date,
    t.Hour || ':' || printf('%02d', t.Minute) AS time,
    l.City                          AS city,
    l.Area                          AS area,
    r.Road_Name                     AS road_name,
    r.Road_Type                     AS road_type,
    s.Severity_Level                AS severity,
    f.Pothole_Count,
    f.Confidence,
    f.Source_Type,
    f.Is_Demo
FROM Pothole_Detection_Fact f
JOIN Date_Dim     d ON f.Date_ID     = d.Date_ID
JOIN Time_Dim     t ON f.Time_ID     = t.Time_ID
JOIN Location_Dim l ON f.Location_ID = l.Location_ID
JOIN Road_Dim     r ON f.Road_ID     = r.Road_ID
JOIN Severity_Dim s ON f.Severity_ID = s.Severity_ID
WHERE s.Severity_Level = 'High'
ORDER BY d.Date DESC, f.Pothole_Count DESC
"""


# ---------------------------------------------------------------------------
# DICE Query (parameterised)
# ---------------------------------------------------------------------------

DICE_FILTER = """
-- DICE: Multi-dimensional filter.
-- This query is a TEMPLATE — the {where_clause} placeholder is filled at
-- runtime by olap/dice.py based on which filters the caller supplies.
-- Supported filter dimensions: Severity, City, Year, Road_Name.
SELECT
    f.Detection_ID,
    d.Date                          AS date,
    d.Year                          AS year,
    d.Month                         AS month,
    l.City                          AS city,
    l.Area                          AS area,
    r.Road_Name                     AS road_name,
    r.Road_Type                     AS road_type,
    s.Severity_Level                AS severity,
    f.Pothole_Count,
    f.Confidence,
    f.Source_Type,
    f.Is_Demo
FROM Pothole_Detection_Fact f
JOIN Date_Dim     d ON f.Date_ID     = d.Date_ID
JOIN Time_Dim     t ON f.Time_ID     = t.Time_ID
JOIN Location_Dim l ON f.Location_ID = l.Location_ID
JOIN Road_Dim     r ON f.Road_ID     = r.Road_ID
JOIN Severity_Dim s ON f.Severity_ID = s.Severity_ID
{where_clause}
ORDER BY d.Date DESC, f.Pothole_Count DESC
"""


# ---------------------------------------------------------------------------
# Helper Function
# ---------------------------------------------------------------------------

def execute_query(
    session,
    query_str: str,
    params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Execute a raw SQL query string and return the results as a list of dicts.

    Each dictionary in the returned list represents one row, with column names
    as keys and cell values as values.  This makes results easy to pass to
    pandas, JSON serialisers, or template engines.

    Args:
        session:    An active SQLAlchemy database session (obtained via
                    ``get_db_session()`` from warehouse/database.py).
        query_str:  The SQL query string to execute.  May contain named
                    bind parameters in the form ``:param_name``.
        params:     Optional dictionary of bind-parameter values, e.g.
                    ``{"severity": "High", "city": "Mumbai"}``.

    Returns:
        A list of dicts, one per row.  Empty list if no rows were returned.

    Example:
        >>> with get_db_session() as sess:
        ...     rows = execute_query(sess, ROLLUP_CITY_LEVEL)
        ...     for row in rows:
        ...         print(row["city"], row["pothole_count"])
    """
    try:
        # Wrap the raw SQL string in SQLAlchemy's text() for safe execution
        stmt = text(query_str)

        # Execute with or without bind parameters
        result = session.execute(stmt, params or {})

        # Fetch column names from the result cursor
        column_names = list(result.keys())

        # Build a list of dicts: one dict per row
        rows_as_dicts = [
            dict(zip(column_names, row))
            for row in result.fetchall()
        ]

        logger.debug(
            "execute_query returned %d rows | params=%s",
            len(rows_as_dicts),
            params,
        )
        return rows_as_dicts

    except Exception as exc:
        logger.error("execute_query failed: %s", exc)
        raise
