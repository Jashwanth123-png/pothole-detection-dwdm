"""
olap/rollup.py
==============
OLAP ROLL-UP Operation

Roll-Up aggregates data at increasing levels of granularity:
  Road Level -> Area Level -> City Level

This demonstrates the OLAP Roll-Up concept:
  - At Road level: total potholes per road
  - At Area level: sum of all potholes in each area
  - At City level: sum of all potholes in each city

This is the opposite of Drill-Down (which goes from general to specific).
"""

import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from warehouse.database import get_db_session
from utils.helpers import setup_logger

logger = setup_logger(__name__)


def rollup_by_road(session=None) -> pd.DataFrame:
    """
    ROLL-UP Level 1: Pothole counts per Road.
    Most granular level of the Road -> Area -> City hierarchy.
    
    Returns:
        DataFrame with columns: road_name, road_type, area, city, pothole_count, detection_count
    """
    query = """
        SELECT 
            r.Road_Name    AS road_name,
            r.Road_Type    AS road_type,
            l.Area         AS area,
            l.City         AS city,
            SUM(f.Pothole_Count)  AS pothole_count,
            COUNT(f.Detection_ID) AS detection_count,
            AVG(f.Confidence)     AS avg_confidence
        FROM Pothole_Detection_Fact f
        JOIN Road_Dim     r ON f.Road_ID     = r.Road_ID
        JOIN Location_Dim l ON f.Location_ID = l.Location_ID
        GROUP BY r.Road_ID, r.Road_Name, r.Road_Type, l.Area, l.City
        ORDER BY pothole_count DESC
    """
    from warehouse.database import get_db_session
    from sqlalchemy import text
    
    def _execute(sess):
        result = sess.execute(text(query))
        rows = result.fetchall()
        cols = result.keys()
        return pd.DataFrame(rows, columns=list(cols))
    
    if session:
        return _execute(session)
    else:
        with get_db_session() as sess:
            return _execute(sess)


def rollup_by_area(session=None) -> pd.DataFrame:
    """
    ROLL-UP Level 2: Pothole counts aggregated by Area.
    Intermediate level in the Road -> Area -> City hierarchy.
    
    Returns:
        DataFrame with columns: area, city, pothole_count, detection_count
    """
    query = """
        SELECT 
            l.Area         AS area,
            l.City         AS city,
            SUM(f.Pothole_Count)  AS pothole_count,
            COUNT(f.Detection_ID) AS detection_count,
            COUNT(DISTINCT f.Road_ID) AS road_count
        FROM Pothole_Detection_Fact f
        JOIN Location_Dim l ON f.Location_ID = l.Location_ID
        GROUP BY l.Area, l.City
        ORDER BY pothole_count DESC
    """
    from sqlalchemy import text
    
    def _execute(sess):
        result = sess.execute(text(query))
        rows = result.fetchall()
        cols = result.keys()
        return pd.DataFrame(rows, columns=list(cols))
    
    if session:
        return _execute(session)
    else:
        with get_db_session() as sess:
            return _execute(sess)


def rollup_by_city(session=None) -> pd.DataFrame:
    """
    ROLL-UP Level 3: Pothole counts aggregated by City.
    Most aggregated level in the Road -> Area -> City hierarchy.
    
    Returns:
        DataFrame with columns: city, pothole_count, detection_count, area_count, road_count
    """
    query = """
        SELECT 
            l.City         AS city,
            SUM(f.Pothole_Count)        AS pothole_count,
            COUNT(f.Detection_ID)       AS detection_count,
            COUNT(DISTINCT l.Area)      AS area_count,
            COUNT(DISTINCT f.Road_ID)   AS road_count
        FROM Pothole_Detection_Fact f
        JOIN Location_Dim l ON f.Location_ID = l.Location_ID
        GROUP BY l.City
        ORDER BY pothole_count DESC
    """
    from sqlalchemy import text
    
    def _execute(sess):
        result = sess.execute(text(query))
        rows = result.fetchall()
        cols = result.keys()
        return pd.DataFrame(rows, columns=list(cols))
    
    if session:
        return _execute(session)
    else:
        with get_db_session() as sess:
            return _execute(sess)


def get_all_rollup_levels() -> dict:
    """
    Get all three roll-up levels at once.
    
    Returns:
        dict with keys 'road', 'area', 'city' each containing a DataFrame
    """
    with get_db_session() as sess:
        return {
            "road": rollup_by_road(sess),
            "area": rollup_by_area(sess),
            "city": rollup_by_city(sess)
        }


if __name__ == "__main__":
    print("\n=== OLAP ROLL-UP DEMONSTRATION ===")
    print("\nRoll-Up: Road -> Area -> City")
    
    data = get_all_rollup_levels()
    
    print("\n--- Road Level (Most Detailed) ---")
    print(data["road"].to_string(index=False))
    
    print("\n--- Area Level (Intermediate) ---")
    print(data["area"].to_string(index=False))
    
    print("\n--- City Level (Most Aggregated) ---")
    print(data["city"].to_string(index=False))
