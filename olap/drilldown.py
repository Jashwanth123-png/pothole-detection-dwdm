"""
olap/drilldown.py
=================
OLAP DRILL-DOWN Operation

Drill-Down navigates from a high-level summary DOWN to progressively
more detailed data.  This is the OPPOSITE of Roll-Up.

Hierarchy used:
  City  ->  Area  ->  Road  ->  Individual Detection Records

Example flow:
  1. drilldown_cities()              - see all cities + totals
  2. drilldown_to_areas("Mumbai")    - see areas inside Mumbai
  3. drilldown_to_roads("Mumbai", "Andheri") - see roads in that area
  4. drilldown_to_records("SV Road") - see every detection on that road

Each step adds MORE detail than the previous one.
"""

import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from warehouse.database import get_db_session
from sqlalchemy import text
from utils.helpers import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Level 0 – Cities  (starting point for drill-down)
# ---------------------------------------------------------------------------

def drilldown_cities(session=None) -> pd.DataFrame:
    """
    DRILL-DOWN Entry Point: List all Cities with aggregate pothole counts.

    Use the results of this function to decide which city to drill into
    next by calling ``drilldown_to_areas(city)``.

    Returns:
        DataFrame with columns:
            city, pothole_count, detection_count, area_count, road_count
    """
    query = """
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

    def _execute(sess):
        result = sess.execute(text(query))
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=list(result.keys()))

    if session:
        return _execute(session)
    else:
        with get_db_session() as sess:
            return _execute(sess)


# ---------------------------------------------------------------------------
# Level 1 – Areas inside a City
# ---------------------------------------------------------------------------

def drilldown_to_areas(city: str, session=None) -> pd.DataFrame:
    """
    DRILL-DOWN Level 1: List all Areas within a given City.

    Args:
        city:    The city name to drill into (e.g. 'Mumbai').
        session: Optional active database session.

    Returns:
        DataFrame with columns:
            area, city, pothole_count, detection_count, road_count
    """
    query = """
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

    def _execute(sess):
        result = sess.execute(text(query), {"city": city})
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=list(result.keys()))

    logger.info("Drill-down to areas: city=%s", city)

    if session:
        return _execute(session)
    else:
        with get_db_session() as sess:
            return _execute(sess)


# ---------------------------------------------------------------------------
# Level 2 – Roads inside an Area (within a City)
# ---------------------------------------------------------------------------

def drilldown_to_roads(city: str, area: str, session=None) -> pd.DataFrame:
    """
    DRILL-DOWN Level 2: List all Roads within a given City and Area.

    Args:
        city:    The city name (e.g. 'Mumbai').
        area:    The area name inside that city (e.g. 'Andheri').
        session: Optional active database session.

    Returns:
        DataFrame with columns:
            road_name, road_type, area, city,
            pothole_count, detection_count, avg_confidence
    """
    query = """
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

    def _execute(sess):
        result = sess.execute(text(query), {"city": city, "area": area})
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=list(result.keys()))

    logger.info("Drill-down to roads: city=%s, area=%s", city, area)

    if session:
        return _execute(session)
    else:
        with get_db_session() as sess:
            return _execute(sess)


# ---------------------------------------------------------------------------
# Level 3 – Individual Detection Records on a Road
# ---------------------------------------------------------------------------

def drilldown_to_records(road_name: str, session=None) -> pd.DataFrame:
    """
    DRILL-DOWN Level 3 (most granular): Individual detection records on a Road.

    Returns every row from the fact table that matches the given road name.
    Supports partial matching (LIKE search), so passing 'SV' will match
    'SV Road', 'SV Highway', etc.

    Args:
        road_name: Road name (or partial name) to look up.
        session:   Optional active database session.

    Returns:
        DataFrame with columns:
            Detection_ID, date, time, city, area,
            road_name, road_type, severity,
            Pothole_Count, Confidence, Source_Type, Is_Demo
    """
    query = """
        SELECT
            f.Detection_ID,
            d.Date                                      AS date,
            t.Hour || ':' || printf('%02d', t.Minute)   AS time,
            l.City                                      AS city,
            l.Area                                      AS area,
            r.Road_Name                                 AS road_name,
            r.Road_Type                                 AS road_type,
            s.Severity_Level                            AS severity,
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

    def _execute(sess):
        result = sess.execute(text(query), {"road_name": f"%{road_name}%"})
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=list(result.keys()))

    logger.info("Drill-down to records: road_name LIKE '%%%s%%'", road_name)

    if session:
        return _execute(session)
    else:
        with get_db_session() as sess:
            return _execute(sess)


# ---------------------------------------------------------------------------
# __main__ – Interactive demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== OLAP DRILL-DOWN DEMONSTRATION ===")
    print("Navigation: City  ->  Area  ->  Road  ->  Individual Records\n")

    # ── Step 0: All cities ────────────────────────────────────────────────
    print("Step 0: All Cities (entry point)")
    cities_df = drilldown_cities()
    print(cities_df.to_string(index=False))

    if cities_df.empty:
        print("\n[No data in warehouse — run etl/load.py first]")
    else:
        # Pick the city with the most potholes to demonstrate drill-down
        top_city = cities_df.iloc[0]["city"]

        # ── Step 1: Areas in top city ─────────────────────────────────────
        print(f"\nStep 1: Areas inside '{top_city}'")
        areas_df = drilldown_to_areas(top_city)
        print(areas_df.to_string(index=False))

        if not areas_df.empty:
            top_area = areas_df.iloc[0]["area"]

            # ── Step 2: Roads in top area ─────────────────────────────────
            print(f"\nStep 2: Roads inside '{top_area}', {top_city}")
            roads_df = drilldown_to_roads(top_city, top_area)
            print(roads_df.to_string(index=False))

            if not roads_df.empty:
                top_road = roads_df.iloc[0]["road_name"]

                # ── Step 3: Individual records on top road ─────────────────
                print(f"\nStep 3: Individual detections on '{top_road}'")
                records_df = drilldown_to_records(top_road)
                print(f"Total records: {len(records_df)}")
                print(records_df.head(10).to_string(index=False))
