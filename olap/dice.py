"""
olap/dice.py
============
OLAP DICE Operation

Dice applies filters on MULTIPLE dimensions simultaneously.
This is different from Slice (which filters on one dimension).

Example:
  Dice where:
    Severity = 'High' AND
    City     = 'Mumbai' AND
    Year     = 2024
  
  Returns only the subset of data matching ALL conditions.
"""

import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from warehouse.database import get_db_session
from sqlalchemy import text
from utils.helpers import setup_logger

logger = setup_logger(__name__)


def dice(
    severity: str = None,
    city: str = None,
    year: int = None,
    road_name: str = None,
    session=None
) -> pd.DataFrame:
    """
    DICE: Multi-dimensional filter.
    
    Any combination of filters can be applied simultaneously.
    Only provided filters are applied (None = no filter on that dimension).
    
    Args:
        severity:  'Low', 'Medium', or 'High' (or None)
        city:      City name filter (or None)
        year:      Year filter (or None)
        road_name: Road name filter (or None)
        session:   Optional database session
    
    Returns:
        Filtered DataFrame
    """
    # Build WHERE clauses dynamically
    conditions = []
    params = {}
    
    if severity:
        conditions.append("s.Severity_Level = :severity")
        params["severity"] = severity
    
    if city:
        conditions.append("l.City = :city")
        params["city"] = city
    
    if year:
        conditions.append("d.Year = :year")
        params["year"] = year
    
    if road_name:
        conditions.append("r.Road_Name LIKE :road_name")
        params["road_name"] = f"%{road_name}%"
    
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    
    query = f"""
        SELECT 
            f.Detection_ID,
            d.Date          AS date,
            d.Year          AS year,
            d.Month         AS month,
            l.City          AS city,
            l.Area          AS area,
            r.Road_Name     AS road_name,
            r.Road_Type     AS road_type,
            s.Severity_Level AS severity,
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
    
    logger.info(f"DICE query - severity={severity}, city={city}, year={year}, road={road_name}")
    
    def _execute(sess):
        result = sess.execute(text(query), params)
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=list(result.keys()))
    
    if session:
        return _execute(session)
    else:
        with get_db_session() as sess:
            return _execute(sess)


def dice_summary(severity=None, city=None, year=None, road_name=None) -> dict:
    """
    Get a summary of dice results.
    
    Returns:
        dict with total_detections, total_potholes, avg_confidence
    """
    df = dice(severity=severity, city=city, year=year, road_name=road_name)
    
    if df.empty:
        return {
            "total_detections": 0,
            "total_potholes": 0,
            "avg_confidence": 0.0,
            "filters_applied": {
                "severity": severity,
                "city": city,
                "year": year,
                "road": road_name
            }
        }
    
    return {
        "total_detections": len(df),
        "total_potholes": int(df["Pothole_Count"].sum()),
        "avg_confidence": float(df["Confidence"].mean()),
        "filters_applied": {
            "severity": severity,
            "city": city,
            "year": year,
            "road": road_name
        }
    }


if __name__ == "__main__":
    print("\n=== OLAP DICE DEMONSTRATION ===")
    
    # Dice: High severity + specific city
    print("\nDice: Severity=High, City=Mumbai")
    df = dice(severity="High", city="Mumbai")
    print(f"Found {len(df)} matching records")
    if not df.empty:
        print(df.head(10).to_string(index=False))
    
    # Dice summary
    summary = dice_summary(severity="High")
    print(f"\nHigh severity summary: {summary}")
