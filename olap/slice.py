"""
olap/slice.py
=============
OLAP SLICE Operation

Slice fixes one dimension to a single value and returns
the resulting subset of data.

Example: Slice where Severity = 'High'
  - Returns all detections where severity is High
  - This is a single-dimension filter

Other examples:
  - Slice by Year = 2024
  - Slice by City = 'Mumbai'
"""

import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from warehouse.database import get_db_session
from sqlalchemy import text
from utils.helpers import setup_logger

logger = setup_logger(__name__)


def slice_by_severity(severity_level: str, session=None) -> pd.DataFrame:
    """
    SLICE: Filter detections by a single severity level.
    
    Example: slice_by_severity('High') returns only High severity detections.
    
    Args:
        severity_level: 'Low', 'Medium', or 'High'
        session: Optional database session
    
    Returns:
        DataFrame of filtered detections
    """
    query = """
        SELECT 
            f.Detection_ID,
            d.Date          AS date,
            t.Hour || ':' || printf('%02d', t.Minute) AS time,
            l.City          AS city,
            l.Area          AS area,
            r.Road_Name     AS road_name,
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
        WHERE s.Severity_Level = :severity
        ORDER BY d.Date DESC, t.Hour DESC
    """
    
    def _execute(sess):
        result = sess.execute(text(query), {"severity": severity_level})
        rows = result.fetchall()
        cols = result.keys()
        return pd.DataFrame(rows, columns=list(cols))
    
    if session:
        return _execute(session)
    else:
        with get_db_session() as sess:
            return _execute(sess)


def slice_by_year(year: int, session=None) -> pd.DataFrame:
    """SLICE: Filter by year."""
    query = """
        SELECT 
            f.Detection_ID,
            d.Date, d.Month, d.Year,
            l.City, l.Area,
            r.Road_Name,
            s.Severity_Level AS severity,
            f.Pothole_Count,
            f.Confidence
        FROM Pothole_Detection_Fact f
        JOIN Date_Dim     d ON f.Date_ID     = d.Date_ID
        JOIN Location_Dim l ON f.Location_ID = l.Location_ID
        JOIN Road_Dim     r ON f.Road_ID     = r.Road_ID
        JOIN Severity_Dim s ON f.Severity_ID = s.Severity_ID
        WHERE d.Year = :year
        ORDER BY d.Date DESC
    """
    
    def _execute(sess):
        result = sess.execute(text(query), {"year": year})
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=list(result.keys()))
    
    if session:
        return _execute(session)
    else:
        with get_db_session() as sess:
            return _execute(sess)


def slice_by_city(city: str, session=None) -> pd.DataFrame:
    """SLICE: Filter by city."""
    query = """
        SELECT 
            f.Detection_ID,
            d.Date,
            l.City, l.Area,
            r.Road_Name,
            s.Severity_Level AS severity,
            f.Pothole_Count,
            f.Confidence
        FROM Pothole_Detection_Fact f
        JOIN Date_Dim     d ON f.Date_ID     = d.Date_ID
        JOIN Location_Dim l ON f.Location_ID = l.Location_ID
        JOIN Road_Dim     r ON f.Road_ID     = r.Road_ID
        JOIN Severity_Dim s ON f.Severity_ID = s.Severity_ID
        WHERE l.City = :city
        ORDER BY d.Date DESC
    """
    
    def _execute(sess):
        result = sess.execute(text(query), {"city": city})
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=list(result.keys()))
    
    if session:
        return _execute(session)
    else:
        with get_db_session() as sess:
            return _execute(sess)


if __name__ == "__main__":
    print("\n=== OLAP SLICE DEMONSTRATION ===")
    
    # Slice: High severity only
    print("\nSlice: Severity = High")
    df = slice_by_severity("High")
    print(f"Found {len(df)} High severity detections")
    if not df.empty:
        print(df.head(10).to_string(index=False))
