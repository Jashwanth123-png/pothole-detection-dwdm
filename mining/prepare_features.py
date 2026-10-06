"""
mining/prepare_features.py
==========================
Feature preparation for K-Means clustering.

Prepares features from the warehouse data for clustering.

Features used (as specified in project requirements):
  1. pothole_count       - Total potholes detected on the road/location
  2. detection_frequency - Number of separate detection events
  3. severity_score      - Encoded severity (Low=1, Medium=2, High=3), averaged
  4. historical_occurrence - Number of unique dates with detections
  5. avg_confidence      - Average detection confidence
  6. road_type_encoded   - Road type as integer (Highway=3, Urban=2, Rural=1, Unknown=0)

Categorical encoding:
  - Severity: Low=1, Medium=2, High=3
  - Road Type: Highway=3, Urban=2, Rural=1, Unknown=0

Normalization:
  - All numeric features are scaled using StandardScaler before clustering.
  - This prevents features with large ranges from dominating the distance metric.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from warehouse.database import get_db_session
from sqlalchemy import text
from utils.helpers import setup_logger

logger = setup_logger(__name__)

# Encoding maps (categorical to numeric)
SEVERITY_ENCODING = {"Low": 1, "Medium": 2, "High": 3}
ROAD_TYPE_ENCODING = {"Highway": 3, "Urban": 2, "Rural": 1, "Unknown": 0}


def load_features_from_warehouse(session=None) -> pd.DataFrame:
    """
    Load and compute features from the warehouse for each road/location combination.

    Each row in the result represents a unique (road, area, city) combination,
    with aggregated statistics that serve as clustering features.

    Args:
        session: Optional SQLAlchemy session. If None, a new session is created.

    Returns:
        DataFrame with feature columns and identifier columns (road_name, area, city)
    """
    query = """
        SELECT
            r.Road_Name         AS road_name,
            r.Road_Type         AS road_type,
            l.Area              AS area,
            l.City              AS city,
            SUM(f.Pothole_Count)           AS pothole_count,
            COUNT(f.Detection_ID)          AS detection_frequency,
            AVG(f.Confidence)              AS avg_confidence,
            COUNT(DISTINCT d.Date)         AS historical_occurrence,
            AVG(CASE s.Severity_Level
                WHEN 'Low'    THEN 1
                WHEN 'Medium' THEN 2
                WHEN 'High'   THEN 3
                ELSE 1 END)                AS severity_score
        FROM Pothole_Detection_Fact f
        JOIN Road_Dim     r ON f.Road_ID     = r.Road_ID
        JOIN Location_Dim l ON f.Location_ID = l.Location_ID
        JOIN Date_Dim     d ON f.Date_ID     = d.Date_ID
        JOIN Severity_Dim s ON f.Severity_ID = s.Severity_ID
        GROUP BY r.Road_ID, r.Road_Name, r.Road_Type, l.Area, l.City
        HAVING COUNT(f.Detection_ID) >= 1
        ORDER BY pothole_count DESC
    """

    def _execute(sess):
        """Execute the query and return a DataFrame."""
        result = sess.execute(text(query))
        rows = result.fetchall()
        cols = result.keys()
        return pd.DataFrame(rows, columns=list(cols))

    if session:
        df = _execute(session)
    else:
        with get_db_session() as sess:
            df = _execute(sess)

    logger.info(f"Loaded {len(df)} road/location records for clustering")
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns to numeric values.

    Applies the following mappings:
      - road_type -> road_type_encoded: Highway=3, Urban=2, Rural=1, Unknown=0

    Args:
        df: DataFrame containing at least a 'road_type' column.

    Returns:
        A copy of the DataFrame with a new 'road_type_encoded' column added.
    """
    df = df.copy()

    # Map road_type strings to integer codes; unmapped values default to 0
    df["road_type_encoded"] = (
        df["road_type"].map(ROAD_TYPE_ENCODING).fillna(0).astype(int)
    )

    logger.debug(f"Encoded road types: {df['road_type'].value_counts().to_dict()}")
    return df


def prepare_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Prepare the normalized feature matrix for K-Means.

    Steps:
      1. Select the six numeric feature columns defined in the module docstring.
      2. Fill any NaN values with 0 (e.g., roads with no confidence data).
      3. Apply StandardScaler so each feature has mean=0 and std=1.

    Args:
        df: DataFrame with all feature columns already present.

    Returns:
        Tuple of:
          - X_scaled (np.ndarray): Normalized feature matrix, shape (n_roads, n_features).
          - scaler (StandardScaler): Fitted scaler (needed for inverse-transform later).
          - feature_names (list[str]): Names of columns used, in order.
    """
    feature_columns = [
        "pothole_count",
        "detection_frequency",
        "severity_score",
        "historical_occurrence",
        "avg_confidence",
        "road_type_encoded",
    ]

    # Warn and drop any columns that are unexpectedly missing
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        logger.warning(f"Missing feature columns (will be skipped): {missing}")
        feature_columns = [c for c in feature_columns if c in df.columns]

    # Build raw matrix; cast to float to satisfy sklearn
    X = df[feature_columns].fillna(0).values.astype(float)

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Feature matrix shape: {X_scaled.shape}")
    logger.debug(f"Feature means (pre-scale): {X.mean(axis=0)}")
    logger.debug(f"Feature stds  (pre-scale): {X.std(axis=0)}")

    return X_scaled, scaler, feature_columns


def get_features_for_clustering(session=None) -> tuple:
    """
    Full pipeline: load warehouse data -> encode categoricals -> normalize -> return.

    This is the single entry-point used by kmeans.py and cluster_analysis.py.

    Args:
        session: Optional SQLAlchemy session.

    Returns:
        Tuple of:
          - df (pd.DataFrame): DataFrame with road_name, area, city, and raw features.
          - X_scaled (np.ndarray): Normalized feature matrix ready for K-Means.
          - scaler (StandardScaler): Fitted scaler instance.
          - feature_names (list[str]): Feature column names in matrix order.

    Raises:
        ValueError: If fewer than 2 records are available (K-Means needs ≥ 2 points).
    """
    df = load_features_from_warehouse(session)

    if len(df) < 2:
        raise ValueError(
            f"Insufficient records for clustering: {len(df)} record(s) found. "
            f"K-Means requires at least 2 data points. "
            f"Please run demo data seeding: python warehouse/seed_demo_data.py"
        )

    df = encode_categorical_features(df)
    X_scaled, scaler, feature_names = prepare_feature_matrix(df)

    return df, X_scaled, scaler, feature_names


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== FEATURE PREPARATION FOR K-MEANS ===")

    try:
        df, X_scaled, scaler, features = get_features_for_clustering()
        print(f"\nRecords loaded:       {len(df)}")
        print(f"Features used:        {features}")
        print(f"Feature matrix shape: {X_scaled.shape}")
        print("\nSample data (first 5 rows):")
        print(df.head().to_string(index=False))
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("Run: python warehouse/seed_demo_data.py")
