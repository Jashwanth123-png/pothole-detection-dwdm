"""
warehouse/__init__.py
=====================
Pothole Detection Data Warehouse package.

Exposes the most commonly used symbols so callers can do:
    from warehouse import get_db_session, ingest_detection_safe
"""

from warehouse.models import (
    Base,
    DateDim,
    TimeDim,
    LocationDim,
    RoadDim,
    SeverityDim,
    PotholeDetectionFact,
)
from warehouse.database import (
    get_engine,
    get_db_session,
    initialize_database,
    test_connection,
)
from warehouse.ingestion import (
    ingest_detection,
    ingest_detection_safe,
)
from warehouse.queries import (
    get_all_detections,
    get_detection_summary,
    get_monthly_trend,
    get_top_roads,
    get_severity_distribution,
    get_detections_by_location,
    get_detections_for_kmeans,
)

__all__ = [
    # Models
    "Base",
    "DateDim",
    "TimeDim",
    "LocationDim",
    "RoadDim",
    "SeverityDim",
    "PotholeDetectionFact",
    # Database
    "get_engine",
    "get_db_session",
    "initialize_database",
    "test_connection",
    # Ingestion
    "ingest_detection",
    "ingest_detection_safe",
    # Queries
    "get_all_detections",
    "get_detection_summary",
    "get_monthly_trend",
    "get_top_roads",
    "get_severity_distribution",
    "get_detections_by_location",
    "get_detections_for_kmeans",
]
