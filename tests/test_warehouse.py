import sys
from pathlib import Path
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warehouse.models import Base, PotholeDetectionFact
from warehouse.database import _seed_severity
from warehouse.ingestion import ingest_detection
from warehouse.queries import (
    get_all_detections,
    get_detection_summary,
    get_monthly_trend,
    get_top_roads,
    get_severity_distribution,
    get_detections_for_kmeans,
)


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    _seed_severity(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def record(det_id, severity="High", month=1):
    return {
        "detection_id": det_id,
        "timestamp": datetime(2024, month, 10, 9, 30),
        "city": "Bangalore",
        "area": "Koramangala",
        "latitude": 12.93,
        "longitude": 77.62,
        "road_name": "80 Feet Road",
        "road_type": "Urban",
        "severity": severity,
        "pothole_count": 2,
        "confidence": 0.9,
        "is_demo": 1,
    }


def test_empty_summary(session):
    summary = get_detection_summary(session)
    assert summary["total_detections"] == 0


def test_ingestion_and_queries(session):
    ingest_detection(session, record("D1", "High", 1))
    ingest_detection(session, record("D2", "Low", 2))
    ingest_detection(session, record("D3", "Medium", 2))
    session.commit()

    facts = session.query(PotholeDetectionFact).all()
    assert len(facts) == 3

    # Duplicate ID must not create another row.
    ingest_detection(session, record("D1", "High", 1))
    session.commit()
    assert session.query(PotholeDetectionFact).count() == 3

    summary = get_detection_summary(session)
    assert summary["total_detections"] == 3
    assert summary["total_potholes"] == 6
    assert summary["high_severity_count"] == 1

    all_rows = get_all_detections(session)
    assert len(all_rows) == 3
    assert all_rows[0]["Road_Name"] == "80 Feet Road"

    monthly = get_monthly_trend(session)
    assert len(monthly) == 2

    top = get_top_roads(session, 5)
    assert top and top[0]["Road_Name"] == "80 Feet Road"

    severity = get_severity_distribution(session)
    assert sum(r["Detection_Count"] for r in severity) == 3

    km = get_detections_for_kmeans(session)
    assert len(km) >= 1
