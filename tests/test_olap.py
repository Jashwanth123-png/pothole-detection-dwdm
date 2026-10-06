import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warehouse.models import Base
from warehouse.database import _seed_severity
from warehouse.ingestion import ingest_detection
from olap.rollup import rollup_by_road, rollup_by_area, rollup_by_city
from olap.drilldown import drilldown_cities, drilldown_to_areas, drilldown_to_roads, drilldown_to_records
from olap.slice import slice_by_severity, slice_by_year, slice_by_city
from olap.dice import dice


@pytest.fixture()
def olap_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    _seed_severity(engine)
    records = [
        {"detection_id": "D1", "timestamp": datetime(2024, 1, 10, 9, 0), "city": "Bangalore", "area": "Koramangala", "latitude": 12.93, "longitude": 77.62, "road_name": "80 Feet Road", "road_type": "Urban", "severity": "High", "pothole_count": 3, "confidence": 0.92, "is_demo": 1},
        {"detection_id": "D2", "timestamp": datetime(2024, 2, 15, 10, 0), "city": "Bangalore", "area": "Koramangala", "latitude": 12.93, "longitude": 77.62, "road_name": "80 Feet Road", "road_type": "Urban", "severity": "Medium", "pothole_count": 2, "confidence": 0.78, "is_demo": 1},
        {"detection_id": "D3", "timestamp": datetime(2024, 2, 20, 11, 0), "city": "Bangalore", "area": "Indiranagar", "latitude": 12.97, "longitude": 77.64, "road_name": "100 Feet Road", "road_type": "Urban", "severity": "Low", "pothole_count": 1, "confidence": 0.65, "is_demo": 1},
        {"detection_id": "D4", "timestamp": datetime(2024, 3, 5, 12, 0), "city": "Mumbai", "area": "Andheri", "latitude": 19.12, "longitude": 72.85, "road_name": "Andheri Link Road", "road_type": "Urban", "severity": "High", "pothole_count": 4, "confidence": 0.90, "is_demo": 1},
    ]
    for r in records:
        ingest_detection(session, r)
    session.commit()
    yield session
    session.close()


def test_rollups(olap_session):
    road = rollup_by_road(olap_session)
    area = rollup_by_area(olap_session)
    city = rollup_by_city(olap_session)
    assert isinstance(road, pd.DataFrame) and not road.empty
    assert isinstance(area, pd.DataFrame) and not area.empty
    assert isinstance(city, pd.DataFrame) and not city.empty
    assert len(city) <= len(area) <= len(road)


def test_drilldown(olap_session):
    cities = drilldown_cities(olap_session)
    assert "Bangalore" in cities["city"].tolist()
    areas = drilldown_to_areas("Bangalore", olap_session)
    assert "Koramangala" in areas["area"].tolist()
    roads = drilldown_to_roads("Bangalore", "Koramangala", olap_session)
    assert "80 Feet Road" in roads["road_name"].tolist()
    recs = drilldown_to_records("80 Feet Road", olap_session)
    assert not recs.empty


def test_slice_and_dice(olap_session):
    high = slice_by_severity("High", olap_session)
    assert not high.empty and (high["severity"] == "High").all()
    yearly = slice_by_year(2024, olap_session)
    assert len(yearly) == 4
    city = slice_by_city("Bangalore", olap_session)
    assert not city.empty
    diced = dice(severity="High", city="Bangalore", year=2024, session=olap_session)
    assert len(diced) == 1
