import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warehouse.models import Base
from warehouse.database import _seed_severity
from warehouse.ingestion import ingest_detection
from mining.prepare_features import encode_categorical_features, prepare_feature_matrix, get_features_for_clustering
from mining.kmeans import PotholeKMeans, run_kmeans


def test_feature_preparation():
    df = pd.DataFrame({
        "pothole_count": [10, 20, 5, 15],
        "detection_frequency": [4, 7, 2, 5],
        "severity_score": [1, 3, 2, 2],
        "historical_occurrence": [2, 5, 1, 3],
        "avg_confidence": [0.55, 0.88, 0.70, 0.80],
        "road_type": ["Rural", "Highway", "Urban", "Urban"],
    })
    encoded = encode_categorical_features(df)
    X, scaler, names = prepare_feature_matrix(encoded)
    assert X.shape == (4, 6)
    assert len(names) == 6
    assert np.allclose(X.mean(axis=0), 0, atol=1e-7)
    assert encoded["road_type_encoded"].tolist() == [1, 3, 2, 2]


def test_kmeans_on_warehouse():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    _seed_severity(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        cities = [("Bangalore", "Koramangala", "80 Feet Road", "Urban"),
                  ("Mumbai", "Andheri", "Link Road", "Urban"),
                  ("Delhi", "Lajpat Nagar", "Ring Road", "Highway"),
                  ("Pune", "Kothrud", "Baner Road", "Rural")]
        for i, (city, area, road, road_type) in enumerate(cities):
            ingest_detection(session, {
                "detection_id": f"K{i}",
                "timestamp": datetime(2024, 1 + i, 10, 9, 0),
                "city": city,
                "area": area,
                "latitude": 10 + i,
                "longitude": 70 + i,
                "road_name": road,
                "road_type": road_type,
                "severity": ["Low", "Medium", "High", "Medium"][i],
                "pothole_count": 2 + i,
                "confidence": 0.6 + i * 0.08,
                "is_demo": 1,
            })
        session.commit()

        df, X, scaler, names = get_features_for_clustering(session)
        assert len(df) == 4
        assert X.shape[0] == 4

        model = PotholeKMeans(n_clusters=2)
        out = model.fit(session)
        assert "cluster_id" in out.columns
        assert out["cluster_id"].nunique() == 2
        assert -1.0 <= model.get_silhouette_score() <= 1.0
        assert len(model.get_cluster_summary()) == 2
    finally:
        session.close()
