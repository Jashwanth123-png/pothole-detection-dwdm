"""
warehouse/seed_demo_data.py
===========================
Seeds the Pothole Detection Data Warehouse with clearly labelled DEMO DATA.

╔══════════════════════════════════════════════════════════════════╗
║  ⚠  DEMO DATA — NOT REAL DETECTIONS — NOT FROM RDD2022  ⚠       ║
║  All records are synthetic and exist only for dashboard testing. ║
║  Every record is stored with Is_Demo = 1.                        ║
╚══════════════════════════════════════════════════════════════════╝

Run directly::

    python -m warehouse.seed_demo_data

Or call from Python::

    from warehouse.seed_demo_data import main
    main()
"""

import random
from datetime import datetime, timedelta

from warehouse.database import initialize_database, get_db_session
from warehouse.ingestion import ingest_detection
from utils.helpers import setup_logger

logger = setup_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic city / area / road data
# All coordinates are realistic bounding-box centres for each city.
# ─────────────────────────────────────────────────────────────────────────────

CITY_DATA = {
    "Mumbai": {
        "base_lat": 19.0760,
        "base_lon": 72.8777,
        "areas": {
            "Andheri West":  {"lat_off": +0.06,  "lon_off": -0.05},
            "Bandra East":   {"lat_off": +0.02,  "lon_off": -0.03},
            "Dadar":         {"lat_off": -0.01,  "lon_off": -0.01},
            "Kurla":         {"lat_off": +0.03,  "lon_off": +0.04},
            "Malad West":    {"lat_off": +0.10,  "lon_off": -0.08},
        },
        "roads": [
            ("Link Road",         "Urban"),
            ("Western Express Hwy", "Highway"),
            ("SV Road",           "Urban"),
            ("LBS Marg",          "Urban"),
            ("Eastern Freeway",   "Highway"),
        ],
    },
    "Delhi": {
        "base_lat": 28.7041,
        "base_lon": 77.1025,
        "areas": {
            "Rohini":        {"lat_off": +0.15, "lon_off": -0.10},
            "Dwarka":        {"lat_off": -0.08, "lon_off": -0.20},
            "Lajpat Nagar":  {"lat_off": -0.03, "lon_off": +0.05},
            "Karol Bagh":    {"lat_off": +0.01, "lon_off": -0.02},
            "Saket":         {"lat_off": -0.06, "lon_off": +0.08},
        },
        "roads": [
            ("Ring Road",         "Highway"),
            ("NH-48",             "Highway"),
            ("Outer Ring Road",   "Highway"),
            ("Mehrauli Road",     "Urban"),
            ("Mathura Road",      "Urban"),
        ],
    },
    "Bangalore": {
        "base_lat": 12.9716,
        "base_lon": 77.5946,
        "areas": {
            "Koramangala":   {"lat_off": -0.04, "lon_off": +0.04},
            "Indiranagar":   {"lat_off": +0.03, "lon_off": +0.06},
            "Whitefield":    {"lat_off": +0.06, "lon_off": +0.18},
            "JP Nagar":      {"lat_off": -0.08, "lon_off": -0.02},
            "Hebbal":        {"lat_off": +0.10, "lon_off": +0.01},
        },
        "roads": [
            ("Outer Ring Road",   "Highway"),
            ("Hosur Road",        "Urban"),
            ("Bellary Road",      "Urban"),
            ("Old Airport Road",  "Urban"),
            ("Sarjapur Road",     "Rural"),
        ],
    },
    "Chennai": {
        "base_lat": 13.0827,
        "base_lon": 80.2707,
        "areas": {
            "Anna Nagar":    {"lat_off": +0.05, "lon_off": -0.04},
            "T Nagar":       {"lat_off": -0.02, "lon_off": -0.02},
            "Velachery":     {"lat_off": -0.10, "lon_off": +0.03},
            "Adyar":         {"lat_off": -0.07, "lon_off": +0.01},
            "Tambaram":      {"lat_off": -0.20, "lon_off": -0.05},
        },
        "roads": [
            ("GST Road",          "Highway"),
            ("Inner Ring Road",   "Urban"),
            ("Mount Road",        "Urban"),
            ("OMR",               "Highway"),
            ("ECR",               "Highway"),
        ],
    },
    "Kolkata": {
        "base_lat": 22.5726,
        "base_lon": 88.3639,
        "areas": {
            "Salt Lake":     {"lat_off": +0.05, "lon_off": +0.10},
            "Howrah":        {"lat_off": -0.01, "lon_off": -0.08},
            "Park Street":   {"lat_off": -0.02, "lon_off": +0.01},
            "Dum Dum":       {"lat_off": +0.10, "lon_off": +0.05},
            "Behala":        {"lat_off": -0.08, "lon_off": -0.04},
        },
        "roads": [
            ("EM Bypass",         "Highway"),
            ("VIP Road",          "Urban"),
            ("Diamond Harbour Rd","Urban"),
            ("NH-12",             "Highway"),
            ("Jessore Road",      "Urban"),
        ],
    },
    "Hyderabad": {
        "base_lat": 17.3850,
        "base_lon": 78.4867,
        "areas": {
            "Banjara Hills":  {"lat_off": +0.02, "lon_off": -0.04},
            "Hitech City":    {"lat_off": +0.06, "lon_off": -0.10},
            "Secunderabad":   {"lat_off": +0.04, "lon_off": +0.03},
            "Kukatpally":     {"lat_off": +0.07, "lon_off": -0.07},
            "LB Nagar":       {"lat_off": -0.05, "lon_off": +0.05},
        },
        "roads": [
            ("PVNR Expressway",   "Highway"),
            ("Outer Ring Road",   "Highway"),
            ("NH-44",             "Highway"),
            ("Madhapur Road",     "Urban"),
            ("Tolichowki Road",   "Urban"),
        ],
    },
    "Pune": {
        "base_lat": 18.5204,
        "base_lon": 73.8567,
        "areas": {
            "Kothrud":       {"lat_off": -0.03, "lon_off": -0.08},
            "Hinjewadi":     {"lat_off": +0.08, "lon_off": -0.16},
            "Viman Nagar":   {"lat_off": +0.03, "lon_off": +0.07},
            "Hadapsar":      {"lat_off": -0.04, "lon_off": +0.10},
            "Shivajinagar":  {"lat_off": +0.01, "lon_off": -0.01},
        },
        "roads": [
            ("Pune-Mumbai Expressway", "Highway"),
            ("Baner Road",             "Urban"),
            ("Nagar Road",             "Urban"),
            ("Solapur Highway",        "Highway"),
            ("Katraj-Dehu Road",       "Rural"),
        ],
    },
}

# Severity weights: ~30 % Low, ~45 % Medium, ~25 % High
SEVERITY_WEIGHTS = [("Low", 0.30), ("Medium", 0.45), ("High", 0.25)]
SEVERITIES       = [s for s, _ in SEVERITY_WEIGHTS]
SEVERITY_PROBS   = [p for _, p in SEVERITY_WEIGHTS]

SOURCE_TYPES     = ["image", "image", "image", "video", "webcam"]  # biased to 'image'

# Year to spread demo data across
DEMO_YEAR = 2024


# ─────────────────────────────────────────────────────────────────────────────
# Record builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_record(city: str, area: str, road_name: str, road_type: str,
                  lat: float, lon: float, rng: random.Random) -> dict:
    """
    Build one synthetic detection record dict.

    All numeric values are randomly sampled within realistic ranges.
    All records carry is_demo=1 to distinguish them from real detections.

    Args:
        city      : City name.
        area      : Area name.
        road_name : Road name.
        road_type : Road type string.
        lat       : Latitude (already offset from city centre).
        lon       : Longitude (already offset from city centre).
        rng       : Seeded Random instance for reproducibility.

    Returns:
        dict: Detection record ready for :func:`ingest_detection`.
    """
    # Random date within DEMO_YEAR (Jan–Dec)
    day_of_year = rng.randint(1, 365)
    ts = datetime(DEMO_YEAR, 1, 1) + timedelta(days=day_of_year - 1,
                                                hours=rng.randint(5, 22),
                                                minutes=rng.randint(0, 59))

    severity      = rng.choices(SEVERITIES, weights=SEVERITY_PROBS, k=1)[0]
    pothole_count = rng.randint(1, 5)

    # Confidence: Low severity → lower confidence, High → higher
    conf_ranges = {"Low": (0.25, 0.55), "Medium": (0.50, 0.80), "High": (0.70, 0.95)}
    lo, hi      = conf_ranges[severity]
    confidence  = round(rng.uniform(lo, hi), 4)

    # Bounding box: realistic pixel ranges for a 640×480 image
    x1 = rng.randint(50, 300)
    y1 = rng.randint(50, 250)
    x2 = x1 + rng.randint(60, 200)
    y2 = y1 + rng.randint(40, 150)

    source_type = rng.choice(SOURCE_TYPES)

    return {
        "timestamp"    : ts,
        "city"         : city,
        "area"         : area,
        "latitude"     : round(lat, 6),
        "longitude"    : round(lon, 6),
        "road_name"    : road_name,
        "road_type"    : road_type,
        "severity"     : severity,
        "pothole_count": pothole_count,
        "confidence"   : confidence,
        "bbox"         : {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "source_type"  : source_type,
        "is_demo"      : 1,   # ← ALWAYS 1 for demo data
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main seeding function
# ─────────────────────────────────────────────────────────────────────────────

def main(n_records: int = 150, seed: int = 42) -> None:
    """
    Seed the warehouse with ``n_records`` synthetic demo detection records.

    Steps:
      1. Initialise (or verify) database tables.
      2. Generate records spread across 7 Indian cities, multiple areas
         and roads, covering all 12 months of 2024.
      3. Ingest each record via the normal ETL pipeline.
      4. Print a summary.

    Args:
        n_records (int): Number of demo records to insert (default 150).
        seed      (int): Random seed for reproducibility (default 42).
    """
    print("=" * 65)
    print("  DEMO DATA SEEDING — NOT RDD2022 RESULTS")
    print(f"  Generating {n_records} synthetic records for dashboard testing.")
    print("  All records are marked with Is_Demo = 1.")
    print("=" * 65)

    # ── Step 1: Ensure tables exist ──────────────────────────────────────────
    ok = initialize_database()
    if not ok:
        print("[ERROR] Could not initialise database. Aborting seed.")
        return

    # ── Step 2: If enough DEMO rows already exist, keep the existing dataset.
    from warehouse.models import PotholeDetectionFact
    with get_db_session() as session:
        existing_demo = session.query(PotholeDetectionFact).filter_by(Is_Demo=1).count()
    if existing_demo >= n_records:
        print(f"  DEMO warehouse already contains {existing_demo} records; no new rows added.")
        return existing_demo

    remaining = n_records - existing_demo

    # ── Step 3: Build a pool of (city, area, road, lat, lon) combos ─────────
    rng = random.Random(seed)

    pool = []
    for city, info in CITY_DATA.items():
        base_lat = info["base_lat"]
        base_lon = info["base_lon"]
        areas    = info["areas"]
        roads    = info["roads"]

        for area_name, offsets in areas.items():
            lat = base_lat + offsets["lat_off"] + rng.uniform(-0.005, 0.005)
            lon = base_lon + offsets["lon_off"] + rng.uniform(-0.005, 0.005)
            for road_name, road_type in roads:
                pool.append((city, area_name, road_name, road_type, lat, lon))

    # ── Step 4: Sample only the missing rows and ingest ──────────────────────
    success_count = 0
    fail_count    = 0

    with get_db_session() as session:
        for i in range(remaining):
            # Randomly pick a city/area/road combo from the pool
            city, area, road_name, road_type, lat, lon = rng.choice(pool)

            record = _build_record(city, area, road_name, road_type, lat, lon, rng)

            try:
                ingest_detection(session, record)
                success_count += 1
            except Exception as exc:
                logger.warning(f"Record {i+1} failed: {exc}")
                fail_count += 1

            # Progress indicator every 25 records
            if (i + 1) % 25 == 0:
                print(f"  ✔  {i + 1}/{remaining} records processed …")

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    print("-" * 65)
    print(f"  Seeding complete.")
    print(f"  ✔  Inserted : {success_count}")
    print(f"  ✖  Failed   : {fail_count}")
    print("  NOTE: Check the dashboard — all demo rows are labelled Is_Demo=1")
    print("=" * 65)
    return success_count


def seed_demo_data(n_records: int = 150, seed: int = 42) -> int:
    """Compatibility entry point used by setup and the dashboard."""
    return main(n_records=n_records, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
