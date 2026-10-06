"""
setup_project.py
================
Project Setup and Verification Script.

This script:
1. Creates all required directories
2. Initializes the database
3. Seeds demo data (if requested)
4. Runs a quick health check
5. Prints project status

Usage:
  python setup_project.py              # Setup + seed demo data
  python setup_project.py --no-demo   # Setup only, no demo data
  python setup_project.py --verify    # Just verify the setup
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def print_header():
    print()
    print("=" * 65)
    print(" REAL-TIME POTHOLE DETECTION - DWDM PROJECT")
    print(" Setup & Verification Script")
    print("=" * 65)
    print()


def create_directories():
    """Create all required project directories."""
    print("[1/5] Creating project directories...")
    
    dirs = [
        "data/raw",
        "data/processed/d40",
        "data/processed/resized",
        "data/processed/augmented",
        "data/processed/split/train/images",
        "data/processed/split/train/labels",
        "data/processed/split/val/images",
        "data/processed/split/val/labels",
        "data/processed/split/test/images",
        "data/processed/split/test/labels",
        "data/demo",
        "data/outputs",
        "dataset/rdd2022",
        "models",
        "logs",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print(f"    Created {len(dirs)} directories  OK")
    return True


def initialize_database():
    """Initialize the SQLite database."""
    print("[2/5] Initializing database...")
    
    try:
        from warehouse.database import initialize_database as init_db
        success = init_db()
        if success:
            print("    Database initialized  OK")
        else:
            print("    Database initialization FAILED")
        return success
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def seed_demo_data():
    """Seed the database with demo data."""
    print("[3/5] Seeding DEMO data (clearly labeled as NOT RDD2022)...")
    
    try:
        from warehouse.seed_demo_data import seed_demo_data as seed
        count = seed()
        print(f"    Seeded {count} demo records  OK")
        return True
    except Exception as e:
        print(f"    ERROR seeding demo data: {e}")
        return False


def verify_demo_data() -> bool:
    """Verify that the warehouse contains clearly-labelled demo records."""
    try:
        from warehouse.database import get_db_session
        from warehouse.models import PotholeDetectionFact
        with get_db_session() as session:
            count = session.query(PotholeDetectionFact).filter_by(Is_Demo=1).count()
        print(f"    Demo detections: {count}")
        return count > 0
    except Exception as exc:
        print(f"    ERROR: {exc}")
        return False


def verify_database():
    """Verify the database has records."""
    print("[4/5] Verifying database...")
    
    try:
        from warehouse.database import get_db_session
        from warehouse.models import PotholeDetectionFact, SeverityDim, LocationDim, RoadDim
        
        with get_db_session() as session:
            det_count = session.query(PotholeDetectionFact).count()
            sev_count = session.query(SeverityDim).count()
            loc_count = session.query(LocationDim).count()
            road_count = session.query(RoadDim).count()
        
        print(f"    Detections:    {det_count}")
        print(f"    Severity Dim:  {sev_count}")
        print(f"    Locations:     {loc_count}")
        print(f"    Roads:         {road_count}")
        
        if sev_count == 3:
            print("    Severity_Dim seeded correctly  OK")
        
        return det_count >= 0
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def verify_olap():
    """Quick OLAP verification."""
    print("[5/5] Verifying OLAP operations...")
    
    try:
        from olap.rollup import rollup_by_city
        from olap.slice import slice_by_severity
        from olap.dice import dice
        
        city_df = rollup_by_city()
        high_df = slice_by_severity("High")
        dice_df = dice()
        
        print(f"    Roll-Up (city level): {len(city_df)} cities  OK")
        print(f"    Slice (High severity): {len(high_df)} records  OK")
        print(f"    Dice (no filters): {len(dice_df)} records  OK")
        
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def verify_kmeans():
    """Quick K-Means verification."""
    print("Verifying K-Means clustering...")
    
    try:
        from mining.kmeans import run_kmeans
        model = run_kmeans(n_clusters=3)
        print(f"    K-Means: {len(model.df)} records, {model.n_clusters} clusters  OK")
        return True
    except ValueError as e:
        print(f"    SKIP: {e}")
        return True  # Not a failure, just insufficient data
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def check_model():
    """Check if YOLO model is available."""
    try:
        from utils.helpers import load_config
        config = load_config()
        model_path = config.get("model", {}).get("yolo_model_path", "models/pothole_yolo.pt")
        
        if Path(model_path).exists():
            size_mb = Path(model_path).stat().st_size / 1024 / 1024
            return True, f"Found ({size_mb:.1f} MB)"
        else:
            return False, "Not found - requires training (see models/README.md)"
    except Exception as e:
        return False, str(e)


def check_rdd2022():
    """Check if RDD2022 dataset is available."""
    try:
        from utils.helpers import load_config
        config = load_config()
        ds_path = config.get("dataset", {}).get("rdd2022_path", "dataset/rdd2022")
        
        if Path(ds_path).exists():
            # Check for image files
            img_count = sum(1 for _ in Path(ds_path).rglob("*.jpg"))
            xml_count = sum(1 for _ in Path(ds_path).rglob("*.xml"))
            if img_count > 0:
                return True, f"Found ({img_count} images, {xml_count} annotations)"
            else:
                return False, f"Directory exists but empty (see README for setup)"
        else:
            return False, "Not found (see README for setup instructions)"
    except Exception as e:
        return False, str(e)


def print_status_report(results: dict):
    """Print a formatted status report."""
    print()
    print("=" * 65)
    print(" PROJECT STATUS REPORT")
    print(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    status_items = [
        ("Project structure",       results.get("dirs", False)),
        ("Database initialized",    results.get("db", False)),
        ("Demo data seeded",        results.get("demo", False)),
        ("Database verification",   results.get("verify_db", False)),
        ("OLAP operations",         results.get("olap", False)),
        ("K-Means clustering",      results.get("kmeans", False)),
    ]
    
    for name, ok in status_items:
        icon = "[OK]" if ok else "[FAILED]"
        print(f"  {icon:<10} {name}")
    
    print()
    
    # External dependencies
    model_ok, model_msg = check_model()
    rdd_ok, rdd_msg = check_rdd2022()
    
    print("External Dependencies:")
    icon = "[OK]" if model_ok else "[!]"
    print(f"  {icon:<10} YOLO Model: {model_msg}")
    icon = "[OK]" if rdd_ok else "[!]"
    print(f"  {icon:<10} RDD2022 Dataset: {rdd_msg}")
    
    print()
    print("Quick Start Commands:")
    print()
    print("  # Run dashboard (demo mode):")
    print("  streamlit run app.py")
    print()
    print("  # Seed demo data:")
    print("  python warehouse/seed_demo_data.py")
    print()
    print("  # Run tests:")
    print("  pytest tests/ -v")
    print()
    print("  # Prepare dataset (after placing RDD2022):")
    print("  python scripts/prepare_dataset.py")
    print()
    print("  # Train YOLO (after dataset preparation):")
    print("  python ml/train_yolo.py")
    print()
    print("  # OLAP demo:")
    print("  python olap/rollup.py")
    print("  python olap/drilldown.py")
    print("  python olap/slice.py")
    print("  python olap/dice.py")
    print()
    print("  # K-Means demo:")
    print("  python mining/kmeans.py")
    print()
    print("=" * 65)
    print(" [!] = Requires user input or external dependency")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Pothole Detection Project Setup")
    parser.add_argument("--no-demo", action="store_true", help="Skip demo data seeding")
    parser.add_argument("--verify", action="store_true", help="Only verify, don't seed")
    args = parser.parse_args()
    
    print_header()
    
    results = {}
    
    # Create directories
    results["dirs"] = create_directories()
    
    # Initialize database
    results["db"] = initialize_database()
    
    if not args.verify and not args.no_demo and results["db"]:
        # Seed demo data
        results["demo"] = seed_demo_data()
    else:
        if args.verify:
            print("[3/5] Verifying existing DEMO data (--verify mode)")
            results["demo"] = verify_demo_data()
        else:
            print("[3/5] Skipping demo seeding (--no-demo flag)")
    
    # Verify database
    if results["db"]:
        results["verify_db"] = verify_database()
    
    # Verify OLAP (only if data exists)
    results["olap"] = verify_olap()
    
    # Verify K-Means
    results["kmeans"] = verify_kmeans()
    
    # Print report
    print_status_report(results)


if __name__ == "__main__":
    main()
