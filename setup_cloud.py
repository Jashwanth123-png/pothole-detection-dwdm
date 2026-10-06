"""
setup_cloud.py
==============
Cloud deployment startup helper.

This module is imported ONCE at the top of app.py during Streamlit startup.
It handles:
  1. Downloading models/best.pt from HuggingFace Hub if not present locally.
  2. Auto-seeding the warehouse with demo data if the DB is empty.

IMPORTANT: This file does NOT change any detection logic, outputs, severity
calculations, class labels, or any other project functionality. It only
ensures the required files exist so that the existing code can run unchanged.

Set environment variable:
  HF_MODEL_REPO   = "YourHuggingFaceUsername/pothole-yolo-rdd2022"
  HF_MODEL_FILE   = "best.pt"   (default)
  SKIP_MODEL_DL   = "1"         (to skip download entirely, e.g. local dev)
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. Model download
# ---------------------------------------------------------------------------

def _get_secret_or_env(key: str, default: str = "") -> str:
    """Read a setting from st.secrets first, falling back to os.environ."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            val = str(st.secrets[key]).strip()
            if val:
                return val
    except Exception:
        pass
    return os.environ.get(key, default).strip()


def ensure_model() -> bool:
    """
    Ensure models/best.pt exists. If missing, attempt to download from
    HuggingFace Hub using HF_MODEL_REPO from st.secrets or environment variable.

    Returns True if model is available (either already present or downloaded).
    Returns False if unavailable (will trigger demo mode in app.py, unchanged).
    """
    if os.environ.get("SKIP_MODEL_DL", "").strip() == "1":
        logger.info("SKIP_MODEL_DL=1 — skipping model download.")
        return _model_exists()

    models_dir = PROJECT_ROOT / "models"
    best_pt    = models_dir / "best.pt"

    # Already present and large enough to be a real model
    if best_pt.exists() and best_pt.stat().st_size > 5_000_000:
        logger.info(f"Model already present: {best_pt} ({best_pt.stat().st_size/1e6:.1f} MB)")
        return True

    # Try to download from HuggingFace Hub (supports st.secrets and os.environ)
    hf_repo = _get_secret_or_env("HF_MODEL_REPO")
    hf_file = _get_secret_or_env("HF_MODEL_FILE", "best.pt")
    hf_token = _get_secret_or_env("HF_TOKEN") or _get_secret_or_env("HUGGINGFACE_TOKEN") or None

    if not hf_repo:
        logger.warning(
            "models/best.pt not found and HF_MODEL_REPO not configured in st.secrets or env. "
            "Running in demo mode. To enable real detection on cloud, add "
            'HF_MODEL_REPO = "username/repo" to Streamlit secrets.'
        )
        return False

    try:
        from huggingface_hub import hf_hub_download
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {hf_file} from HuggingFace Hub: {hf_repo} ...")
        local_path = hf_hub_download(
            repo_id=hf_repo,
            filename=hf_file,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        # Move/copy to standard location if needed
        downloaded = Path(local_path)
        if downloaded.resolve() != best_pt.resolve():
            import shutil
            shutil.copy2(str(downloaded), str(best_pt))
        size_mb = best_pt.stat().st_size / 1e6
        logger.info(f"Model downloaded successfully: {best_pt} ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        logger.error(f"Failed to download model from HuggingFace Hub: {exc}")
        return False


def _model_exists() -> bool:
    best_pt = PROJECT_ROOT / "models" / "best.pt"
    return best_pt.exists() and best_pt.stat().st_size > 5_000_000


# ---------------------------------------------------------------------------
# 2. Warehouse auto-seed
# ---------------------------------------------------------------------------

def ensure_warehouse_seeded() -> None:
    """
    On cloud deployments the SQLite DB starts empty every run.
    If the warehouse has no data, seed it with demo data automatically
    so the dashboard is immediately usable.

    This preserves the exact same demo data that the local "Seed Demo Data"
    button produces — no data or logic change.
    """
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from warehouse.database import get_db_session, create_schema
        from warehouse.queries  import get_detection_summary

        # Ensure schema exists first
        try:
            create_schema()
        except Exception:
            pass  # Schema may already exist

        # Check if data exists
        with get_db_session() as session:
            summary = get_detection_summary(session)
            total = summary.get("total_count", 0) if summary else 0

        if total == 0:
            logger.info("Warehouse is empty — auto-seeding demo data for cloud deployment.")
            try:
                from warehouse.seed_demo_data import seed_all
                seed_all()
                logger.info("Demo data seeded successfully.")
            except ImportError:
                # Fallback: try database.seed_demo_data
                try:
                    from warehouse.database import seed_demo_data
                    seed_demo_data()
                    logger.info("Demo data seeded via database.seed_demo_data.")
                except Exception as e2:
                    logger.warning(f"Could not auto-seed demo data: {e2}")
        else:
            logger.info(f"Warehouse already has {total} records — skipping auto-seed.")
    except Exception as exc:
        # Never crash the app — warehouse seeding is best-effort
        logger.warning(f"Warehouse auto-seed skipped: {exc}")


# ---------------------------------------------------------------------------
# 3. Entry point — called once from app.py
# ---------------------------------------------------------------------------

def run_cloud_setup() -> None:
    """
    Run all cloud setup tasks. Safe to call on local dev too (no-ops if
    everything is already in place).
    """
    logging.basicConfig(level=logging.INFO)
    ensure_model()
    ensure_warehouse_seeded()
