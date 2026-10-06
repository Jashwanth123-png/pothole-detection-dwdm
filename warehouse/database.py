"""
warehouse/database.py
=====================
Database connection and session management.

Supports:
  - SQLite (default, easy to run)
  - MySQL  (configurable via config.yaml)

To switch to MySQL:
  1. Set database.type = 'mysql' in config.yaml
  2. Fill in mysql_host, mysql_port, mysql_user, mysql_password, mysql_database
  3. Install: pip install pymysql
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator

from warehouse.models import Base
from utils.helpers import load_config, get_project_root, setup_logger

logger = setup_logger(__name__)

# ── Module-level singletons (created once per process) ──────────────────────
_engine       = None
_SessionLocal = None


# ─────────────────────────────────────────────────────────────────────────────
# Connection string builder
# ─────────────────────────────────────────────────────────────────────────────

def get_connection_string(config: dict = None) -> str:
    """
    Build the SQLAlchemy database connection string from the project config.

    Reads the ``database`` section of config.yaml:
      - ``type``             : 'sqlite' (default) or 'mysql'
      - ``sqlite_path``      : relative path inside the project root
      - ``mysql_host``       : MySQL host
      - ``mysql_port``       : MySQL port (default 3306)
      - ``mysql_user``       : MySQL username
      - ``mysql_password``   : MySQL password
      - ``mysql_database``   : MySQL database name

    Args:
        config (dict, optional): Pre-loaded configuration dict.
                                 If None, loads from config.yaml automatically.

    Returns:
        str: A valid SQLAlchemy connection string.

    Raises:
        ValueError: If an unsupported database type is specified.
    """
    if config is None:
        config = load_config()

    db_cfg  = config.get("database", {})
    db_type = db_cfg.get("type", "sqlite")

    if db_type == "sqlite":
        # ── SQLite ─────────────────────────────────────────────────────────
        project_root = get_project_root()
        sqlite_path  = db_cfg.get("sqlite_path", "data/warehouse.db")
        db_path      = str(project_root / sqlite_path)

        # Make sure the parent directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn_str = f"sqlite:///{db_path}"
        logger.info(f"Using SQLite database: {db_path}")

    elif db_type == "mysql":
        # ── MySQL ──────────────────────────────────────────────────────────
        host     = db_cfg.get("mysql_host",     "localhost")
        port     = db_cfg.get("mysql_port",     3306)
        user     = db_cfg.get("mysql_user",     "root")
        password = db_cfg.get("mysql_password", "")
        database = db_cfg.get("mysql_database", "pothole_dw")

        conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        logger.info(f"Using MySQL database: {host}:{port}/{database}")

    else:
        raise ValueError(
            f"Unsupported database type: '{db_type}'. "
            "Allowed values are 'sqlite' or 'mysql'."
        )

    return conn_str


# ─────────────────────────────────────────────────────────────────────────────
# Engine (singleton)
# ─────────────────────────────────────────────────────────────────────────────

def get_engine(config: dict = None):
    """
    Get (or create) the SQLAlchemy engine.

    The engine is created once per process and reused (singleton pattern).
    This avoids repeatedly opening / closing connection pools.

    Args:
        config (dict, optional): Configuration dictionary.

    Returns:
        sqlalchemy.engine.Engine: The active database engine.
    """
    global _engine
    if _engine is None:
        conn_str = get_connection_string(config)
        _engine = create_engine(
            conn_str,
            echo=False,         # Set to True for verbose SQL debug output
            pool_pre_ping=True  # Test connection health before handing it out
        )
        logger.debug("SQLAlchemy engine created.")
    return _engine


# ─────────────────────────────────────────────────────────────────────────────
# Database initialisation
# ─────────────────────────────────────────────────────────────────────────────

def initialize_database(config: dict = None) -> bool:
    """
    Create all database tables defined in the ORM models.

    This is safe to call multiple times — SQLAlchemy uses
    ``CREATE TABLE IF NOT EXISTS`` internally, so existing data is preserved.

    After creating tables, seeds Severity_Dim with the three fixed severity
    levels (Low / Medium / High).

    Args:
        config (dict, optional): Configuration dictionary.

    Returns:
        bool: True if initialisation succeeded, False on error.
    """
    try:
        engine = get_engine(config)

        # Create every table registered on Base.metadata
        Base.metadata.create_all(engine)
        logger.info("Database tables created / verified successfully.")

        # Populate the Severity_Dim lookup table
        _seed_severity(engine)

        return True

    except Exception as exc:
        logger.error(f"Failed to initialize database: {exc}")
        return False


def _seed_severity(engine) -> None:
    """
    Insert the three fixed severity rows into Severity_Dim.

    Skips rows that already exist (idempotent). Called automatically by
    :func:`initialize_database`.

    Args:
        engine: Active SQLAlchemy engine.
    """
    from warehouse.models import SeverityDim

    SessionFactory = sessionmaker(bind=engine)
    with SessionFactory() as session:
        severity_data = [
            {
                "Severity_Level": "Low",
                "Description": "Small potholes with low confidence — requires monitoring"
            },
            {
                "Severity_Level": "Medium",
                "Description": "Moderate potholes — schedule for repair"
            },
            {
                "Severity_Level": "High",
                "Description": "Large/severe potholes — immediate repair required"
            },
        ]

        for item in severity_data:
            existing = session.query(SeverityDim).filter_by(
                Severity_Level=item["Severity_Level"]
            ).first()
            if not existing:
                session.add(SeverityDim(**item))

        session.commit()
        logger.debug("Severity_Dim seeded with Low / Medium / High rows.")


# ─────────────────────────────────────────────────────────────────────────────
# Session factory (singleton)
# ─────────────────────────────────────────────────────────────────────────────

def get_session_factory(config: dict = None):
    """
    Get the SQLAlchemy session factory (sessionmaker).

    Created once and reused. Bind it to the current engine.

    Args:
        config (dict, optional): Configuration dictionary.

    Returns:
        sessionmaker: Callable that produces new Session objects.
    """
    global _SessionLocal
    if _SessionLocal is None:
        engine        = get_engine(config)
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
    return _SessionLocal


# ─────────────────────────────────────────────────────────────────────────────
# Context-manager session (the recommended way to use sessions)
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_db_session(config: dict = None) -> Generator[Session, None, None]:
    """
    Context manager that yields a database session.

    Automatically commits on clean exit and rolls back on any exception,
    then closes the session in all cases.

    Usage::

        with get_db_session() as session:
            records = session.query(PotholeDetectionFact).all()

    Args:
        config (dict, optional): Configuration dictionary.

    Yields:
        sqlalchemy.orm.Session: An active database session.

    Raises:
        Exception: Re-raises any exception after rolling back the transaction.
    """
    SessionLocal = get_session_factory(config)
    session      = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error(f"Database session error (rolled back): {exc}")
        raise
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Connection health check
# ─────────────────────────────────────────────────────────────────────────────

def test_connection(config: dict = None) -> bool:
    """
    Verify that a database connection can be established.

    Executes a trivial ``SELECT 1`` query to confirm liveness.

    Args:
        config (dict, optional): Configuration dictionary.

    Returns:
        bool: True if the connection succeeds, False otherwise.
    """
    try:
        engine = get_engine(config)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection test: PASSED")
        return True
    except Exception as exc:
        logger.error(f"Database connection test FAILED: {exc}")
        return False



def get_connection(config: dict = None):
    """Return a SQLAlchemy Connection for pandas/SQL queries."""
    return get_engine(config).connect()


def create_schema(config: dict = None) -> bool:
    """Compatibility wrapper for initialize_database()."""
    return initialize_database(config)


def insert_detection(detection_record: dict, config: dict = None) -> str:
    """Insert one detection record using the normal warehouse ETL pipeline.

    The dashboard may provide ``detection_date`` instead of a full timestamp;
    this wrapper normalizes that UI payload before passing it to the ETL layer.
    """
    from datetime import datetime
    from warehouse.ingestion import ingest_detection

    record = dict(detection_record)
    if "timestamp" not in record:
        raw_date = record.get("detection_date")
        if raw_date is None:
            record["timestamp"] = datetime.now()
        elif hasattr(raw_date, "year"):
            record["timestamp"] = datetime(raw_date.year, raw_date.month, raw_date.day)
        else:
            record["timestamp"] = datetime.fromisoformat(str(raw_date))

    source_map = {
        "YOLO Image": "image",
        "YOLO Video": "video",
        "YOLO Webcam": "webcam",
        "Manual Entry": "manual",
    }
    record["source_type"] = source_map.get(record.get("source_type"), record.get("source_type", "image"))

    with get_db_session(config) as session:
        return ingest_detection(session, record)


def seed_demo_data(n_records: int = 150, seed: int = 42) -> int:
    """Seed clearly-marked synthetic demo records."""
    from warehouse.seed_demo_data import main
    return main(n_records=n_records, seed=seed)

# ─────────────────────────────────────────────────────────────────────────────
# Engine reset (used in unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def reset_engine() -> None:
    """
    Dispose the current engine and clear all singletons.

    Useful in test suites that need a fresh engine / session factory per test.
    After calling this function, the next call to :func:`get_engine` will
    create a brand-new engine.
    """
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
        logger.debug("SQLAlchemy engine disposed.")
    _engine       = None
    _SessionLocal = None
