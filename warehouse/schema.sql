-- ============================================================
-- POTHOLE DETECTION DATA WAREHOUSE - STAR SCHEMA
-- ============================================================
-- This SQL script creates the data warehouse using a STAR SCHEMA.
-- The schema consists of one FACT table and five DIMENSION tables.
--
-- STAR SCHEMA DIAGRAM:
--
--         Date_Dim
--            |
-- Location_Dim --- Pothole_Detection_Fact --- Road_Dim
--            |
--         Time_Dim
--            |
--       Severity_Dim
--
-- ============================================================

PRAGMA foreign_keys = ON;

-- ============================================================
-- DIMENSION TABLE: Date_Dim
-- Stores date-related attributes for time intelligence.
-- ============================================================
CREATE TABLE IF NOT EXISTS Date_Dim (
    Date_ID     INTEGER PRIMARY KEY AUTOINCREMENT,
    Date        TEXT    NOT NULL UNIQUE,   -- ISO format: YYYY-MM-DD
    Day         INTEGER NOT NULL,          -- Day of month (1-31)
    Month       INTEGER NOT NULL,          -- Month number (1-12)
    Year        INTEGER NOT NULL,          -- 4-digit year
    Quarter     INTEGER NOT NULL,          -- Quarter (1-4)
    Day_Name    TEXT,                      -- e.g., 'Monday'
    Month_Name  TEXT                       -- e.g., 'January'
);

-- ============================================================
-- DIMENSION TABLE: Time_Dim
-- Stores time-of-day attributes.
-- ============================================================
CREATE TABLE IF NOT EXISTS Time_Dim (
    Time_ID     INTEGER PRIMARY KEY AUTOINCREMENT,
    Hour        INTEGER NOT NULL,          -- Hour (0-23)
    Minute      INTEGER NOT NULL,          -- Minute (0-59)
    Time_Period TEXT    NOT NULL           -- 'Morning', 'Afternoon', 'Evening', 'Night'
);

-- ============================================================
-- DIMENSION TABLE: Location_Dim
-- Stores geographic location attributes.
-- ============================================================
CREATE TABLE IF NOT EXISTS Location_Dim (
    Location_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    City        TEXT    NOT NULL,
    Area        TEXT    NOT NULL,
    Latitude    REAL,                      -- Decimal degrees (NULL if GPS unavailable)
    Longitude   REAL,                      -- Decimal degrees (NULL if GPS unavailable)
    Is_Demo     INTEGER DEFAULT 0          -- 1 if this is demo/synthetic data
);

-- ============================================================
-- DIMENSION TABLE: Road_Dim
-- Stores road attributes.
-- ============================================================
CREATE TABLE IF NOT EXISTS Road_Dim (
    Road_ID     INTEGER PRIMARY KEY AUTOINCREMENT,
    Road_Name   TEXT    NOT NULL,
    Road_Type   TEXT    NOT NULL           -- 'Highway', 'Urban', 'Rural', 'Unknown'
);

-- ============================================================
-- DIMENSION TABLE: Severity_Dim
-- Stores severity level attributes.
-- NOTE: Severity is project-defined, NOT from RDD2022.
-- ============================================================
CREATE TABLE IF NOT EXISTS Severity_Dim (
    Severity_ID    INTEGER PRIMARY KEY AUTOINCREMENT,
    Severity_Level TEXT    NOT NULL UNIQUE, -- 'Low', 'Medium', 'High'
    Description    TEXT                     -- Human-readable description
);

-- ============================================================
-- FACT TABLE: Pothole_Detection_Fact
-- Central fact table storing pothole detection events.
-- Each row represents one detection event (one image/frame analysis).
-- ============================================================
CREATE TABLE IF NOT EXISTS Pothole_Detection_Fact (
    Detection_ID    TEXT    PRIMARY KEY,   -- Unique ID: DET-YYYYMMDD-HHMMSS-XXXXXX
    Date_ID         INTEGER NOT NULL,
    Time_ID         INTEGER NOT NULL,
    Location_ID     INTEGER NOT NULL,
    Road_ID         INTEGER NOT NULL,
    Severity_ID     INTEGER NOT NULL,
    Pothole_Count   INTEGER NOT NULL DEFAULT 0,
    Confidence      REAL    NOT NULL,      -- Average confidence of detections (0-1)
    BBox_X1         REAL,                  -- Bounding box coordinates (pixels)
    BBox_Y1         REAL,
    BBox_X2         REAL,
    BBox_Y2         REAL,
    Source_Type     TEXT    DEFAULT 'image', -- 'image', 'video', 'webcam'
    Is_Demo         INTEGER DEFAULT 0,     -- 1 = DEMO DATA, 0 = real detection
    Created_At      TEXT    DEFAULT (datetime('now')),

    FOREIGN KEY (Date_ID)     REFERENCES Date_Dim(Date_ID),
    FOREIGN KEY (Time_ID)     REFERENCES Time_Dim(Time_ID),
    FOREIGN KEY (Location_ID) REFERENCES Location_Dim(Location_ID),
    FOREIGN KEY (Road_ID)     REFERENCES Road_Dim(Road_ID),
    FOREIGN KEY (Severity_ID) REFERENCES Severity_Dim(Severity_ID)
);

-- ============================================================
-- INDEXES for query performance
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_fact_date     ON Pothole_Detection_Fact(Date_ID);
CREATE INDEX IF NOT EXISTS idx_fact_location ON Pothole_Detection_Fact(Location_ID);
CREATE INDEX IF NOT EXISTS idx_fact_road     ON Pothole_Detection_Fact(Road_ID);
CREATE INDEX IF NOT EXISTS idx_fact_severity ON Pothole_Detection_Fact(Severity_ID);
CREATE INDEX IF NOT EXISTS idx_date_year     ON Date_Dim(Year);
CREATE INDEX IF NOT EXISTS idx_date_month    ON Date_Dim(Month);
CREATE INDEX IF NOT EXISTS idx_loc_city      ON Location_Dim(City);

-- ============================================================
-- Seed Severity_Dim with fixed values
-- ============================================================
INSERT OR IGNORE INTO Severity_Dim (Severity_Level, Description) VALUES
    ('Low',    'Small potholes with low confidence — requires monitoring'),
    ('Medium', 'Moderate potholes — schedule for repair'),
    ('High',   'Large/severe potholes — immediate repair required');
