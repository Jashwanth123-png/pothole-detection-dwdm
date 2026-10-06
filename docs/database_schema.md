# Data Warehouse Schema Documentation

## 1. What is a Star Schema?

A **star schema** is a dimensional modelling approach used in data warehouses.
It organises data around a central **fact table** that records business events,
surrounded by multiple **dimension tables** that provide descriptive context.

The schema looks like a star when drawn — the fact table at the centre, dimension
tables at the tips.

**Why star schema (and not snowflake)?**
- Fewer joins → faster OLAP queries.
- Simpler SQL → easier to understand for analysts.
- Dimension tables in our project are small (< 1 000 rows), so de-normalisation
  overhead is negligible.
- Snowflake normalisation would add join depth without meaningful storage savings.

---

## 2. Star Schema ASCII Diagram

```
                        ┌─────────────────────┐
                        │    Severity_Dim      │
                        │─────────────────────│
                        │ PK severity_id (INT) │
                        │    severity_label    │
                        │    description       │
                        └──────────┬──────────┘
                                   │ FK severity_id
                                   │
┌─────────────────┐    ┌──────────▼──────────────────────────────┐    ┌──────────────────────┐
│   Road_Dim      │    │            Detection_Fact                │    │   Location_Dim       │
│─────────────────│    │─────────────────────────────────────────│    │──────────────────────│
│ PK road_id      │◄───┤ PK fact_id           (INT, autoincr.)   ├───►│ PK location_id (INT) │
│    road_name    │    │ FK road_id           → Road_Dim         │    │    area              │
│    road_type    │    │ FK location_id       → Location_Dim     │    │    city              │
│    surface_type │    │ FK time_id           → Time_Dim         │    │    state             │
└─────────────────┘    │ FK severity_id       → Severity_Dim     │    └──────────────────────┘
                       │    detection_id      (TEXT, UNIQUE)     │
                       │    confidence        (FLOAT)            │    ┌──────────────────────┐
                       │    bbox_area_ratio   (FLOAT)            │    │   Time_Dim           │
                       │    image_source      (TEXT)             ├───►│──────────────────────│
                       └─────────────────────────────────────────┘    │ PK time_id    (INT)  │
                                                                       │    date       (DATE) │
                                                                       │    month      (INT)  │
                                                                       │    year       (INT)  │
                                                                       │    quarter    (INT)  │
                                                                       └──────────────────────┘
```

---

## 3. Fact Table — `Detection_Fact`

The fact table records **each individual pothole detection event**.

| Column | Type | Description |
|---|---|---|
| `fact_id` | INTEGER (PK, autoincrement) | Surrogate primary key |
| `detection_id` | TEXT (UNIQUE, NOT NULL) | Natural business key — e.g. `"DET-20240115-001"`. Ensures idempotency: the same detection can never be ingested twice. |
| `road_id` | INTEGER (FK → Road_Dim) | Which road the pothole was on |
| `location_id` | INTEGER (FK → Location_Dim) | Area/city of the detection |
| `time_id` | INTEGER (FK → Time_Dim) | Date dimension surrogate key |
| `severity_id` | INTEGER (FK → Severity_Dim) | Severity classification |
| `confidence` | FLOAT | YOLO model confidence score (0.0 – 1.0) |
| `bbox_area_ratio` | FLOAT | Bounding-box area ÷ image area; used to assign severity |
| `image_source` | TEXT | Origin: `"upload"`, `"webcam"`, or `"demo"` |

**Measures (numeric columns used in aggregations):**
- `confidence` — average confidence per road / area / city
- `bbox_area_ratio` — proxy for pothole size

---

## 4. Dimension Tables

### 4.1 `Severity_Dim`

Lookup table for the three severity tiers, seeded once at startup.

| Column | Type | Description |
|---|---|---|
| `severity_id` | INTEGER (PK) | Surrogate key |
| `severity_label` | TEXT | `"Low"`, `"Medium"`, or `"High"` |
| `description` | TEXT | Human-readable explanation, e.g. `"Small pothole < 3 % of image area"` |

**Seeded rows:**

| severity_id | severity_label | description |
|---|---|---|
| 1 | Low | bbox_area_ratio < 0.03 — minor surface crack |
| 2 | Medium | 0.03 ≤ bbox_area_ratio < 0.07 — moderate pothole |
| 3 | High | bbox_area_ratio ≥ 0.07 — large pothole; urgent repair needed |

---

### 4.2 `Road_Dim`

Describes the road where the detection occurred.

| Column | Type | Description |
|---|---|---|
| `road_id` | INTEGER (PK, autoincrement) | Surrogate key |
| `road_name` | TEXT | Human-readable road name, e.g. `"80 Feet Road"` |
| `road_type` | TEXT | Classification: `"Urban"`, `"Highway"`, `"Rural"` |
| `surface_type` | TEXT | Optional: `"Asphalt"`, `"Concrete"`, `"Gravel"` |

---

### 4.3 `Location_Dim`

Captures the geographic hierarchy used by OLAP Roll-Up/Drill-Down.

| Column | Type | Description |
|---|---|---|
| `location_id` | INTEGER (PK, autoincrement) | Surrogate key |
| `area` | TEXT | Neighbourhood / sub-district level, e.g. `"Koramangala"` |
| `city` | TEXT | City level, e.g. `"Bangalore"` |
| `state` | TEXT | State/province level, e.g. `"Karnataka"` |

**Geographic hierarchy used in OLAP:**
```
State → City → Area → Road (in Road_Dim)
```

---

### 4.4 `Time_Dim`

Decomposes detection dates into calendar attributes for time-based analysis.

| Column | Type | Description |
|---|---|---|
| `time_id` | INTEGER (PK, autoincrement) | Surrogate key |
| `date` | DATE | Full date of detection (YYYY-MM-DD) |
| `month` | INTEGER | Month number (1 – 12) |
| `year` | INTEGER | Four-digit year |
| `quarter` | INTEGER | Quarter number (1 – 4) |

---

## 5. Example SQL Queries

### 5.1 Total detections per city

```sql
SELECT
    ld.city,
    COUNT(df.fact_id)  AS total_detections,
    AVG(df.confidence) AS avg_confidence
FROM Detection_Fact df
JOIN Location_Dim ld ON df.location_id = ld.location_id
GROUP BY ld.city
ORDER BY total_detections DESC;
```

### 5.2 Roll-Up — road → area → city

```sql
-- Road level
SELECT rd.road_name, COUNT(*) AS detections
FROM Detection_Fact df JOIN Road_Dim rd ON df.road_id = rd.road_id
GROUP BY rd.road_name;

-- Area level (Roll-Up one step)
SELECT ld.area, COUNT(*) AS detections
FROM Detection_Fact df JOIN Location_Dim ld ON df.location_id = ld.location_id
GROUP BY ld.area;

-- City level (Roll-Up two steps)
SELECT ld.city, COUNT(*) AS detections
FROM Detection_Fact df JOIN Location_Dim ld ON df.location_id = ld.location_id
GROUP BY ld.city;
```

### 5.3 Slice — only High-severity detections

```sql
SELECT df.*, sd.severity_label
FROM Detection_Fact df
JOIN Severity_Dim sd ON df.severity_id = sd.severity_id
WHERE sd.severity_label = 'High';
```

### 5.4 Dice — High severity AND specific city AND year

```sql
SELECT df.detection_id, rd.road_name, ld.area, td.year
FROM Detection_Fact df
JOIN Severity_Dim sd ON df.severity_id  = sd.severity_id
JOIN Location_Dim ld ON df.location_id  = ld.location_id
JOIN Road_Dim     rd ON df.road_id      = rd.road_id
JOIN Time_Dim     td ON df.time_id      = td.time_id
WHERE sd.severity_label = 'High'
  AND ld.city            = 'Bangalore'
  AND td.year            = 2024;
```

### 5.5 Monthly trend

```sql
SELECT
    td.year,
    td.month,
    COUNT(df.fact_id) AS monthly_count
FROM Detection_Fact df
JOIN Time_Dim td ON df.time_id = td.time_id
GROUP BY td.year, td.month
ORDER BY td.year, td.month;
```

---

## 6. Design Choices

| Decision | Rationale |
|---|---|
| Surrogate keys (INTEGER autoincrement) | Decouple physical row identity from business meaning; safe for merges |
| `detection_id` UNIQUE constraint | Prevents duplicate ingestion on re-runs — crucial for idempotent ETL |
| SQLite for storage | Zero-setup, single-file, cross-platform; trivial to swap for PostgreSQL |
| SQLAlchemy ORM | Database-agnostic; connection string change is all that's needed to scale |
| Severity as dimension (not just a string in Fact) | Enables fast WHERE / GROUP-BY on severity without full-text comparison |
| Time_Dim with month/year/quarter columns | Avoids repeated `STRFTIME` calls; supports quarterly trend analysis |

---

*Schema version 1.0 — Pothole Detection & Analysis System*
