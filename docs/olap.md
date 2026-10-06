# OLAP Operations Documentation

## 1. What is OLAP?

**OLAP** (Online Analytical Processing) refers to a category of software and
query techniques designed for **multidimensional analysis** of large volumes of
historical data.

OLAP answers questions like:
- "How many potholes were detected in Bangalore last quarter?"
- "Which area has the highest concentration of High-severity potholes?"
- "Show me monthly trends broken down by city."

### OLAP vs OLTP

| Feature | OLTP | OLAP |
|---|---|---|
| Purpose | Day-to-day transactions | Analytical queries |
| Operations | INSERT / UPDATE / DELETE | SELECT with aggregations |
| Data volume | Current data | Historical data |
| Query speed | Fast (single row) | Slower (many rows) |
| Schema style | Normalised (3NF) | Denormalised (star schema) |
| Example | Recording a detection | Monthly trend across all cities |

Our project uses **SQLite as an OLTP store** for recording detections, but
the **star schema design** and the OLAP query layer (`olap.py`) make it
behave like an analytical warehouse.

---

## 2. Star Schema and OLAP Connection

OLAP operations are most natural on a **star schema** because:
- The central **fact table** holds the measurable events (detections).
- **Dimension tables** provide the "axes" along which we analyse the facts.
- Joining fact + dimensions creates a **multidimensional cube** in concept.

Our cube has these dimensions (axes):
- **Location** (road → area → city → state)
- **Time** (date → month → quarter → year)
- **Severity** (Low, Medium, High)

The four classical OLAP operations navigate and filter this cube.

---

## 3. Roll-Up

### Definition

**Roll-Up** (also called "aggregation") moves **up** the dimension hierarchy,
combining fine-grained data into coarser summaries.

In our location dimension:
```
Road  →  Area  →  City  →  State
(fine)                      (coarse)
```

### Example from This Project

- **Road level**: "MG Road has 12 potholes, 80 Feet Road has 8 potholes."
- **Area level** (roll up one step): "Koramangala has 20 potholes."
- **City level** (roll up two steps): "Bangalore has 45 potholes."

### SQL Queries Used

```python
# rollup_by_road()
SELECT rd.road_name,
       COUNT(df.fact_id)       AS detection_count,
       AVG(df.confidence)      AS avg_confidence,
       AVG(df.bbox_area_ratio) AS avg_size
FROM Detection_Fact df
JOIN Road_Dim     rd ON df.road_id     = rd.road_id
JOIN Location_Dim ld ON df.location_id = ld.location_id
GROUP BY rd.road_name
ORDER BY detection_count DESC;

# rollup_by_area()
SELECT ld.area,
       COUNT(df.fact_id) AS detection_count,
       AVG(df.confidence) AS avg_confidence
FROM Detection_Fact df
JOIN Location_Dim ld ON df.location_id = ld.location_id
GROUP BY ld.area
ORDER BY detection_count DESC;

# rollup_by_city()
SELECT ld.city,
       COUNT(df.fact_id) AS detection_count,
       AVG(df.confidence) AS avg_confidence
FROM Detection_Fact df
JOIN Location_Dim ld ON df.location_id = ld.location_id
GROUP BY ld.city
ORDER BY detection_count DESC;
```

**In the Streamlit dashboard:** The user selects "Roll-Up level" (Road / Area / City)
from a dropdown and sees a bar chart that dynamically updates.

---

## 4. Drill-Down

### Definition

**Drill-Down** is the **inverse of Roll-Up**. It navigates **down** the hierarchy,
breaking a coarse summary into finer details.

### Example from This Project

1. User sees: "Bangalore has 45 potholes."
2. Drills down: "Which areas in Bangalore?"
   → Koramangala: 20, Indiranagar: 15, Whitefield: 10.
3. Drills down again: "Which roads in Koramangala?"
   → 80 Feet Road: 12, Sarjapur Road: 8.

### SQL Queries Used

```python
# drilldown_cities() — list all cities
SELECT DISTINCT city FROM Location_Dim ORDER BY city;

# drilldown_to_areas(city='Bangalore') — areas in a given city
SELECT DISTINCT ld.area,
       COUNT(df.fact_id) AS detection_count
FROM Detection_Fact df
JOIN Location_Dim ld ON df.location_id = ld.location_id
WHERE ld.city = 'Bangalore'
GROUP BY ld.area
ORDER BY detection_count DESC;

# drilldown_to_roads(area='Koramangala') — roads in a given area
SELECT rd.road_name, COUNT(df.fact_id) AS detection_count
FROM Detection_Fact df
JOIN Road_Dim rd      ON df.road_id     = rd.road_id
JOIN Location_Dim ld  ON df.location_id = ld.location_id
WHERE ld.area = 'Koramangala'
GROUP BY rd.road_name;
```

**In the Streamlit dashboard:** A selectbox lets the user pick a city, then another
selectbox shows its areas, enabling interactive hierarchical exploration.

---

## 5. Slice

### Definition

**Slice** fixes **one dimension** to a specific value, returning a 2-D sub-cube
(a flat table) from the full multidimensional cube.

Think of slicing a loaf of bread — you cut along one axis and get a flat cross-section.

### Example from This Project

- "Show me ALL detections, but only for **High** severity."
- Result: a table of every High-severity pothole across all roads, areas, and months.

### SQL Query Used

```python
# slice_by_severity(severity='High')
SELECT df.detection_id,
       rd.road_name,
       ld.area,
       ld.city,
       td.date,
       sd.severity_label,
       df.confidence,
       df.bbox_area_ratio
FROM Detection_Fact df
JOIN Severity_Dim sd ON df.severity_id  = sd.severity_id
JOIN Road_Dim     rd ON df.road_id      = rd.road_id
JOIN Location_Dim ld ON df.location_id  = ld.location_id
JOIN Time_Dim     td ON df.time_id      = td.time_id
WHERE sd.severity_label = 'High';
```

**In the Streamlit dashboard:** A radio button lets the user pick a severity level.
The filtered table and a map of those detections are shown immediately.

---

## 6. Dice

### Definition

**Dice** applies **multiple filters across multiple dimensions** simultaneously,
producing a sub-cube restricted by all conditions at once.

Think of cutting a cube into a smaller cube — you cut along multiple axes.

### Example from This Project

- "Show me **High-severity** potholes in **Bangalore** detected in **January 2024**."
- All three filters (severity AND city AND month) are applied together.

### SQL Query Used

```python
# dice(severity='High', city='Bangalore', year=2024, month=1)
SELECT df.detection_id,
       rd.road_name,
       ld.area,
       td.date,
       sd.severity_label,
       df.confidence
FROM Detection_Fact df
JOIN Severity_Dim sd ON df.severity_id  = sd.severity_id
JOIN Road_Dim     rd ON df.road_id      = rd.road_id
JOIN Location_Dim ld ON df.location_id  = ld.location_id
JOIN Time_Dim     td ON df.time_id      = td.time_id
WHERE sd.severity_label = 'High'
  AND ld.city            = 'Bangalore'
  AND td.year            = 2024
  AND td.month           = 1;
```

**Dynamic WHERE clause:** The `dice()` function in `olap.py` builds the WHERE
clause programmatically from keyword arguments, so any combination of filters
can be applied without writing new SQL.

**In the Streamlit dashboard:** Multiple filter widgets (multi-select dropdowns)
map directly to `dice()` arguments.

---

## 7. Implementation in This Project

### File: `olap.py`

```
olap.py
├── rollup_by_road(session)           → pd.DataFrame
├── rollup_by_area(session)           → pd.DataFrame
├── rollup_by_city(session)           → pd.DataFrame
├── drilldown_cities(session)         → list[str]
├── drilldown_to_areas(session, city) → list[str]
├── slice_by_severity(session, sev)   → pd.DataFrame
└── dice(session, **filters)          → pd.DataFrame
```

All functions:
1. Accept a SQLAlchemy `Session` object.
2. Build the SQL query using SQLAlchemy Core expressions.
3. Return results as a **pandas DataFrame** for immediate use in Streamlit charts.

### Performance Considerations

- All joins are on indexed surrogate key columns — fast even on large tables.
- SQLite handles up to ~1 million rows without noticeable latency on these queries.
- For production scale (millions of rows), migrate to PostgreSQL and add composite
  indexes on `(severity_id, location_id, time_id)`.

---

## 8. Quick Reference

| Operation | Dimension Fixed | Varies | Goal |
|---|---|---|---|
| Roll-Up | None | Aggregation level ↑ | Summarise detail into coarser totals |
| Drill-Down | None | Aggregation level ↓ | Expand summary into finer details |
| Slice | **One** dimension | All others | Cross-section along one axis |
| Dice | **Multiple** dimensions | Remaining | Sub-cube satisfying multiple conditions |

---

*OLAP documentation — Pothole Detection & Analysis System v1.0*
