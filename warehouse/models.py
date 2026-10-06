"""
warehouse/models.py
===================
SQLAlchemy ORM models for the Pothole Detection Data Warehouse.

This module defines the STAR SCHEMA:
  - Fact Table:  Pothole_Detection_Fact
  - Dimensions:  Date_Dim, Time_Dim, Location_Dim, Road_Dim, Severity_Dim

Using SQLAlchemy's declarative_base for ORM mapping.
"""

from sqlalchemy import (
    Column, Integer, Text, Float, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship

# Base class for all ORM models
Base = declarative_base()


class DateDim(Base):
    """
    DATE DIMENSION
    Stores date-related attributes for time-series analysis.
    One row per unique calendar date.
    """
    __tablename__ = "Date_Dim"

    Date_ID    = Column(Integer, primary_key=True, autoincrement=True)
    Date       = Column(Text, nullable=False, unique=True)  # YYYY-MM-DD
    Day        = Column(Integer, nullable=False)
    Month      = Column(Integer, nullable=False)
    Year       = Column(Integer, nullable=False)
    Quarter    = Column(Integer, nullable=False)
    Day_Name   = Column(Text)
    Month_Name = Column(Text)

    # Relationship to fact table
    detections = relationship("PotholeDetectionFact", back_populates="date_dim")

    def __repr__(self):
        return f"<DateDim Date={self.Date}>"


class TimeDim(Base):
    """
    TIME DIMENSION
    Stores time-of-day attributes.
    """
    __tablename__ = "Time_Dim"

    Time_ID     = Column(Integer, primary_key=True, autoincrement=True)
    Hour        = Column(Integer, nullable=False)
    Minute      = Column(Integer, nullable=False)
    Time_Period = Column(Text, nullable=False)  # Morning/Afternoon/Evening/Night

    detections = relationship("PotholeDetectionFact", back_populates="time_dim")

    def __repr__(self):
        return f"<TimeDim {self.Hour:02d}:{self.Minute:02d} ({self.Time_Period})>"


class LocationDim(Base):
    """
    LOCATION DIMENSION
    Stores geographic location attributes.
    If GPS is unavailable, Latitude/Longitude may be NULL.
    """
    __tablename__ = "Location_Dim"

    Location_ID = Column(Integer, primary_key=True, autoincrement=True)
    City        = Column(Text, nullable=False)
    Area        = Column(Text, nullable=False)
    Latitude    = Column(Float)              # NULL if GPS unavailable
    Longitude   = Column(Float)              # NULL if GPS unavailable
    Is_Demo     = Column(Integer, default=0)  # 1 = demo/synthetic data

    detections = relationship("PotholeDetectionFact", back_populates="location_dim")

    def __repr__(self):
        return f"<LocationDim {self.City}/{self.Area}>"


class RoadDim(Base):
    """
    ROAD DIMENSION
    Stores road information.
    """
    __tablename__ = "Road_Dim"

    Road_ID   = Column(Integer, primary_key=True, autoincrement=True)
    Road_Name = Column(Text, nullable=False)
    Road_Type = Column(Text, nullable=False)  # Highway / Urban / Rural / Unknown

    detections = relationship("PotholeDetectionFact", back_populates="road_dim")

    def __repr__(self):
        return f"<RoadDim {self.Road_Name} ({self.Road_Type})>"


class SeverityDim(Base):
    """
    SEVERITY DIMENSION
    Stores severity levels.
    NOTE: These are project-defined levels, NOT official RDD2022 labels.
    """
    __tablename__ = "Severity_Dim"

    Severity_ID    = Column(Integer, primary_key=True, autoincrement=True)
    Severity_Level = Column(Text, nullable=False, unique=True)  # Low/Medium/High
    Description    = Column(Text)

    detections = relationship("PotholeDetectionFact", back_populates="severity_dim")

    def __repr__(self):
        return f"<SeverityDim {self.Severity_Level}>"


class PotholeDetectionFact(Base):
    """
    FACT TABLE: Pothole_Detection_Fact
    Central table of the star schema.
    Each row = one pothole detection event.
    """
    __tablename__ = "Pothole_Detection_Fact"

    Detection_ID  = Column(Text, primary_key=True)   # DET-YYYYMMDD-HHMMSS-XXXXXX
    Date_ID       = Column(Integer, ForeignKey("Date_Dim.Date_ID"),         nullable=False)
    Time_ID       = Column(Integer, ForeignKey("Time_Dim.Time_ID"),         nullable=False)
    Location_ID   = Column(Integer, ForeignKey("Location_Dim.Location_ID"), nullable=False)
    Road_ID       = Column(Integer, ForeignKey("Road_Dim.Road_ID"),         nullable=False)
    Severity_ID   = Column(Integer, ForeignKey("Severity_Dim.Severity_ID"), nullable=False)
    Pothole_Count = Column(Integer, nullable=False, default=0)
    Confidence    = Column(Float, nullable=False)
    BBox_X1       = Column(Float)
    BBox_Y1       = Column(Float)
    BBox_X2       = Column(Float)
    BBox_Y2       = Column(Float)
    Source_Type   = Column(Text, default="image")   # image / video / webcam
    Is_Demo       = Column(Integer, default=0)       # 1 = DEMO DATA
    Created_At    = Column(Text)

    # Relationships to dimension tables
    date_dim     = relationship("DateDim",     back_populates="detections")
    time_dim     = relationship("TimeDim",     back_populates="detections")
    location_dim = relationship("LocationDim", back_populates="detections")
    road_dim     = relationship("RoadDim",     back_populates="detections")
    severity_dim = relationship("SeverityDim", back_populates="detections")

    def __repr__(self):
        return (f"<PotholeDetectionFact id={self.Detection_ID} "
                f"count={self.Pothole_Count} severity_id={self.Severity_ID}>")
