from datetime import date

from sqlalchemy import Date, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base


class DimDate(Base):
    __tablename__ = "dim_date"

    date_id: Mapped[date] = mapped_column(Date, primary_key=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    month_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    month_name: Mapped[str | None] = mapped_column(String(20), nullable=True)


class FactTouristArrivals(Base):
    __tablename__ = "fact_tourist_arrivals"

    date_id: Mapped[date] = mapped_column(Date, primary_key=True)
    international_tourist_arrivals: Mapped[int | None] = mapped_column(Integer, nullable=True)


class FactForecastsEWS(Base):
    __tablename__ = "fact_forecasts_ews"

    date_id: Mapped[date] = mapped_column(Date, primary_key=True)
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    predicted_arrivals: Mapped[int] = mapped_column(Integer, nullable=False)
    seasonal_baseline: Mapped[int] = mapped_column(Integer, nullable=False)
    deviation_pct: Mapped[float] = mapped_column(Float, nullable=False)
    alert_status: Mapped[str] = mapped_column(String(100), nullable=False)
