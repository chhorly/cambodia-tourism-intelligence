from datetime import date
from sqlalchemy import Column, Date, Float, Integer, String
from database.connection import Base

class ForecastRecord(Base):
    __tablename__ = "forecasts_ews"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    date_id = Column(Date, unique=True, nullable=False, index=True)
    model_version = Column(String, default="Tuned_LightGBM_v1")
    actual_arrivals = Column(Integer, nullable=True)
    predicted_arrivals = Column(Integer, nullable=False)
    seasonal_baseline = Column(Integer, nullable=False)
    deviation_pct = Column(Float, nullable=False)
    alert_status = Column(String, nullable=False)

def get_all_forecasts(db):
    return db.query(ForecastRecord).order_by(ForecastRecord.date_id.asc()).all()

def save_or_update_forecast(db, payload: dict):
    rec = db.query(ForecastRecord).filter(ForecastRecord.date_id == payload["date_id"]).first()
    if rec:
        for k, v in payload.items():
            setattr(rec, k, v)
    else:
        rec = ForecastRecord(**payload)
        db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec