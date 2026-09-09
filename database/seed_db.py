import os
from pathlib import Path
import pandas as pd
from database.connection import Base, engine, SessionLocal
from database.crud import ForecastRecord

def seed():
    # 1. Initialize tables in Docker Postgres
    print("⏳ Creating database tables in PostgreSQL...")
    Base.metadata.create_all(bind=engine)
    
    # 2. Locate CSV
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "Cambodia_Tourism_Forecast_EWS.csv"
    if not csv_path.exists():
        print(f"❌ Cannot find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    db = SessionLocal()
    try:
        inserted = 0
        for _, row in df.iterrows():
            d_val = pd.to_datetime(row["Date"]).date()
            existing = db.query(ForecastRecord).filter(ForecastRecord.date_id == d_val).first()
            if not existing:
                record = ForecastRecord(
                    date_id=d_val,
                    model_version="Tuned_LightGBM_v1",
                    actual_arrivals=int(row["Actual_Arrivals"]) if pd.notna(row.get("Actual_Arrivals")) else None,
                    predicted_arrivals=int(round(float(row["Tuned_Forecast"]))),
                    seasonal_baseline=int(round(float(row["Seasonal_Baseline"]))),
                    deviation_pct=float(row["Deviation_%"]),
                    alert_status=str(row["Alert_Status"])
                )
                db.add(record)
                inserted += 1
        db.commit()
        print(f"✅ Successfully seeded {inserted} monthly forecast records into PostgreSQL Docker.")
    except Exception as e:
        db.rollback()
        print(f"❌ Seeding failed: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed()