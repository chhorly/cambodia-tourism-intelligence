from datetime import date
from typing import Any
import os
import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field

# Database layer with resilient CSV fallback
try:
    from database.connection import get_db, engine, Base
    from database.crud import get_all_forecasts, save_or_update_forecast

    DATABASE_AVAILABLE = True
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"⚠️ Database connection warning: {e}")
except ModuleNotFoundError as e:
    DATABASE_AVAILABLE = False

    def get_db():
        yield None

    print(f"⚠️ Database modules unavailable; using CSV fallback: {e}")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cambodia_tourism_lgbm.pkl")

app = FastAPI(
    title="Cambodia Tourism Demand & Early Warning API",
    version="1.0.0",
    description="Production REST API connecting PostgreSQL, LightGBM inference, and EWS alerts."
)

# Load serialized champion model artifact
try:
    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict):
        model = artifact["model"]
        feature_names = artifact.get("features", None)
    else:
        model = artifact
        feature_names = [
            "Lag_1", "Lag_3", "Lag_12", "Rolling_3_Mean", "Rolling_12_Mean",
            "Average_Temperature", "Rainfall", "Cambodia_Holiday_Days",
            "COVID_Indicator", "Month_Number", "Total_Origin_Holidays", "China_Holidays"
        ]
    print("✅ Model artifact loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

class ForecastInput(BaseModel):
    date: date
    seasonal_baseline: float = Field(..., example=364436.0)
    Lag_1: float = Field(..., example=327239.0)
    Lag_3: float = Field(..., example=347492.0)
    Lag_12: float = Field(..., example=300000.0)
    Rolling_3_Mean: float = Field(..., example=337694.0)
    Rolling_12_Mean: float = Field(..., example=350000.0)
    Average_Temperature: float = Field(..., example=28.4)
    Rainfall: float = Field(..., example=180.2)
    Cambodia_Holiday_Days: int = Field(default=2, example=2)
    COVID_Indicator: int = Field(default=0, example=0)
    Month_Number: int = Field(..., ge=1, le=12, example=10)
    Total_Origin_Holidays: int = Field(default=11, example=11)
    China_Holidays: int = Field(default=1, example=1)

class ForecastResponse(BaseModel):
    date: date
    predicted_arrivals: int
    seasonal_baseline: int
    deviation_pct: float
    alert_status: str
    recommended_action: str

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database": "available" if DATABASE_AVAILABLE else "unavailable (CSV fallback)"
    }

@app.get("/api/v1/forecasts/latest", tags=["Early Warning System"])
def get_forecasts(db: Any = Depends(get_db)):
    """Fetches all forecasts from PostgreSQL, fallback to CSV if DB is empty."""
    try:
        if DATABASE_AVAILABLE:
            records = get_all_forecasts(db)
            if records:
                return [
                    {
                        "date_id": str(r.date_id),
                        "predicted_arrivals": r.predicted_arrivals,
                        "seasonal_baseline": r.seasonal_baseline,
                        "deviation_pct": r.deviation_pct,
                        "alert_status": r.alert_status
                    }
                    for r in records
                ]
    except Exception:
        pass

    csv_path = os.path.join(PROJECT_ROOT, "data", "Cambodia_Tourism_Forecast_EWS.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    raise HTTPException(status_code=404, detail="No forecast data found in database or CSV.")

@app.post("/api/v1/predict", response_model=ForecastResponse, tags=["Forecasting"])
def predict_and_store(payload: ForecastInput, db: Any = Depends(get_db)):
    """Runs LightGBM inference, evaluates EWS threshold, and persists to DB."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available.")

    input_data = payload.model_dump()
    pred_date = input_data.pop("date")
    baseline = input_data.pop("seasonal_baseline")

    input_df = pd.DataFrame([input_data])[feature_names]
    pred_val = float(model.predict(input_df)[0])

    dev_pct = ((pred_val - baseline) / baseline) * 100.0
    if dev_pct > 15.0:
        status = "High Demand Alert (🔴)"
        action = "Surge immigration personnel, inspect hotel inventory, and adjust shuttle schedules."
    elif dev_pct < -15.0:
        status = "Low Demand Warning (🟡)"
        action = "Initiate targeted promotional campaigns and airline ticket subsidies."
    else:
        status = "Normal Demand (🟢)"
        action = "Maintain standard seasonal operations and regular resource deployment."

    try:
        if DATABASE_AVAILABLE:
            save_or_update_forecast(db, {
                "date_id": pred_date,
                "model_version": "Tuned_LightGBM_v1",
                "predicted_arrivals": round(pred_val),
                "seasonal_baseline": round(baseline),
                "deviation_pct": round(dev_pct, 2),
                "alert_status": status
            })
    except Exception as e:
        print(f"Warning: Could not commit prediction to DB: {e}")

    return {
        "date": pred_date,
        "predicted_arrivals": round(pred_val),
        "seasonal_baseline": round(baseline),
        "deviation_pct": round(dev_pct, 2),
        "alert_status": status,
        "recommended_action": action
    }