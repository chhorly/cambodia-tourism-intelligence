from datetime import datetime, timedelta
import os
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from sqlalchemy import create_engine

# Database Connection URI (Docker Postgres)
DB_URI = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/tourism_db"
)

default_args = {
    "owner": "RY_Chhorly",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def extract_and_transform():
    """Extract raw monthly figures, compute lag metrics, and output staged records."""
    data_dir = Path("/opt/airflow/data") if Path("/opt/airflow/data").exists() else Path("data")
    source_file = data_dir / "Cambodia_Tourism_Enriched.csv"
    
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found at {source_file}")

    df = pd.read_csv(source_file, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    
    # Feature lag construction
    target = "International_Tourist_Arrivals"
    df["Lag_1"] = df[target].shift(1)
    df["Lag_3"] = df[target].shift(3)
    df["Lag_12"] = df[target].shift(12)
    df["Rolling_3_Mean"] = df[target].shift(1).rolling(3).mean()
    df["Rolling_12_Mean"] = df[target].shift(1).rolling(12).mean()
    
    staged_path = data_dir / "staged_transformed_arrivals.csv"
    df.to_csv(staged_path, index=False)
    print(f"✅ Staged feature set constructed: {staged_path}")

def load_to_postgres():
    """Load transformed records into containerized PostgreSQL."""
    data_dir = Path("/opt/airflow/data") if Path("/opt/airflow/data").exists() else Path("data")
    forecast_file = data_dir / "Cambodia_Tourism_Forecast_EWS.csv"
    
    if not forecast_file.exists():
        raise FileNotFoundError(f"Forecast file not found at {forecast_file}")

    df = pd.read_csv(forecast_file)
    df.columns = [c.strip().replace(" ", "_").replace("%", "pct").lower() for c in df.columns]
    
    engine = create_engine(DB_URI)
    df.to_sql("forecasts_ews_airflow", engine, if_exists="replace", index=False)
    print("✅ Successfully ingested batch into PostgreSQL table: forecasts_ews_airflow")

with DAG(
    "cambodia_tourism_etl_pipeline",
    default_args=default_args,
    description="Monthly automated ingestion, feature calculation, and DB sync",
    schedule_interval="@monthly",
    catchup=False,
) as dag:

    task_extract_transform = PythonOperator(
        task_id="extract_and_transform_features",
        python_callable=extract_and_transform,
    )

    task_load_db = PythonOperator(
        task_id="load_forecasts_to_postgres",
        python_callable=load_to_postgres,
    )

    task_extract_transform >> task_load_db