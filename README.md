# Cambodia Tourism Intelligence

A bilingual (Khmer/English) tourism analytics and demand early-warning project for Cambodia. It combines historical tourism data, a LightGBM forecasting model, a FastAPI prediction service, and a Streamlit dashboard for reviewing forecasts, demand signals, and scenario results.

## What the project does

- Trains a LightGBM regression model to estimate monthly international tourist arrivals.
- Builds lag, rolling-average, holiday, weather, and seasonal features from historical data.
- Evaluates the model on a chronological holdout rather than a random split.
- Serves predictions and forecast records through a REST API.
- Displays overview KPIs, forecast comparisons, alert levels, and operational guidance in a dashboard.
- Lets users test scenario assumptions and appends submitted scenario inputs and results to a local CSV file.
- Includes an optional Airflow and PostgreSQL stack for scheduled ETL work.

Alert levels compare a forecast with its seasonal baseline:

- **High demand:** more than 15% above baseline.
- **Normal range:** between -15% and +15% of baseline.
- **Low demand:** more than 15% below baseline.

These are signals for review, not guarantees of actual visitor volume.

## Project components

```text
Historical CSV
   |                    Forecast/EWS CSV
   v                           |
Model training                 +----> Streamlit dashboard
   |                           |        | Overview, Forecasts,
LightGBM artifact              |        | Alerts, Scenarios, Technical
   |                           |        |
   +----> FastAPI prediction service    +----> scenario_transactions.csv
                |
          PostgreSQL (optional)

Airflow DAG (optional) --> staged CSV --> PostgreSQL ETL table
```

### Dashboard modules

The sidebar provides these sections:

1. **Overview** - forecast KPIs, actual-versus-forecast trends, and items needing attention.
2. **Forecasts** - monthly forecast performance, forecast errors, annual comparisons, and downloads.
3. **Alerts & Actions** - deviations from the seasonal baseline, the +/-15% thresholds, and suggested operational responses.
4. **Scenario Explorer** - adjust a target month’s assumptions and save the resulting prediction to transaction history.
5. **Technical** - data, model, validation, and serving details.

The dashboard uses the bundled CSV files and model artifact. If the API is running locally, it uses the API for scenario predictions; otherwise it can run supported scenarios with the bundled model.

## Repository layout

```text
app/
  dashboard.py                 Streamlit dashboard
  main.py                      FastAPI application
  scenario_transactions.py      Append-only scenario CSV writer and schema
database/
  connection.py                PostgreSQL connection and sessions
  crud.py                      Forecast record model and persistence
  seed_db.py                   Seed forecast records into PostgreSQL
dags/
  cambodia_tourism_dag.py       Monthly Airflow ETL DAG
data/
  Cambodia_Tourism_Enriched.csv Historical training data
  Cambodia_Tourism_Forecast_EWS.csv Forecast, baseline, and alert data
models/
  cambodia_tourism_lgbm.pkl     Serialized LightGBM model artifact
tests/
  test_scenario_transactions.py Scenario CSV tests
model_training.py               Feature preparation, evaluation, and training
requirements.txt                Python dependencies
docker-compose.airflow.yml       Optional Airflow/PostgreSQL services
test_pipeline.py                Manual API integration smoke test
```

Runtime scenario history (`data/scenario_transactions.csv`) and generated ETL output (`data/staged_transformed_arrivals.csv`) are local data files and are not intended for source control.

## Data and model

The dashboard and training script expect these files:

- `data/Cambodia_Tourism_Enriched.csv` - monthly historical arrivals and predictors, including temperature, rainfall, holidays, COVID indicator, lags, and rolling averages.
- `data/Cambodia_Tourism_Forecast_EWS.csv` - monthly actuals, tuned forecast, seasonal baseline, deviation, alert status, and error values.
- `models/cambodia_tourism_lgbm.pkl` - serialized model artifact and feature metadata.

The training pipeline prepares features in date order, trains on `log1p`-transformed arrivals, and converts predictions back to visitor counts with `expm1`. It uses recency-based sample weights and reduces the weight of the 2020-2021 observations. Evaluation reserves the latest 12 months as a chronological holdout and uses `TimeSeriesSplit` for parameter search.

Keep the CSV column names and model feature contract consistent when replacing the bundled data or model. The dashboard can resolve some duplicate downloaded filenames, but the canonical filenames above are recommended.

The bundled forecast CSV is a historical validation sample, not a live feed. The dashboard flags an expired forecast window; refresh the source data before using it to make current operational decisions.

## Requirements

- Python 3.10 or newer.
- The files listed in the **Data and model** section.
- PostgreSQL only if you want database persistence.
- Docker Desktop with Docker Compose only if you want to run the optional Airflow stack.

## Install and run on Windows

From the repository root, in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents environment activation, allow it for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Start the dashboard:

```powershell
streamlit run app/dashboard.py
```

Streamlit prints the local URL (normally `http://localhost:8501`). The dashboard reads the bundled data and model from the repository, so the working directory should be the repository root.

### Start the API (optional)

Open a second terminal, activate the same virtual environment, and run:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Useful endpoints:

- Health check: `http://127.0.0.1:8000/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`
- Latest forecast records: `GET http://127.0.0.1:8000/api/v1/forecasts/latest`
- Run a prediction: `POST http://127.0.0.1:8000/api/v1/predict`

The dashboard currently checks the API at `http://127.0.0.1:8000`. If the API is stopped, forecasts still display from the local CSV data and supported scenarios can use the local model.

### PostgreSQL persistence (optional)

The API and database layer use the `DATABASE_URL` environment variable for PostgreSQL. Set it in the terminal before starting the API, for example:

```powershell
$env:DATABASE_URL = "postgresql+psycopg2://<user>:<password>@<host>:5432/<database>"
```

Replace the placeholders with your own PostgreSQL connection details; do not commit credentials. To seed the database from the bundled forecast CSV:

```powershell
python -m database.seed_db
```

The API can serve forecast data from the CSV when database records cannot be read. Prediction persistence is attempted when PostgreSQL is configured.

## Retrain the model

From the repository root with the dependencies installed:

```powershell
python model_training.py
```

This reads `data/Cambodia_Tourism_Enriched.csv`, evaluates a chronological holdout, performs a time-series grid search, and writes the artifact to `models/cambodia_tourism_lgbm.pkl`. Training can take significantly longer than starting the dashboard and will replace the existing artifact.

## Scenario transaction history

When a scenario is submitted, its input assumptions and prediction result are appended to `data/scenario_transactions.csv`. Each row includes a unique transaction ID and UTC timestamp; actual arrivals and absolute percentage error are included when available. The dashboard lets users choose how many rows to display and download the CSV. The transaction file remains local and is ignored by Git.

## Optional Airflow ETL stack

The Compose file runs Airflow with a dedicated PostgreSQL service for local ETL development. It is separate from the dashboard’s FastAPI/PostgreSQL setup.

```powershell
docker compose -f docker-compose.airflow.yml up airflow-init
docker compose -f docker-compose.airflow.yml up -d postgres-airflow airflow-webserver airflow-scheduler
```

The Airflow UI is normally available at `http://localhost:8080`. The `cambodia_tourism_etl_pipeline` DAG is scheduled monthly. It writes lag/rolling-feature data to `data/staged_transformed_arrivals.csv` and loads the forecast CSV into the `forecasts_ews_airflow` table.

The supplied Compose configuration is for local development, not production. Review and replace its default credentials, database settings, and security configuration before exposing it beyond a trusted local machine.

## Tests

Run the focused automated tests from the repository root:

```powershell
python -m pytest tests/test_scenario_transactions.py -q
```

`test_pipeline.py` is a manual API smoke test. Start the API first, then run:

```powershell
python test_pipeline.py
```

It checks the health endpoint, retrieves forecasts, and submits a sample prediction.

## Troubleshooting

- **Dashboard reports missing data or model:** confirm the required CSV files and `.pkl` artifact exist at the paths listed above.
- **API is unavailable in the dashboard:** start Uvicorn on port 8000; the dashboard still supports local CSV and model operation.
- **Database connection errors:** verify `DATABASE_URL`, that PostgreSQL is running, and that the selected database exists. The dashboard does not require PostgreSQL for local CSV-based viewing.
- **Scenario transaction file has an unexpected schema:** preserve the header generated by the app; the writer intentionally rejects a mismatched existing CSV rather than silently mixing formats.
- **Airflow does not start:** run the initialization command first and check that Docker Desktop is running and the PostgreSQL service has become healthy.
