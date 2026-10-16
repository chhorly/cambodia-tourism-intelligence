import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_TRANSACTIONS_PATH = PROJECT_ROOT / "data" / "scenario_transactions.csv"
INPUT_COLUMNS = (
    "date",
    "seasonal_baseline",
    "Lag_1",
    "Lag_3",
    "Lag_12",
    "Rolling_3_Mean",
    "Rolling_12_Mean",
    "Average_Temperature",
    "Rainfall",
    "Cambodia_Holiday_Days",
    "COVID_Indicator",
    "Month_Number",
    "Total_Origin_Holidays",
    "China_Holidays",
)
OUTPUT_COLUMNS = (
    "predicted_arrivals",
    "deviation_pct",
    "alert_status",
    "prediction_source",
    "actual_arrivals",
    "actual_absolute_percentage_error",
)
TRANSACTION_COLUMNS = (
    "transaction_id",
    "recorded_at_utc",
    *INPUT_COLUMNS,
    *OUTPUT_COLUMNS,
)


def append_scenario_transaction(
    inputs: Mapping[str, object],
    result: Mapping[str, object],
    path: Path = SCENARIO_TRANSACTIONS_PATH,
) -> str:
    """Append one completed scenario's inputs and results to the transaction CSV."""
    missing_inputs = set(INPUT_COLUMNS).difference(inputs)
    missing_results = {"prediction", "deviation", "status", "source"}.difference(result)
    if missing_inputs or missing_results:
        missing = sorted(missing_inputs | missing_results)
        raise ValueError(f"Scenario transaction is missing fields: {', '.join(missing)}")

    record_id = str(uuid4())
    row = {
        "transaction_id": record_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        **{column: inputs[column] for column in INPUT_COLUMNS},
        "predicted_arrivals": result["prediction"],
        "deviation_pct": result["deviation"],
        "alert_status": result["status"],
        "prediction_source": result["source"],
        "actual_arrivals": result.get("actual_arrivals"),
        "actual_absolute_percentage_error": result.get("actual_ape"),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            existing_columns = next(csv.reader(csv_file), [])
        if existing_columns != list(TRANSACTION_COLUMNS):
            raise ValueError(
                f"Scenario transaction CSV has an unexpected schema: {path}"
            )

    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRANSACTION_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return record_id
