import csv

import pytest

from app.scenario_transactions import (
    INPUT_COLUMNS,
    TRANSACTION_COLUMNS,
    append_scenario_transaction,
)


@pytest.fixture
def scenario_data():
    inputs = {column: 1 for column in INPUT_COLUMNS}
    inputs["date"] = "2026-01-01"
    result = {
        "prediction": 1200,
        "deviation": 20.0,
        "status": "high",
        "source": "local_model",
        "actual_arrivals": None,
        "actual_ape": None,
    }
    return inputs, result


def test_append_scenario_transaction_persists_inputs_and_outputs(
    tmp_path, scenario_data
):
    path = tmp_path / "scenario_transactions.csv"
    inputs, result = scenario_data

    transaction_id = append_scenario_transaction(inputs, result, path)
    append_scenario_transaction(inputs, result, path)

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert reader.fieldnames == list(TRANSACTION_COLUMNS)
    assert len(rows) == 2
    assert rows[0]["transaction_id"] == transaction_id
    assert rows[0]["date"] == inputs["date"]
    assert rows[0]["predicted_arrivals"] == str(result["prediction"])
    assert rows[0]["deviation_pct"] == str(result["deviation"])
    assert rows[0]["prediction_source"] == result["source"]
    assert rows[0]["actual_arrivals"] == ""


def test_append_scenario_transaction_rejects_existing_schema_mismatch(
    tmp_path, scenario_data
):
    path = tmp_path / "scenario_transactions.csv"
    path.write_text("unexpected,columns\n", encoding="utf-8")
    inputs, result = scenario_data

    with pytest.raises(ValueError, match="unexpected schema"):
        append_scenario_transaction(inputs, result, path)
