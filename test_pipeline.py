import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_system():
    print("=" * 60)
    print("      PHASE 8.9: AUTOMATED SYSTEM PIPELINE TEST       ")
    print("=" * 60)

    # 1. Test /health
    print("\n[Test 1] Checking API Health...")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print("✅ Health Check Passed:", r.json())

    # 2. Test /api/v1/forecasts/latest
    print("\n[Test 2] Fetching Stored Forecasts & Alerts...")
    r = requests.get(f"{BASE_URL}/api/v1/forecasts/latest")
    assert r.status_code == 200, f"Fetch forecasts failed: {r.text}"
    data = r.json()
    print(f"✅ Retrieved {len(data)} forecast records.")
    print("Sample record:", data[0] if data else "Empty")

    # 3. Test /api/v1/predict
    print("\n[Test 3] Running Real-Time Inference with EWS...")
    payload = {
        "date": "2026-07-01",
        "seasonal_baseline": 352050.0,
        "Lag_1": 212860.0,
        "Lag_3": 292399.0,
        "Lag_12": 347492.0,
        "Rolling_3_Mean": 245913.0,
        "Rolling_12_Mean": 310000.0,
        "Average_Temperature": 28.5,
        "Rainfall": 175.0,
        "Cambodia_Holiday_Days": 0,
        "COVID_Indicator": 0,
        "Month_Number": 7,
        "Total_Origin_Holidays": 5,
        "China_Holidays": 0
    }
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=payload)
    assert r.status_code == 200, f"Prediction failed: {r.text}"
    res = r.json()
    print("✅ Inference Successful:")
    print(json.dumps(res, indent=2))
    print("\n🎉 ALL PIPELINE TESTS PASSED!")

if __name__ == "__main__":
    test_system()