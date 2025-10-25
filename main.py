import numpy as np
import pandas as pd
import requests
import json
import datetime
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import os

# ============================================================
# CONFIG
# ============================================================
MODEL_JSON = "btc_model.json"
MODEL_WEIGHTS = "btc_model_weights.h5"
SCALER_FILE = "btc_scaler.npy"
PREDICTIONS_CSV = "predictions.csv"
DAYS_TO_PREDICT = 5

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================
if not (os.path.exists(MODEL_JSON) and os.path.exists(MODEL_WEIGHTS) and os.path.exists(SCALER_FILE)):
    raise FileNotFoundError("❌ Model or scaler files missing in repo.")

# ✅ FIXED: Read model JSON as text string, not as dict
with open(MODEL_JSON, "r") as json_file:
    model_json_str = json_file.read()

model = model_from_json(model_json_str)
model.load_weights(MODEL_WEIGHTS)

# Load scaler
scaler_data = np.load(SCALER_FILE, allow_pickle=True)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.min_, scaler.scale_ = scaler_data

print("✅ Model & scaler loaded successfully.")

# ============================================================
# FETCH LATEST DATA (Bitcoin)
# ============================================================
def fetch_btc_data():
    url = "https://api.coindesk.com/v1/bpi/historical/close.json"
    end = datetime.date.today()
    start = end - datetime.timedelta(days=60)
    response = requests.get(f"{url}?start={start}&end={end}")
    data = response.json()["bpi"]
    df = pd.DataFrame(list(data.items()), columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    return df

df = fetch_btc_data()
print(f"📈 Loaded {len(df)} days of BTC data.")

# ============================================================
# PREPARE INPUT
# ============================================================
data_scaled = scaler.fit_transform(df["close"].values.reshape(-1, 1))
lookback = 10

X_input = data_scaled[-lookback:].reshape(1, lookback, 1)
predictions = []

for _ in range(DAYS_TO_PREDICT):
    pred = model.predict(X_input)
    predictions.append(pred[0, 0])
    X_input = np.append(X_input[:, 1:, :], [[pred]], axis=1)

pred_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# ============================================================
# SAVE RESULTS
# ============================================================
pred_dates = [df["date"].iloc[-1] + datetime.timedelta(days=i+1) for i in range(DAYS_TO_PREDICT)]
pred_df = pd.DataFrame({"date": pred_dates, "predicted": pred_prices})

# Try fetching actuals for comparison
try:
    actual_df = fetch_btc_data()
    merged = pd.merge(pred_df, actual_df, on="date", how="left", suffixes=("_pred", "_actual"))
    merged["error_%"] = abs(merged["predicted"] - merged["close"]) / merged["close"] * 100
    merged.rename(columns={"close": "actual"}, inplace=True)
except Exception:
    merged = pred_df
    merged["actual"] = np.nan
    merged["error_%"] = np.nan

# Append to CSV
if os.path.exists(PREDICTIONS_CSV):
    old = pd.read_csv(PREDICTIONS_CSV)
    final = pd.concat([old, merged]).drop_duplicates(subset=["date"], keep="last")
else:
    final = merged

final.to_csv(PREDICTIONS_CSV, index=False)
print("💾 Saved predictions to predictions.csv")

# ============================================================
# DISPLAY PREVIEW
# ============================================================
print(final.tail(10))
