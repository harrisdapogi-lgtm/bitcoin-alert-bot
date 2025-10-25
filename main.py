import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
import os

# ============================================================
# Load Model + Scaler
# ============================================================
MODEL_JSON = "btc_model.json"
MODEL_WEIGHTS = "btc_model_weights.h5"
SCALER_FILE = "btc_scaler.npy"
CSV_FILE = "predictions.csv"

if not all(os.path.exists(f) for f in [MODEL_JSON, MODEL_WEIGHTS, SCALER_FILE]):
    raise FileNotFoundError("❌ Model or scaler files missing in repo.")

with open(MODEL_JSON, "r") as f:
    model = model_from_json(f.read())
model.load_weights(MODEL_WEIGHTS)
print("✅ Model loaded successfully.")

scaler_params = np.load(SCALER_FILE, allow_pickle=True)
scaler = StandardScaler()
scaler.mean_, scaler.scale_ = scaler_params
print("✅ Scaler loaded successfully.")

# ============================================================
# Get Latest BTC Data
# ============================================================
df = yf.download("BTC-USD", period="60d", interval="1d")[["Close"]]
df = df.dropna()
scaled_data = scaler.transform(df.values)

# Use last 10 days as model input
X_input = scaled_data[-10:].reshape(1, 10, 1)

# ============================================================
# Predict Next 5 Days
# ============================================================
predictions_scaled = []
current_input = X_input.copy()

for _ in range(5):
    next_scaled = model.predict(current_input, verbose=0)[0, 0]
    predictions_scaled.append(next_scaled)
    current_input = np.append(current_input[:, 1:, :], [[[next_scaled]]], axis=1)

predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
dates = [datetime.now() + timedelta(days=i + 1) for i in range(5)]

# ============================================================
# Save & Compare
# ============================================================
new_data = pd.DataFrame({
    "date": dates,
    "predicted_close": predictions,
    "actual_close": [None] * 5
})

# Load old predictions if exist
if os.path.exists(CSV_FILE):
    old = pd.read_csv(CSV_FILE)
    combined = pd.concat([old, new_data]).drop_duplicates(subset=["date"], keep="last")
else:
    combined = new_data

# Update actuals where available
actual_data = df.reset_index()[["Date", "Close"]]
for i, row in combined.iterrows():
    match = actual_data.loc[actual_data["Date"] == pd.to_datetime(row["date"]).normalize()]
    if not match.empty:
        combined.at[i, "actual_close"] = match["Close"].values[0]

combined.to_csv(CSV_FILE, index=False)
print("✅ Predictions saved to predictions.csv")
