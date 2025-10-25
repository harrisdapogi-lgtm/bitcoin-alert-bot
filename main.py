import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ============================================================
# Config
# ============================================================
MODEL_FILE = "btc_model.keras"
SCALER_FILE = "btc_scaler.npy"
RESULT_FILE = "prediction_log.csv"
DAYS_TO_PREDICT = 5

# ============================================================
# Load Model and Scaler
# ============================================================
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("❌ Model file not found.")
if not os.path.exists(SCALER_FILE):
    raise FileNotFoundError("❌ Scaler file not found.")

model = load_model(MODEL_FILE, compile=False)
scaler = StandardScaler()
scaler.mean_, scaler.scale_ = np.load(SCALER_FILE, allow_pickle=True)
print("✅ Model and scaler loaded.")

# ============================================================
# Fetch Latest BTC Data
# ============================================================
today = datetime.utcnow()
start_date = today - timedelta(days=60)
btc = yf.download("BTC-USD", start=start_date, end=today)
if btc.empty:
    raise ValueError("⚠️ No BTC data downloaded.")

btc = btc[["Close"]].reset_index()
btc["Date"] = btc["Date"].dt.date

# ============================================================
# Prepare Input
# ============================================================
data_scaled = scaler.transform(btc[["Close"]])
X_input = data_scaled[-10:].reshape(1, 10, 1)  # last 10 days

# ============================================================
# Make 5-Day Predictions
# ============================================================
preds = []
for i in range(DAYS_TO_PREDICT):
    pred = model.predict(X_input)[0, 0]
    preds.append(pred)
    new_input = np.append(X_input.flatten()[1:], pred).reshape(1, 10, 1)
    X_input = new_input

# Inverse scale predictions
preds_rescaled = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

future_dates = [(today + timedelta(days=i+1)).date() for i in range(DAYS_TO_PREDICT)]
pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": preds_rescaled
})

# ============================================================
# Merge with Actual if available
# ============================================================
if os.path.exists(RESULT_FILE):
    old_df = pd.read_csv(RESULT_FILE)
else:
    old_df = pd.DataFrame(columns=["Date", "Actual_Close", "Predicted_Close"])

# Update actuals (for yesterday, since today’s price might not be final)
if len(old_df) > 0:
    last_logged = old_df.tail(1)["Date"].values[0]
    actual_row = btc[btc["Date"] == pd.to_datetime(last_logged).date()]
    if not actual_row.empty:
        old_df.loc[old_df["Date"] == last_logged, "Actual_Close"] = actual_row["Close"].values[0]

# Combine new predictions
merged_df = pd.concat([old_df, pred_df], ignore_index=True)
merged_df.to_csv(RESULT_FILE, index=False)

print("📈 Predictions saved to prediction_log.csv")
print(merged_df.tail(10))
