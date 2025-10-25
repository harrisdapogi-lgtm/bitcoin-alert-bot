import os
import numpy as np
import pandas as pd
import requests
import subprocess
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# Configuration
# ============================================================
MODEL_FILE = "btc_model.keras"
SCALER_MIN = "btc_scaler.npy"
SCALER_MAX = "btc_scaler_max.npy"
PREDICTION_FILE = "predictions.csv"
LOOKBACK = 10
PREDICT_DAYS = 5

# ============================================================
# 1️⃣ Fetch Bitcoin data (Coindesk → CoinGecko → Yahoo Finance)
# ============================================================
def get_bitcoin_data():
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=1000)

    # --- Try Coindesk ---
    try:
        url = f"https://api.coindesk.com/v1/bpi/historical/close.json?start={start_date}&end={end_date}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()["bpi"]
        df = pd.DataFrame(list(data.items()), columns=["Date", "Close"])
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        print("✅ Using Coindesk data")
        return df
    except Exception:
        pass

    # --- Try CoinGecko ---
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1000"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()["prices"]
        df = pd.DataFrame(data, columns=["Timestamp", "Close"])
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df = df[["Date", "Close"]].sort_values("Date")
        print("✅ Using CoinGecko data")
        return df
    except Exception:
        pass

    # --- Try Yahoo Finance ---
    try:
        btc = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
        df = btc.reset_index()[["Date", "Close"]]
        print("✅ Using Yahoo Finance data")
        return df
    except Exception as e:
        raise RuntimeError("❌ All data sources failed.") from e


# ============================================================
# 2️⃣ Load model and scalers
# ============================================================
if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_MIN) and os.path.exists(SCALER_MAX)):
    raise FileNotFoundError("❌ Model or scaler files missing. Please retrain first.")

model = load_model(MODEL_FILE)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.data_min_ = np.load(SCALER_MIN)
scaler.data_max_ = np.load(SCALER_MAX)
scaler.scale_ = 1 / (scaler.data_max_ - scaler.data_min_)
scaler.min_ = -scaler.data_min_ * scaler.scale_

# ============================================================
# 3️⃣ Prepare latest data and make 5-day predictions
# ============================================================
df = get_bitcoin_data()
last_close = df["Close"].values[-LOOKBACK:]
scaled_last = scaler.transform(last_close.reshape(-1, 1)).reshape(1, LOOKBACK, 1)

predictions = []
for i in range(PREDICT_DAYS):
    pred_scaled = model.predict(scaled_last, verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0, 0]
    predictions.append(pred_price)
    scaled_last = np.append(scaled_last[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)

future_dates = [df["Date"].iloc[-1] + timedelta(days=i + 1) for i in range(PREDICT_DAYS)]
pred_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": predictions})

# ============================================================
# 4️⃣ Compare with previous predictions (if any)
# ============================================================
if os.path.exists(PREDICTION_FILE):
    old_df = pd.read_csv(PREDICTION_FILE)
    merged = pd.merge(old_df, df[["Date", "Close"]], on="Date", how="left")
    merged.rename(columns={"Close": "Actual_Close"}, inplace=True)
    merged["Error_%"] = np.abs(merged["Predicted_Close"] - merged["Actual_Close"]) / merged["Actual_Close"] * 100
    print("\n📊 Comparison of previous predictions:")
    print(merged.tail(10))
else:
    merged = pd.DataFrame()

# ============================================================
# 5️⃣ Save predictions to CSV
# ============================================================
pred_df.to_csv(PREDICTION_FILE, index=False)
print("\n✅ New predictions saved to predictions.csv")
print(pred_df)

# ============================================================
# 6️⃣ Auto-commit results back to GitHub
# ============================================================
try:
    subprocess.run(["git", "config", "--global", "user.name", "github-actions"], check=True)
    subprocess.run(["git", "config", "--global", "user.email", "actions@github.com"], check=True)
    subprocess.run(["git", "add", PREDICTION_FILE], check=True)
    subprocess.run(["git", "commit", "-m", f"🕒 Auto update predictions on {datetime.now().strftime('%Y-%m-%d %H:%M')}"], check=True)
    subprocess.run(["git", "push"], check=True)
    print("✅ Auto-commit pushed successfully!")
except subprocess.CalledProcessError:
    print("⚠️ No changes to commit or push.")
