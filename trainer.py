import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf

# ============================================================
# 1️⃣ Fetch Bitcoin data from multiple free APIs (no key needed)
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
        return df
    except Exception:
        pass

    # --- Try Yahoo Finance ---
    try:
        btc = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
        df = btc.reset_index()[["Date", "Close"]]
        return df
    except Exception as e:
        raise RuntimeError("❌ All data sources failed.") from e

# ============================================================
# 2️⃣ Prepare data
# ============================================================
def prepare_data(df, lookback=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[["Close"]])

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# ============================================================
# 3️⃣ Build model
# ============================================================
def build_model(lookback):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ============================================================
# 4️⃣ Train and save
# ============================================================
df = get_bitcoin_data()
X, y, scaler = prepare_data(df)

model = build_model(lookback=10)
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Save model and scaler
model.save("btc_model.keras")
np.save("btc_scaler.npy", scaler.data_min_.reshape(1, -1))
np.save("btc_scaler_max.npy", scaler.data_max_.reshape(1, -1))

print("✅ Training complete! Model and scaler saved successfully.")
