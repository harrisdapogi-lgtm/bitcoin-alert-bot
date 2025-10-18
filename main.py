import os
import smtplib
import time
import requests
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from email.mime.text import MIMEText
from flask import Flask

# ========== CONFIG ==========
MODEL_FILE = "btc_predictor.h5"
SCALER_FILE = "btc_scaler.npy"

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

CHECK_INTERVAL = 3600  # 1 hour
ALERT_THRESHOLD = -3   # % drop to trigger alert
API_URL = "https://api.coindesk.com/v1/bpi/currentprice/BTC.json"

app = Flask(__name__)

# ========== MODEL LOADING ==========
model = load_model(MODEL_FILE)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.min_, scaler.scale_ = np.load(SCALER_FILE, allow_pickle=True)
print("✅ Model and scaler loaded successfully.")

# ========== PRICE PREDICTOR ==========
def get_bitcoin_price():
    data = requests.get(API_URL).json()
    return float(data["bpi"]["USD"]["rate"].replace(",", ""))

def send_email(subject, message):
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
    print(f"📧 Alert sent: {subject}")

def check_price_loop():
    print("🔁 Starting Bitcoin alert loop...")
    last_price = get_bitcoin_price()

    while True:
        try:
            time.sleep(CHECK_INTERVAL)
            current_price = get_bitcoin_price()
            percent_change = ((current_price - last_price) / last_price) * 100

            print(f"BTC: {current_price:.2f} USD | Δ {percent_change:.2f}%")

            if percent_change <= ALERT_THRESHOLD:
                send_email("📉 Bitcoin Price Drop Alert!",
                           f"BTC dropped by {percent_change:.2f}%. Current price: ${current_price:.2f}")
            last_price = current_price
        except Exception as e:
            print("⚠️ Error:", e)
            time.sleep(60)

@app.route("/")
def home():
    return "✅ Bitcoin Alert Bot is running on Render.com"

if __name__ == "__main__":
    import threading
    t = threading.Thread(target=check_price_loop)
    t.start()
    app.run(host="0.0.0.0", port=10000)
