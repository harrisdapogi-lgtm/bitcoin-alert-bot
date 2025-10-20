import os
import smtplib
import time
import requests
import numpy as np
import threading
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from email.mime.text import MIMEText
from flask import Flask

# ========== CONFIG ==========
MODEL_FILE = "btc_predictor.h5"   # or .keras if you resaved
SCALER_FILE = "btc_scaler.npy"

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

CHECK_INTERVAL = 3600   # hourly price check
ALERT_THRESHOLD = -3    # % drop for alert
DAILY_REPORT_HOUR = 0   # UTC hour for daily summary (0 = midnight)
API_URL = "https://api.coindesk.com/v1/bpi/currentprice/BTC.json"

app = Flask(__name__)

# ========== LOAD MODEL ==========
model = load_model(MODEL_FILE, compile=False)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.min_, scaler.scale_ = np.load(SCALER_FILE, allow_pickle=True)
print("✅ Model and scaler loaded successfully.")

# ========== HELPER FUNCTIONS ==========
def get_bitcoin_price():
    """Fetch current Bitcoin price (USD)."""
    data = requests.get(API_URL).json()
    return float(data["bpi"]["USD"]["rate"].replace(",", ""))

def send_email(subject, message):
    """Send an email alert via Gmail."""
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
    print(f"📧 Sent email: {subject}")

# ========== MAIN MONITOR ==========
def monitor_bitcoin():
    print("🔁 Starting Bitcoin alert loop...")
    last_price = get_bitcoin_price()
    last_report_date = None

    while True:
        try:
            current_price = get_bitcoin_price()
            percent_change = ((current_price - last_price) / last_price) * 100

            # --- Hourly check for alerts ---
            if percent_change <= ALERT_THRESHOLD:
                send_email(
                    "📉 Bitcoin Price Drop Alert!",
                    f"BTC dropped by {percent_change:.2f}%\nCurrent price: ${current_price:,.2f}"
                )

            print(f"[{datetime.utcnow()}] BTC: ${current_price:,.2f} | Δ {percent_change:.2f}%")

            last_price = current_price

            # --- Daily report at midnight UTC ---
            now = datetime.utcnow()
            if now.hour == DAILY_REPORT_HOUR and (not last_report_date or now.date() != last_report_date):
                yesterday_price = get_bitcoin_price()
                change_24h = ((current_price - yesterday_price) / yesterday_price) * 100
                direction = "📈 UP" if change_24h > 0 else "📉 DOWN"

                summary = (
                    f"Daily Bitcoin Summary ({now.date()} UTC)\n\n"
                    f"Price: ${current_price:,.2f}\n"
                    f"Change (24h): {change_24h:.2f}% {direction}\n"
                    f"Threshold Alert: {ALERT_THRESHOLD}%\n\n"
                    f"Bot running on Render ✅"
                )
                send_email("📊 Daily Bitcoin Summary", summary)
                last_report_date = now.date()

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print("⚠️ Error:", e)
            time.sleep(60)

@app.route("/")
def home():
    return "✅ Bitcoin Alert Bot with Daily Email Summary is running on Render.com"

if __name__ == "__main__":
    t = threading.Thread(target=monitor_bitcoin)
    t.start()
    app.run(host="0.0.0.0", port=10000)
