import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import pytz
import smtplib
from email.mime.text import MIMEText

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "btc_model.h5"
SCALER_FILE = "btc_scaler.npy"

EMAIL_ADDRESS = os.environ.get("harrisdapogi@gmail.com") or "your_email@gmail.com"
EMAIL_PASSWORD = os.environ.get("cfpf dipr dazt vdda") or "your_app_password"
RECEIVER_EMAIL = os.environ.get("harrisdapogi@gmail.com") or "receiver_email@gmail.com"

# ============================================================
# INITIALIZATION
# ============================================================

app = Flask(__name__)

# Load model & scaler
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    model = load_model(MODEL_FILE)
    scaler_data = np.load(SCALER_FILE, allow_pickle=True)
    # Handle tuple structure from save
    if isinstance(scaler_data, np.ndarray) and len(scaler_data) == 2:
        min_, scale_ = scaler_data
    else:
        min_, scale_ = scaler_data.item().get("min_"), scaler_data.item().get("scale_")
    print("✅ Model & scaler loaded successfully.")
else:
    raise FileNotFoundError("❌ Model or scaler not found in your Render project folder.")

# ============================================================
# EMAIL FUNCTION
# ============================================================

def send_email(subject, body):
    """Send email via Gmail SMTP."""
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = RECEIVER_EMAIL

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("📧 Email sent successfully.")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# ============================================================
# DAILY EMAIL TASK
# ============================================================

def send_daily_email():
    """Sends a Bitcoin summary email daily."""
    subject = "Daily Bitcoin Alert 📊"
    body = f"Good morning! Here's your Bitcoin update for {datetime.now(pytz.timezone('Asia/Manila')).strftime('%Y-%m-%d %H:%M')}."
    send_email(subject, body)

# ============================================================
# FLASK ROUTE
# ============================================================

@app.route('/')
def index():
    return "🚀 Bitcoin Alert Bot is running on Render!"

# ============================================================
# SCHEDULER CONFIG
# ============================================================

scheduler = BackgroundScheduler(timezone='Asia/Manila')

# Schedule daily email at 8 AM Philippine time
scheduler.add_job(
    func=send_daily_email,
    trigger='cron',
    hour=8,
    minute=0,
    timezone='Asia/Manila'
)

scheduler.start()

# ============================================================
# RUN APP
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
