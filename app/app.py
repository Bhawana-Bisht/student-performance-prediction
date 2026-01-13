from flask import Flask, render_template, request, url_for
import joblib
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

app = Flask(__name__)

# Ensure the static folder exists for charts
if not os.path.exists("app/static/charts"):
    os.makedirs("app/static/charts")

# Load model and scaler
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    processing_time = None
    chart_url = None

    if request.method == "POST":
        start = time.time()
        try:
            # Collect all 15 features (numeric + categorical)
            features = [
                float(request.form.get("study_hours", 0)),
                float(request.form.get("attendance", 0)),
                float(request.form.get("previous_score", 0)),
                float(request.form.get("sleep_hours", 0)),
                float(request.form.get("internet_access", 0)),
                float(request.form.get("extra_classes", 0)),
                float(request.form.get("parent_education", 1)),
                float(request.form.get("family_income", 1)),
                float(request.form.get("screen_time", 0)),
                float(request.form.get("health_status", 1)),
                float(request.form.get("motivation_level", 1)),
                float(request.form.get("school_support", 0)),
                float(request.form.get("travel_time", 1)),
                float(request.form.get("failures", 0)),
                float(request.form.get("participation", 1)),
            ]

            data = np.array([features])
            data_scaled = scaler.transform(data)

            # Make prediction
            prediction = round(model.predict(data_scaled)[0], 2)
            processing_time = round(time.time() - start, 3)

            # Generate unique chart filename using timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            chart_filename = f"charts/chart_{timestamp}.png"
            chart_path = os.path.join("app/static", chart_filename)

            # Create chart
            plt.figure(figsize=(6,3))
            plt.bar(["Predicted Score"], [prediction], color="#2563eb")
            plt.ylim(0, 100)
            plt.ylabel("Score")
            plt.title("Student Performance Prediction")
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()

            # URL to display chart in HTML
            chart_url = url_for('static', filename=chart_filename)

        except Exception as e:
            prediction = None
            chart_url = None
            print("Error:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        processing_time=processing_time,
        chart_path=chart_url
    )

if __name__ == "__main__":
    app.run(debug=True)
