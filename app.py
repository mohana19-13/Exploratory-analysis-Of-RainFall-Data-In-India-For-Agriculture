
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)

# load trained model
with open("rainfall.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    location = int(request.form["location"])
    mintemp = float(request.form["mintemp"])
    maxtemp = float(request.form["maxtemp"])
    humidity = float(request.form["humidity"])
    pressure = float(request.form["pressure"])
    windspeed = float(request.form["windspeed"])
    cloud = float(request.form["cloud"])
    rainfall = float(request.form["rainfall"])
    month= int(request.form["month"])
    day= int(request.form["day"])
    year= int(request.form["year"])
    season= int(request.form["season"])


    # ⚠️ ONLY 7 FEATURES (model already trained)
    input_array = np.array([[location,
    mintemp,
    maxtemp,
    humidity,
    pressure,
    windspeed,
    cloud,
    rainfall,
    month,
    day,
    year,
    season
]])
    # probability of rain (class 1)
    import pandas as pd
    input_df = pd.DataFrame(input_array, columns=["location", "mintemp", "maxtemp", "humidity", "pressure", "windspeed", "cloud", "rainfall", "month", "day", "year", "season"])
    prob = model.predict_proba(input_array)[0][1]
    print("Input features:", input_array)
    print("Probability of rain:", prob)

    if prob >= 0.6:
        return redirect(url_for("chance"))
    else:
        return redirect(url_for("nochance"))


@app.route("/chance")
def chance():
    return render_template("chance.html")


@app.route("/nochance")
def nochance():
    return render_template("nochance.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)