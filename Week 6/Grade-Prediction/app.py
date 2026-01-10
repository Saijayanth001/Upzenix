from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/grade_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        attendance = float(request.form["attendance"])
        previous = float(request.form["previous"])

        prediction = round(
            model.predict([[hours, attendance, previous]])[0], 2
        )

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
