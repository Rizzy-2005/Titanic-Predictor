from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

#Load model and scaler saved from Colab
model = joblib.load("titanic_log_reg_model.joblib")
scaler = joblib.load("titanic_scaler.joblib")


@app.route("/")
def home():
    # Show the main form (index.html)
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Read values from the HTML form
    pclass = int(request.form["pclass"])
    sex = int(request.form["sex"])          # 0 = male, 1 = female
    age = float(request.form["age"])
    sibsp = int(request.form["sibsp"])
    parch = int(request.form["parch"])
    fare = float(request.form["fare"])
    print(pclass, sex, age, sibsp, parch, fare)

    # Create the feature array in the SAME order as training:
    # [Pclass, Sex, Age, SibSp, Parch, Fare]
    features = np.array([[pclass, sex, age, sibsp, parch, fare]])

    # Scale features using the same scaler as training
    features_scaled = scaler.transform(features)

    # Get predicted probability of survival (class 1)
    prob_survive = model.predict_proba(features_scaled)[0][1]
    prediction = model.predict(features_scaled)[0]
    print("hello",prob_survive)

    # Round probability for display
    prob_percent = round(prob_survive * 100, 2)

    # Map sex back to string for display
    sex_text = "Female" if sex == 1 else "Male"

    return render_template(
        "result.html",
        prediction=prediction,
        probability=prob_percent,
        pclass=pclass,
        sex=sex_text,
        age=age,
        sibsp=sibsp,
        parch=parch,
        fare=fare
    )


if __name__ == "__main__":
    app.run(debug=True)
