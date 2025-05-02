from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv("symptom_disease_dataset_5000.csv")
df["symptoms"] = df["symptoms"].apply(ast.literal_eval)
df["precautions"] = df["precautions"].apply(ast.literal_eval)
df["prescription"] = df["prescription"].apply(ast.literal_eval)

# Extract all unique symptoms
all_symptoms = sorted({symptom for symptoms in df["symptoms"] for symptom in symptoms})

# Train model and binarizer if not already saved
MODEL_PATH = "diagnosis_model.pkl"
BINARIZER_PATH = "symptom_binarizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(BINARIZER_PATH):
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["symptoms"])
    y = df["disease"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(mlb, BINARIZER_PATH)
else:
    model = joblib.load(MODEL_PATH)
    mlb = joblib.load(BINARIZER_PATH)
@app.route('/')
def homepage():
    return render_template('homepage.html')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route("/diagnose", methods=["POST"])
def diagnose():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", [])

        input_vector = mlb.transform([symptoms])
        prediction = model.predict(input_vector)[0]

        result = df[df["disease"] == prediction].iloc[0]
        response = {
            "disease": prediction,
            "precautions": result["precautions"],
            "prescription": result["prescription"]
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/print')
def print_page():
    return render_template('print.html')
@app.route('/history')
def history():
    return render_template('history.html')
@app.route("/generate_icd9", methods=["POST"])
def generate_icd9():
    try:
        data = request.json
        symptoms = data.get("symptoms", [])

        input_vector = mlb.transform([symptoms])
        prediction = model.predict(input_vector)[0]

        result = df[df["disease"] == prediction].iloc[0]
        response = {
            "disease": prediction,
            "precautions": result["precautions"],
            "prescription": result["prescription"]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
