from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load the trained model
model = load('random_forest_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract features from the form
        features = [
            float(request.form['age']),
            int(request.form['freq_no_purpose']),
            int(request.form['freq_distracted']),
            int(request.form['restless']),
            int(request.form['worry_level']),
            int(request.form['difficulty_concentrating']),
            int(request.form['compare_to_successful_people']),
            int(request.form['feelings_about_comparisons']),
            int(request.form['freq_seeking_validation']),
            int(request.form['freq_feeling_depressed']),
            int(request.form['interest_fluctuation']),
            int(request.form['sleep_issues'])
        ]

        # Predict the distractibility scale
        prediction = model.predict([features])[0]
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
