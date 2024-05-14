from flask import Flask, render_template, request
import pandas as pd
from joblib import load
import google.generativeai as genai
import markdown
from dotenv import load_dotenv
load_dotenv()

import os

app = Flask(__name__)

# Load the trained model
model = load('random_forest_model.joblib')

def get_personalized_suggestions(distraction_level, age, freq_no_purpose, freq_distracted, restless, worry_level, difficulty_concentrating, compare_to_successful_people, feelings_about_comparisons, freq_seeking_validation, freq_feeling_depressed, interest_fluctuation, sleep_issues):
    # Formulate a prompt based on the input data
    prompt = f"Based on the below data in the range (1-5) except age, give me the detailed assesment and personalised recommendations / suggestion. Distraction Level: {distraction_level}\nAge: {age}\nFrequency of No Purpose: {freq_no_purpose}\nFrequency of Being Distracted: {freq_distracted}\nRestlessness Level: {restless}\nWorry Level: {worry_level}\nDifficulty Concentrating: {difficulty_concentrating}\nComparison to Successful People: {compare_to_successful_people}\nFeelings About Comparisons: {feelings_about_comparisons}\nFrequency of Seeking Validation: {freq_seeking_validation}\nFrequency of Feeling Depressed: {freq_feeling_depressed}\nInterest Fluctuation: {interest_fluctuation}\nSleep Issues: {sleep_issues}\n"

    print(distraction_level, age, freq_no_purpose, freq_distracted, restless, worry_level, difficulty_concentrating, compare_to_successful_people, feelings_about_comparisons, freq_seeking_validation, freq_feeling_depressed, interest_fluctuation, sleep_issues)

    # Generate personalized suggestions using Gemini AI
    model = genai.GenerativeModel('gemini-pro')
    api_key = os.getenv("GOOGLE_API_KEY")  # Retrieve API key from environment variable
    genai.configure(api_key=api_key)

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error occurred during Gemini API call: {e}")
        return "An error occurred while fetching personalized suggestions from Gemini. Please try again later."

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

        # Get personalized suggestions
        suggestions = get_personalized_suggestions(prediction, *features)
        html_content = markdown.markdown(suggestions)

        return render_template('result.html', prediction=prediction, suggestions=html_content)

if __name__ == '__main__':
    app.run(debug=True)
