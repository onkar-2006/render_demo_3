from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Step 1: Load the saved model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Step 2: Define the home route for rendering the form
@app.route('/')
def home():
    return render_template('heart_disease.html')  # Renders the HTML form


# Step 3: Define the predict route to handle the form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Prepare the input data for the model as a numpy array
        input_data = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        ]])

        # Predict the probability of heart disease (risk score)
        risk_score = model.predict_proba(input_data)[0][1]  # Probability for class 1 (heart disease)

        # Predict the class (heart disease or not) based on threshold (0.5)
        if risk_score > 0.5:
            prediction = "The person has heart disease."
        else:
            prediction = "The person does not have heart disease."

        # Return the result as a JSON response
        return jsonify({"prediction": prediction, "risk_score": round(risk_score, 2)})

    except Exception as e:
        # In case of any error, return a message to the user
        return jsonify({"error": str(e)})


# Step 4: Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
