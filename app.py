from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load models
rf_model = joblib.load('notebooks/rf_model.joblib')
logreg_model = joblib.load('notebooks/logreg_model.joblib')

def interpret_prediction(pred):
    return "Good" if pred == 1 else "Bad" if pred == 0 else "Undetermined"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/rf', methods=['POST'])
def predict_rf():
    try:
        features = [float(request.form[f'feature{i+1}']) for i in range(4)]  # Update 4 to your feature count
        features_array = np.array(features).reshape(1, -1)
        prediction = rf_model.predict(features_array)
        return render_template('index.html', 
                             prediction_text=f'Random Forest Prediction: {interpret_prediction(prediction[0])}',
                             model_used='rf')
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}',
                             is_error=True)

@app.route('/predict/logreg', methods=['POST'])
def predict_logreg():
    try:
        features = [float(request.form[f'feature{i+1}']) for i in range(4)]  # Update 4 to your feature count
        features_array = np.array(features).reshape(1, -1)
        prediction = logreg_model.predict(features_array)
        return render_template('index.html', 
                             prediction_text=f'Logistic Regression Prediction: {interpret_prediction(prediction[0])}',
                             model_used='logreg')
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}',
                             is_error=True)

if __name__ == '__main__':
    app.run(debug=True)