from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open("savings_predictor.pkl", "rb"))

@app.route("/predictions")
def predictions():
    return render_template("predictions.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Create DataFrame with EXACT same feature names as training data
        features_df = pd.DataFrame({
            'Age': [float(data['age'])],
            'Income': [float(data['income'])],  # Changed from 'Monthly Income'
            'Rent': [float(data['rent'])],
            'Food Expences': [float(data['food'])]  # Changed to match training data
        })

        # Make prediction
        prediction = model.predict(features_df)[0]

        return jsonify({
            'prediction': float(prediction),
            'status': 'success'
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(debug=True)

