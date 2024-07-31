
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load models
churn_model = joblib.load('churn_model.pkl')

app = Flask(__name__)

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    data = request.get_json(force=True)
    prediction = churn_model.predict([list(data.values())])
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
