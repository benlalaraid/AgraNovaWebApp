import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to the model file
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks and best models",
    "crop recommendation",
    "best_random_forest_model_for_Crop_recommendation.pkl"
)

# Load the model
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the crop labels (must match the order used during model training)
crop_labels = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
    'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
    'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
]


@app.route('/api/crop-recommendation', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get input data from request
        data = request.json
        
        # Extract features
        N = float(data.get('N', 0))
        P = float(data.get('P', 0))
        K = float(data.get('K', 0))
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        ph = float(data.get('ph', 0))
        rainfall = float(data.get('rainfall', 0))
        
        # Create input array for model
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        
        # Get the predicted crop
        predicted_crop = crop_labels[prediction]
        
        # Get confidence (probability of the predicted class)
        confidence = float(probabilities[prediction]) * 100
        
        # Get alternative crops (next highest probabilities)
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        alternative_indices = sorted_indices[1:3]  # Get the next 2 highest probability indices
        alternative_crops = [crop_labels[i] for i in alternative_indices]
        
        # Return the prediction result
        return jsonify({
            'recommendedCrop': predicted_crop,
            'confidence': round(confidence),
            'alternativeCrops': alternative_crops,
            'probabilities': {crop_labels[i]: float(probabilities[i]) for i in range(len(crop_labels))}
        })
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
