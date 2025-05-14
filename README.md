# AgraNova Web App

A smart agriculture application developed for the Algerian AI Olympiades.

## Features

- **Crop Disease Detection**: Identify plant diseases using AI-powered image analysis
- **Water Irrigation Prediction**: Optimize water usage based on crop and location data
- **Crop Recommendation**: Get ideal crop suggestions based on soil parameters
- **Crop Yield Prediction**: Forecast crop yields using advanced AI models

## Tech Stack

- Frontend: Next.js, React, TypeScript
- Backend: Python Flask API with machine learning models
- Models: XGBoost for yield prediction, Random Forest for crop recommendation

## Project Structure

- `/frontend`: Next.js application
- `/backend`: Python Flask API and ML models

## Getting Started

### Frontend

```bash
cd frontend/agranova-app
npm install
npm run dev
```

### Backend

```bash
cd backend
pip install -r requirements.txt
python api/crop_recommendation_model.py
python api/crop_yield_prediction_model.py
```

Visit `http://localhost:3000` to see the application.
