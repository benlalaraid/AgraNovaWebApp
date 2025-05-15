'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { recordActivity } from '@/utils/statisticsService';

const cropTypes = ['wheat','tomato','potato', 'strawberry', 'onion', 'carrot'];

// Mock API response
const mockPredictionResult = {
  waterRequirement: '450 mm',
  irrigationSchedule: [
    { week: 1, amount: '50 mm' },
    { week: 2, amount: '75 mm' },
    { week: 3, amount: '100 mm' },
    { week: 4, amount: '125 mm' },
    { week: 5, amount: '100 mm' },
  ],
  recommendations: 'Based on your location and planting time, we recommend drip irrigation for optimal water usage. Monitor soil moisture regularly and adjust irrigation accordingly.'
};

export default function WaterIrrigation(){
  const router = useRouter();
  
  const [formData, setFormData] = useState({
    cropType: '',
    latitude: '',
    longitude: '',
    plantingTime: ''
  });
  
  const [errors, setErrors] = useState<{[key: string]: string}>({});
  const [prediction, setPrediction] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData((prev:any) => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when user types
    if (errors[name]) {
      setErrors((prev:any) => {
        const newErrors = {...prev};
        delete newErrors[name];
        return newErrors;
      });
    }
  };

  const validateForm = () => {
    const newErrors: {[key: string]: string} = {};
    
    if (!formData.cropType) {
      newErrors.cropType = 'Please select a crop type';
    }
    
    if (!formData.latitude) {
      newErrors.latitude = 'Latitude is required';
    } else if (isNaN(Number(formData.latitude)) || Number(formData.latitude) < -90 || Number(formData.latitude) > 90) {
      newErrors.latitude = 'Latitude must be between -90 and 90';
    }
    
    if (!formData.longitude) {
      newErrors.longitude = 'Longitude is required';
    } else if (isNaN(Number(formData.longitude)) || Number(formData.longitude) < -180 || Number(formData.longitude) > 180) {
      newErrors.longitude = 'Longitude must be between -180 and 180';
    }
    
    if (!formData.plantingTime) {
      newErrors.plantingTime = 'Planting time is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  function calculateDays(plantingTime:any) {
    if (!plantingTime) return;

    const selectedDate = new Date(plantingTime);
    const today = new Date();

    // Reset time part to avoid partial days
    selectedDate.setHours(0, 0, 0, 0);
    today.setHours(0, 0, 0, 0);

    const diffTime = today - selectedDate;
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    return diffDays
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    const req_body = {
        long:Number(formData.longitude),
        lat:Number(formData.latitude),
        crop:formData.cropType,
        days:calculateDays(formData.plantingTime),
      }
    console.log('formData : ', formData)
    console.log('req_body : ', req_body)
    setIsLoading(true);
    // const response = await fetch('https://achref888.app.n8n.cloud/webhook/1d30ace0-f76d-47f0-8860-9e6d94d80cdc',{
    //   method:"POST",
    //   body:JSON.stringify(req_body),
    // })

    const response = await fetch('/api/water-irrigation', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req_body),
    });

    const result = await response.json();
    
    console.log('result: ',result)
    setPrediction((prev:any)=>({
      daily_irrigation:result.data.daily_irrigation,
      recommendation:result.data.recommendation,
    }))
    
    setIsLoading(false);
    
    
    // // Simulate API call
    // setTimeout(() => {
    //   setPrediction(mockPredictionResult);
    //   setIsLoading(false);
      
    //   // Record this activity for statistics
    //   recordActivity('irrigation', `Water irrigation prediction for ${formData.cropType}`);
    // }, 1500);


  };

  const resetForm = () => {
    setFormData({
      cropType: '',
      latitude: '',
      longitude: '',
      plantingTime: ''
    });
    setPrediction(null);
    setErrors({});
  };

  return (
    <div className="form-page-container">
      <div className="header">
        <button className="back-button" onClick={() => router.push('/')}>←</button>
        <h1>Water Irrigation Prediction</h1>
        <div style={{ width: '24px' }}></div>
      </div>
      
      <div className="prediction-layout">
        <div className="form-column">
          <div className="card">
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="cropType">Crop Type</label>
                <select
                  id="cropType"
                  name="cropType"
                  value={formData.cropType}
                  onChange={handleChange}
                  className={errors.cropType ? 'error' : ''}
                >
                  <option value="">Select a crop</option>
                  {cropTypes.map(crop => (
                    <option key={crop} value={crop}>{crop}</option>
                  ))}
                </select>
                {errors.cropType && <p className="error-text">{errors.cropType}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="latitude">Latitude</label>
                <input
                  type="number"
                  id="latitude"
                  name="latitude"
                  placeholder="e.g., 36.7"
                  value={formData.latitude}
                  onChange={handleChange}
                  className={errors.latitude ? 'error' : ''}
                  step="0.000001"
                />
                {errors.latitude && <p className="error-text">{errors.latitude}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="longitude">Longitude</label>
                <input
                  type="number"
                  id="longitude"
                  name="longitude"
                  placeholder="e.g., 3.2"
                  value={formData.longitude}
                  onChange={handleChange}
                  className={errors.longitude ? 'error' : ''}
                  step="0.000001"
                />
                {errors.longitude && <p className="error-text">{errors.longitude}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="plantingTime">Planting Time</label>
                <input
                  type="date"
                  id="plantingTime"
                  name="plantingTime"
                  value={formData.plantingTime}
                  onChange={handleChange}
                  className={errors.plantingTime ? 'error' : ''}
                />
                {errors.plantingTime && <p className="error-text">{errors.plantingTime}</p>}
              </div>

              <div className="button-container">
                <button 
                  type="button" 
                  className="btn-secondary"
                  onClick={resetForm}
                  disabled={isLoading}
                >
                  Reset
                </button>
                <button 
                  type="submit"
                  className="btn-primary"
                  disabled={isLoading}
                >
                  {isLoading ? 'Predicting...' : 'Predict Water Requirements'}
                </button>
              </div>
            </form>
          </div>
        </div>
        
        <div className="results-column">
          <div className="results-card">
            <h2>Irrigation Prediction Results</h2>
            
            {isLoading ? (
              <div className="loading-indicator">
                <div className="spinner"></div>
                <p>Analyzing crop and location data...</p>
              </div>
            ) : prediction ? (
              <div className="results-content">
                <div className="prediction-result">
                  <h3>Total Water Requirement</h3>
                  <div className="prediction-value">{prediction.daily_irrigation}</div>
                  
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
                    gap: '8px',
                    marginTop: '10px'
                  }}>
                    
                  </div>
                  
                  <h3 style={{ marginTop: '20px' }}>Recommendations</h3>
                  <p>{prediction.recommendation}</p>
                </div>
              </div>
            ) : (
              <div className="results-placeholder">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                  <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                  <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
                <h3>Enter Parameters to Get Irrigation Prediction</h3>
                <p>Fill out the form on the left to receive irrigation recommendations based on your crop and location data.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
