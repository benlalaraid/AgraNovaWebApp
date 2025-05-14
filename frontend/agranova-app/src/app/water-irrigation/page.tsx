'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';

const cropTypes = ['Potato', 'Tomato', 'Wheat', 'Rice', 'Corn', 'Barley', 'Soybean', 'Cotton'];

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

export default function WaterIrrigation() {
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
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when user types
    if (errors[name]) {
      setErrors(prev => {
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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setPrediction(mockPredictionResult);
      setIsLoading(false);
    }, 1500);
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
    <div className="container">
      <div className="header">
        <button className="back-button" onClick={() => router.push('/')}>←</button>
        <h1>Water Irrigation Prediction</h1>
        <div style={{ width: '24px' }}></div>
      </div>

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

          <div style={{ display: 'flex', gap: '16px', marginTop: '24px' }}>
            <button 
              type="button" 
              onClick={resetForm}
              style={{ backgroundColor: '#9e9e9e' }}
            >
              Reset
            </button>
            <button 
              type="submit"
              disabled={isLoading}
            >
              {isLoading ? 'Predicting...' : 'Predict Water Requirements'}
            </button>
          </div>
        </form>
      </div>

      {prediction && (
        <div className="result-container">
          <h2 style={{ marginBottom: '16px' }}>Irrigation Prediction Results</h2>
          
          <div style={{ marginBottom: '16px' }}>
            <h3 style={{ fontSize: '18px', marginBottom: '8px' }}>Total Water Requirement</h3>
            <p>{prediction.waterRequirement}</p>
          </div>
          
          <div style={{ marginBottom: '16px' }}>
            <h3 style={{ fontSize: '18px', marginBottom: '8px' }}>Irrigation Schedule</h3>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
              gap: '8px'
            }}>
              {prediction.irrigationSchedule.map((item: any, index: number) => (
                <div key={index} style={{ 
                  padding: '8px', 
                  backgroundColor: 'var(--background-color)', 
                  borderRadius: '8px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontWeight: 'bold' }}>Week {item.week}</div>
                  <div>{item.amount}</div>
                </div>
              ))}
            </div>
          </div>
          
          <div>
            <h3 style={{ fontSize: '18px', marginBottom: '8px' }}>Recommendations</h3>
            <p>{prediction.recommendations}</p>
          </div>
        </div>
      )}
    </div>
  );
}
