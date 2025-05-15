'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { predictCrop, CropRecommendationInput, CropRecommendationResult } from '@/utils/cropRecommendationModel';
import { recordActivity } from '@/utils/statisticsService';

export default function CropRecommendation() {
  const router = useRouter();
  
  const [formData, setFormData] = useState({
    N: '',
    P: '',
    K: '',
    temperature: '',
    humidity: '',
    ph: '',
    rainfall: ''
  });
  
  const [errors, setErrors] = useState<{[key: string]: string}>({});
  const [recommendation, setRecommendation] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
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
    const fields = [
      { name: 'N', label: 'Nitrogen', min: 0, max: 140 },
      { name: 'P', label: 'Phosphorus', min: 5, max: 145 },
      { name: 'K', label: 'Potassium', min: 5, max: 205 },
      { name: 'temperature', label: 'Temperature', min: 0, max: 50 },
      { name: 'humidity', label: 'Humidity', min: 0, max: 100 },
      { name: 'ph', label: 'pH', min: 0, max: 14 },
      { name: 'rainfall', label: 'Rainfall', min: 0, max: 300 }
    ];
    
    fields.forEach(field => {
      if (!formData[field.name as keyof typeof formData]) {
        newErrors[field.name] = `${field.label} is required`;
      } else {
        const value = Number(formData[field.name as keyof typeof formData]);
        if (isNaN(value)) {
          newErrors[field.name] = `${field.label} must be a number`;
        } else if (value < field.min || value > field.max) {
          newErrors[field.name] = `${field.label} must be between ${field.min} and ${field.max}`;
        }
      }
    });
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    setIsLoading(true);
    
    try {
      // Convert form data to numbers for model input
      const modelInput: CropRecommendationInput = {
        N: Number(formData.N),
        P: Number(formData.P),
        K: Number(formData.K),
        temperature: Number(formData.temperature),
        humidity: Number(formData.humidity),
        ph: Number(formData.ph),
        rainfall: Number(formData.rainfall)
      };
      
      // Get prediction from the model
      const result = await predictCrop(modelInput) as any;
      setRecommendation(result);
      
      // Record this activity for statistics
      recordActivity('recommendation', `Recommended crop: ${result.prediction}`);
    } catch (error) {
      console.error('Error predicting crop:', error);
      // Fallback to mock data if model fails
      alert('An error occurred with the model. Using fallback data.');
      const fallbackResult = {
        recommendedCrop: 'Coffee',
        confidence: 87,
        alternativeCrops: ['Tea', 'Cocoa'],
        details: 'Coffee thrives in the soil conditions you provided. The combination of high humidity, moderate rainfall, and slightly acidic pH creates an ideal environment for coffee cultivation.'
      };
      setRecommendation(fallbackResult);
      
      // Record this activity for statistics even with fallback data
      recordActivity('recommendation', `Recommended crop: ${fallbackResult.recommendedCrop}`);
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      N: '',
      P: '',
      K: '',
      temperature: '',
      humidity: '',
      ph: '',
      rainfall: ''
    });
    setRecommendation(null);
    setErrors({});
  };

  return (
    <div className="form-page-container">
      <div className="header">
        <button className="back-button" onClick={() => router.push('/')}>←</button>
        <h1>Crop Recommendation</h1>
        <div style={{ width: '24px' }}></div>
      </div>
      
      <div className="prediction-layout">
        <div className="form-column">
          <div className="card">
            <form onSubmit={handleSubmit}>
              <div style={{ marginBottom: '16px' }}>
                <p>Enter soil and environmental parameters to get crop recommendations.</p>
              </div>

              <div className="form-group">
                <label htmlFor="N">Nitrogen (N) - kg/ha</label>
                <input
                  type="number"
                  id="N"
                  name="N"
                  placeholder="0-140"
                  value={formData.N}
                  onChange={handleChange}
                  className={errors.N ? 'error' : ''}
                  step="0.01"
                />
                {errors.N && <p className="error-text">{errors.N}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="P">Phosphorus (P) - kg/ha</label>
                <input
                  type="number"
                  id="P"
                  name="P"
                  placeholder="5-145"
                  value={formData.P}
                  onChange={handleChange}
                  className={errors.P ? 'error' : ''}
                  step="0.01"
                />
                {errors.P && <p className="error-text">{errors.P}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="K">Potassium (K) - kg/ha</label>
                <input
                  type="number"
                  id="K"
                  name="K"
                  placeholder="5-205"
                  value={formData.K}
                  onChange={handleChange}
                  className={errors.K ? 'error' : ''}
                  step="0.01"
                />
                {errors.K && <p className="error-text">{errors.K}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="temperature">Temperature (°C)</label>
                <input
                  type="number"
                  id="temperature"
                  name="temperature"
                  placeholder="0-50"
                  value={formData.temperature}
                  onChange={handleChange}
                  className={errors.temperature ? 'error' : ''}
                  step="0.01"
                />
                {errors.temperature && <p className="error-text">{errors.temperature}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="humidity">Humidity (%)</label>
                <input
                  type="number"
                  id="humidity"
                  name="humidity"
                  placeholder="0-100"
                  value={formData.humidity}
                  onChange={handleChange}
                  className={errors.humidity ? 'error' : ''}
                  step="0.01"
                />
                {errors.humidity && <p className="error-text">{errors.humidity}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="ph">pH Level</label>
                <input
                  type="number"
                  id="ph"
                  name="ph"
                  placeholder="0-14"
                  value={formData.ph}
                  onChange={handleChange}
                  className={errors.ph ? 'error' : ''}
                  step="0.01"
                />
                {errors.ph && <p className="error-text">{errors.ph}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="rainfall">Rainfall (mm)</label>
                <input
                  type="number"
                  id="rainfall"
                  name="rainfall"
                  placeholder="0-300"
                  value={formData.rainfall}
                  onChange={handleChange}
                  className={errors.rainfall ? 'error' : ''}
                  step="0.01"
                />
                {errors.rainfall && <p className="error-text">{errors.rainfall}</p>}
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
                  {isLoading ? 'Analyzing...' : 'Get Recommendation'}
                </button>
              </div>
            </form>
          </div>
        </div>
        
        <div className="results-column">
          <div className="results-card">
            <h2>Recommendation Results</h2>
            
            {isLoading ? (
              <div className="loading-indicator">
                <div className="spinner"></div>
                <p>Analyzing soil and environmental parameters...</p>
              </div>
            ) : recommendation ? (
              <div className="results-content">
                <div className="recommendation-result">
                  <h3>Best Crop for Your Conditions</h3>
                  <div className="crop-recommendation">
                    <div className="recommended-crop">{recommendation.prediction}</div>
                    
                    {/* <div>
                      <div className="confidence-label">Confidence: {recommendation.confidence}%</div>
                      <div className="confidence-bar">
                        <div 
                          className="confidence-level" 
                          style={{ width: `${recommendation.confidence}%` }}
                        ></div>
                      </div>
                    </div> */}
                    
                    <p className="recommendation-details">{recommendation.details}</p>
                    
                    {recommendation.alternativeCrops && recommendation.alternativeCrops.length > 0 && (
                      <div>
                        <h4>Alternative Crops</h4>
                        <div className="alternative-crops">
                          {recommendation.alternativeCrops.map((crop, index) => (
                            <span key={index} className="alternative-crop">{crop}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="results-placeholder">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M18 16.8a7.14 7.14 0 0 0 2.24-3.22 8.34 8.34 0 0 0 .25-2.08c.04-1.03-.43-1.99-1.24-2.65a3.65 3.65 0 0 0-2.53-.95c-1.36 0-2.5.8-3.03 2.1a3.38 3.38 0 0 0-2.66 0c-.54-1.3-1.67-2.1-3.03-2.1-.95 0-1.83.35-2.53.95A3.95 3.95 0 0 0 4.21 11.5c-.1.78.1 1.56.25 2.08.4 1.25 1.22 2.42 2.24 3.22"></path>
                  <path d="M12 20.8c1.74 0 3.41-.47 4.84-1.34l-.28-.45c-.23-.37-.4-.8-.4-1.26 0-1.31 1.34-2.24 2.74-2.24.71 0 1.37.25 1.87.66a7.99 7.99 0 0 0-3.79-6.68 7.99 7.99 0 0 0-10 0 7.99 7.99 0 0 0-3.79 6.68c.5-.41 1.16-.66 1.87-.66 1.4 0 2.74.93 2.74 2.24 0 .46-.17.89-.4 1.26l-.28.45A7.98 7.98 0 0 0 12 20.8z"></path>
                </svg>
                <h3>Enter Parameters to Get Recommendations</h3>
                <p>Fill out the form on the left to receive crop recommendations based on your soil and environmental conditions.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
