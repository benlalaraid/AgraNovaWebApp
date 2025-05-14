'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { predictYield, CropYieldPredictionInput, CropYieldPredictionResult } from '@/utils/cropYieldPredictionModel';
import { recordActivity } from '@/utils/statisticsService';

// Country and crop lists as provided in the requirements
const countries = ['Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia',
  'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
  'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 'Brazil',
  'Bulgaria', 'Burkina Faso', 'Burundi', 'Cameroon', 'Canada',
  'Central African Republic', 'Chile', 'Colombia', 'Croatia',
  'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
  'Eritrea', 'Estonia', 'Finland', 'France', 'Germany', 'Ghana',
  'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras',
  'Hungary', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Italy',
  'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 'Lebanon',
  'Lesotho', 'Libya', 'Lithuania', 'Madagascar', 'Malawi',
  'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico',
  'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal',
  'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Norway',
  'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal',
  'Qatar', 'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal',
  'Slovenia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan',
  'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand',
  'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom',
  'Uruguay', 'Zambia', 'Zimbabwe'];

const crops = ['Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Wheat',
  'Cassava', 'Sweet potatoes', 'Plantains and others', 'Yams'];

export default function YieldPrediction() {
  const router = useRouter();
  
  const [formData, setFormData] = useState({
    area: '',
    item: '',
    year: new Date().getFullYear().toString(),
    average_rain_fall_mm_per_year: '',
    pesticides_tonnes: '',
    avg_temp: ''
  });
  
  const [errors, setErrors] = useState<{[key: string]: string}>({});
  const [prediction, setPrediction] = useState<number | null>(null);
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
    
    if (!formData.area) {
      newErrors.area = 'Please select a country';
    }
    
    if (!formData.item) {
      newErrors.item = 'Please select a crop';
    }
    
    if (!formData.year) {
      newErrors.year = 'Year is required';
    } else {
      const yearNum = Number(formData.year);
      if (isNaN(yearNum) || yearNum < 1900 || yearNum > 2100) {
        newErrors.year = 'Year must be between 1900 and 2100';
      }
    }
    
    if (!formData.average_rain_fall_mm_per_year) {
      newErrors.average_rain_fall_mm_per_year = 'Average rainfall is required';
    } else if (isNaN(Number(formData.average_rain_fall_mm_per_year)) || Number(formData.average_rain_fall_mm_per_year) < 0) {
      newErrors.average_rain_fall_mm_per_year = 'Rainfall must be a positive number';
    }
    
    if (!formData.pesticides_tonnes) {
      newErrors.pesticides_tonnes = 'Pesticides amount is required';
    } else if (isNaN(Number(formData.pesticides_tonnes)) || Number(formData.pesticides_tonnes) < 0) {
      newErrors.pesticides_tonnes = 'Pesticides must be a positive number';
    }
    
    if (!formData.avg_temp) {
      newErrors.avg_temp = 'Average temperature is required';
    } else if (isNaN(Number(formData.avg_temp)) || Number(formData.avg_temp) < -50 || Number(formData.avg_temp) > 50) {
      newErrors.avg_temp = 'Temperature must be between -50 and 50°C';
    }
    
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
      // Prepare the input data for the model
      const modelInput: CropYieldPredictionInput = {
        area: formData.area,
        item: formData.item,
        year: Number(formData.year),
        average_rain_fall_mm_per_year: Number(formData.average_rain_fall_mm_per_year),
        pesticides_tonnes: Number(formData.pesticides_tonnes),
        avg_temp: Number(formData.avg_temp)
      };
      
      // Call the API to get the prediction from the XGBoost model
      const result = await predictYield(modelInput);
      
      // Set the prediction result
      setPrediction(result.yield_hg_ha);
      
      // Record this activity for statistics
      recordActivity('yield', `Predicted yield for ${formData.item} in ${formData.area}: ${result.yield_hg_ha.toFixed(2)} hg/ha`);
    } catch (error) {
      console.error('Error predicting crop yield:', error);
      alert('An error occurred while predicting crop yield. Please try again or check if the backend service is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      area: '',
      item: '',
      year: new Date().getFullYear().toString(),
      average_rain_fall_mm_per_year: '',
      pesticides_tonnes: '',
      avg_temp: ''
    });
    setPrediction(null);
    setErrors({});
  };

  return (
    <div className="form-page-container">
      <div className="header">
        <button className="back-button" onClick={() => router.push('/')}>←</button>
        <h1>Crop Yield Prediction</h1>
        <div style={{ width: '24px' }}></div>
      </div>
      
      <div className="prediction-layout">
        <div className="form-column">
          <div className="card">
            <form onSubmit={handleSubmit}>
              <div style={{ marginBottom: '16px' }}>
                <p>Enter crop and environmental parameters to predict yield in hectograms per hectare (hg/ha).</p>
              </div>

              <div className="form-group">
                <label htmlFor="area">Country</label>
                <select
                  id="area"
                  name="area"
                  value={formData.area}
                  onChange={handleChange}
                  className={errors.area ? 'error' : ''}
                >
                  <option value="">Select a country</option>
                  {countries.map(country => (
                    <option key={country} value={country}>{country}</option>
                  ))}
                </select>
                {errors.area && <p className="error-text">{errors.area}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="item">Crop</label>
                <select
                  id="item"
                  name="item"
                  value={formData.item}
                  onChange={handleChange}
                  className={errors.item ? 'error' : ''}
                >
                  <option value="">Select a crop</option>
                  {crops.map(crop => (
                    <option key={crop} value={crop}>{crop}</option>
                  ))}
                </select>
                {errors.item && <p className="error-text">{errors.item}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="year">Year</label>
                <input
                  type="number"
                  id="year"
                  name="year"
                  placeholder="1900-2100"
                  value={formData.year}
                  onChange={handleChange}
                  className={errors.year ? 'error' : ''}
                />
                {errors.year && <p className="error-text">{errors.year}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="average_rain_fall_mm_per_year">Average Rainfall (mm/year)</label>
                <input
                  type="number"
                  id="average_rain_fall_mm_per_year"
                  name="average_rain_fall_mm_per_year"
                  placeholder="Average rainfall in mm per year"
                  value={formData.average_rain_fall_mm_per_year}
                  onChange={handleChange}
                  className={errors.average_rain_fall_mm_per_year ? 'error' : ''}
                  step="0.01"
                />
                {errors.average_rain_fall_mm_per_year && <p className="error-text">{errors.average_rain_fall_mm_per_year}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="pesticides_tonnes">Pesticides (tonnes)</label>
                <input
                  type="number"
                  id="pesticides_tonnes"
                  name="pesticides_tonnes"
                  placeholder="Pesticides usage in tonnes"
                  value={formData.pesticides_tonnes}
                  onChange={handleChange}
                  className={errors.pesticides_tonnes ? 'error' : ''}
                  step="0.01"
                />
                {errors.pesticides_tonnes && <p className="error-text">{errors.pesticides_tonnes}</p>}
              </div>

              <div className="form-group">
                <label htmlFor="avg_temp">Average Temperature (°C)</label>
                <input
                  type="number"
                  id="avg_temp"
                  name="avg_temp"
                  placeholder="-50 to 50"
                  value={formData.avg_temp}
                  onChange={handleChange}
                  className={errors.avg_temp ? 'error' : ''}
                  step="0.01"
                />
                {errors.avg_temp && <p className="error-text">{errors.avg_temp}</p>}
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
                  {isLoading ? 'Predicting...' : 'Predict Yield'}
                </button>
              </div>
            </form>
          </div>
        </div>
        
        <div className="results-column">
          <div className="results-card">
            <h2>Yield Prediction Results</h2>
            
            {isLoading ? (
              <div className="loading-indicator">
                <div className="spinner"></div>
                <p>Analyzing crop and environmental data...</p>
              </div>
            ) : prediction !== null ? (
              <div className="results-content">
                <div className="prediction-result">
                  <h3>Predicted Yield</h3>
                  <div className="prediction-value">{prediction.toFixed(2)} hg/ha</div>
                  <p>This prediction is based on the selected country, crop type, and environmental conditions.</p>
                  
                  <div style={{marginTop: '20px'}}>
                    <h4>What does this mean?</h4>
                    <p>The predicted yield is measured in hectograms per hectare (hg/ha). A hectogram is 100 grams, so this value represents how many hundreds of grams of crop you can expect to harvest per hectare of land.</p>
                    
                    <h4>Factors affecting yield:</h4>
                    <ul>
                      <li>Average rainfall: {formData.average_rain_fall_mm_per_year} mm/year</li>
                      <li>Pesticides usage: {formData.pesticides_tonnes} tonnes</li>
                      <li>Average temperature: {formData.avg_temp}°C</li>
                    </ul>
                  </div>
                </div>
              </div>
            ) : (
              <div className="results-placeholder">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                  <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                  <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
                <h3>Enter Parameters to Get Yield Prediction</h3>
                <p>Fill out the form on the left to receive a yield prediction based on your crop and environmental conditions.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
