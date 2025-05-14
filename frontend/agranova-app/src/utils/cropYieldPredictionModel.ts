// This file handles crop yield prediction model predictions via API

// Interface for model input
export interface CropYieldPredictionInput {
  area: string;
  item: string;
  year: number;
  average_rain_fall_mm_per_year: number;
  pesticides_tonnes: number;
  avg_temp: number;
}

// Interface for model output
export interface CropYieldPredictionResult {
  yield_hg_ha: number;
  area: string;
  item: string;
  year: number;
  message?: string;
}

// Function to predict the crop yield using the actual model via API
export const predictYield = async (input: CropYieldPredictionInput): Promise<CropYieldPredictionResult> => {
  try {
    // First, try to use the Next.js API route which will communicate with the Python backend
    const response = await fetch('/api/yield-prediction', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    const data = await response.json();
    return data as CropYieldPredictionResult;
  } catch (error) {
    console.error('Error predicting crop yield:', error);
    
    // If the API call fails, try to use the direct Python backend if available
    try {
      // This assumes you have a Python Flask backend running on port 5001
      const backendUrl = 'http://localhost:5001/api/yield-prediction';
      
      const directResponse = await fetch(backendUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(input),
      });
      
      if (!directResponse.ok) {
        throw new Error(`Direct backend request failed with status ${directResponse.status}`);
      }
      
      const directData = await directResponse.json();
      return directData as CropYieldPredictionResult;
    } catch (backendError) {
      console.error('Error connecting to Python backend:', backendError);
      
      // If both API calls fail, throw an error
      throw new Error('Failed to get crop yield prediction. Please ensure the backend service is running.');
    }
  }
};
