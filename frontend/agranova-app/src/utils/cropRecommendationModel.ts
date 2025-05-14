// This file handles crop recommendation model predictions via API

// Define the crop labels based on the model training
const cropLabels = [
  'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
  'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
  'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
  'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
];

// Interface for model input
export interface CropRecommendationInput {
  N: number;
  P: number;
  K: number;
  temperature: number;
  humidity: number;
  ph: number;
  rainfall: number;
}

// Interface for model output
export interface CropRecommendationResult {
  recommendedCrop: string;
  confidence: number;
  alternativeCrops: string[];
  details: string;
}



// Function to get crop details
const getCropDetails = (crop: string): string => {
  // This would be replaced with actual crop information in a production environment
  const cropDetails: {[key: string]: string} = {
    'rice': 'Rice thrives in warm, humid environments with plenty of rainfall. It requires well-draining soil with good water retention.',
    'maize': 'Maize (corn) grows best in warm soil with moderate rainfall. It needs full sun exposure and well-draining soil.',
    'chickpea': 'Chickpeas prefer cool, dry growing conditions. They are drought-tolerant and require well-draining soil.',
    'kidneybeans': 'Kidney beans need warm soil and full sun. They require moderate, consistent moisture and well-draining soil.',
    'pigeonpeas': 'Pigeon peas are drought-tolerant and thrive in warm climates. They can grow in poor soil conditions.',
    'mothbeans': 'Moth beans are extremely drought-tolerant and grow well in hot, arid conditions with minimal rainfall.',
    'mungbean': 'Mung beans prefer warm temperatures and moderate rainfall. They need well-draining soil and full sun.',
    'blackgram': 'Black gram thrives in warm, humid conditions. It requires moderate rainfall and well-draining soil.',
    'lentil': 'Lentils prefer cool growing conditions and can tolerate some drought. They need well-draining soil.',
    'pomegranate': 'Pomegranates thrive in hot, dry climates. They need full sun and well-draining soil.',
    'banana': 'Bananas need consistent warmth, humidity, and moisture. They prefer rich, well-draining soil.',
    'mango': 'Mangoes thrive in tropical climates with a distinct dry season. They need full sun and well-draining soil.',
    'grapes': 'Grapes need full sun and well-draining soil. They prefer hot, dry summers and cool winters.',
    'watermelon': 'Watermelons need warm soil, hot temperatures, and consistent moisture. They require full sun and sandy, well-draining soil.',
    'muskmelon': 'Muskmelons need warm soil, hot temperatures, and moderate moisture. They require full sun and well-draining soil.',
    'apple': 'Apples need cold winters and moderate summers. They require well-draining soil and full sun.',
    'orange': 'Oranges thrive in warm, sunny conditions with moderate humidity. They need well-draining soil.',
    'papaya': 'Papayas need consistent warmth and moisture. They prefer rich, well-draining soil and protection from strong winds.',
    'coconut': 'Coconuts thrive in tropical coastal areas with high humidity. They need sandy, well-draining soil and full sun.',
    'cotton': 'Cotton needs a long, hot growing season with moderate rainfall. It requires full sun and well-draining soil.',
    'jute': 'Jute thrives in hot, humid conditions with high rainfall. It needs fertile, well-draining soil.',
    'coffee': 'Coffee grows best in tropical highlands with moderate temperatures. It needs rich, well-draining soil and partial shade.'
  };
  
  return cropDetails[crop] || 'No detailed information available for this crop.';
};

// Function to predict the recommended crop using the actual model via API
export const predictCrop = async (input: CropRecommendationInput): Promise<CropRecommendationResult> => {
  try {
    // First, try to use the Next.js API route which will communicate with the Python backend
    const response = await fetch('/api/crop-recommendation', {
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
    
    // Add crop details to the response
    if (data.recommendedCrop && !data.details) {
      data.details = getCropDetails(data.recommendedCrop);
    }
    
    return data as CropRecommendationResult;
  } catch (error) {
    console.error('Error predicting crop:', error);
    
    // If the API call fails, try to use the direct Python backend if available
    try {
      // This assumes you have a Python Flask backend running on port 5000
      const backendUrl = 'http://localhost:5000/api/crop-recommendation';
      
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
      
      // Add crop details to the response
      if (directData.recommendedCrop && !directData.details) {
        directData.details = getCropDetails(directData.recommendedCrop);
      }
      
      return directData as CropRecommendationResult;
    } catch (backendError) {
      console.error('Error connecting to Python backend:', backendError);
      
      // If both API calls fail, throw an error
      throw new Error('Failed to get crop recommendation. Please ensure the backend service is running.');
    }
  }
};


