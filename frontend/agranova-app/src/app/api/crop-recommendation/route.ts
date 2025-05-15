import { NextRequest, NextResponse } from 'next/server';

// Define the crop labels based on the model training
const cropLabels = [
  'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
  'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
  'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
  'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
];

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

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { N, P, K, temperature, humidity, ph, rainfall } = body;
    
    // Validate input
    if (
      N === undefined || P === undefined || K === undefined || 
      temperature === undefined || humidity === undefined || 
      ph === undefined || rainfall === undefined
    ) {
      return NextResponse.json(
        { error: 'Missing required parameters' },
        { status: 400 }
      );
    }
    
    // In a production environment, this would call a Python backend service
    // that loads the .pkl model and makes predictions
    // For now, we'll make a simulated API call to demonstrate the flow
    
    // This is where we would make an API call to a Python backend
    // that would load the model and make predictions
    
    // For demonstration purposes, we'll use a simplified approach
    // In a real implementation, this would be replaced with an actual API call
    
    // Create a URL to the Python backend service (to be implemented)
    const backendUrl = 'http://127.0.0.1:5000/api/crop-recommendation';
    
    // Make a fetch request to the Python backend
    // This is commented out since we don't have the Python backend yet
    
   
    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ N, P, K, temperature, humidity, ph, rainfall }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to get prediction from backend');
    }
    
    const data = await response.json();
    const { prediction } = data;
    console.log('prediction : ',prediction)
    return NextResponse.json({
      prediction,
    });
   
    // For now, we'll use a placeholder prediction based on common crop requirements
    // This will be replaced with the actual model prediction from the Python backend
    
    // IMPORTANT: This is a temporary placeholder and should be replaced with actual model predictions
    // from the .pkl model in the backend
    
    // Determine the most suitable crop based on input parameters
    let bestCrop = 'rice'; // Default
    let confidence = 70; // Default confidence
    
    // pH-based selection (simplified)
    if (ph < 5.5) {
      bestCrop = 'rice';
    } else if (ph >= 5.5 && ph < 6.5) {
      bestCrop = 'maize';
    } else if (ph >= 6.5 && ph < 7.5) {
      bestCrop = 'cotton';
    } else {
      bestCrop = 'chickpea';
    }
    
    // Adjust based on temperature
    if (temperature < 20) {
      if (ph < 6) bestCrop = 'rice';
      else bestCrop = 'chickpea';
    } else if (temperature > 30) {
      if (humidity > 70) bestCrop = 'rice';
      else bestCrop = 'cotton';
    }
    
    // Adjust based on rainfall
    if (rainfall > 200) {
      bestCrop = 'rice';
    } else if (rainfall < 80) {
      bestCrop = 'chickpea';
    }
    
    // Get alternative crops (different from best crop)
    const alternativeCrops = cropLabels
      .filter(crop => crop !== bestCrop)
      .sort(() => 0.5 - Math.random()) // Shuffle
      .slice(0, 2); // Take first 2
    
    // Return the prediction result
    return NextResponse.json({
      recommendedCrop: bestCrop,
      confidence: confidence,
      alternativeCrops: alternativeCrops,
      details: getCropDetails(bestCrop),
      message: "NOTE: This is a placeholder prediction. To use the actual model, you need to implement a Python backend service that loads the .pkl model."
    });
    
  } catch (error) {
    console.error('Error in crop recommendation API:', error);
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    );
  }
}
