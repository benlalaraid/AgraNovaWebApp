import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { area, item, year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp } = body;
    
    // Validate input
    if (
      !area || !item || !year || 
      average_rain_fall_mm_per_year === undefined || 
      pesticides_tonnes === undefined || 
      avg_temp === undefined
    ) {
      return NextResponse.json(
        { error: 'Missing required parameters' },
        { status: 400 }
      );
    }
    
    // Connect to the Python backend service that loads the XGBoost model and makes predictions
    try {
      // Connect to the Python backend running on port 5001
      const backendUrl = 'http://localhost:5001/api/yield-prediction';
      
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ area, item, year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get prediction from backend: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Return the prediction result from the actual model
      return NextResponse.json(data);
    } catch (backendError) {
      console.error('Error connecting to Python backend:', backendError);
      
      // If we can't connect to the Python backend, use a deterministic calculation instead of random
      // This is still a fallback, but without randomness
      const yield_hg_ha = calculateDeterministicYield(area, item, year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp);
      
      return NextResponse.json({
        yield_hg_ha,
        area,
        item,
        year,
        message: "NOTE: This is a fallback prediction. The Python backend service is not available."
      });
    }
  } catch (error) {
    console.error('Error in yield prediction API:', error);
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    );
  }
}

// Deterministic fallback function to calculate yield based on input parameters
// This is used only when the Python backend is not available
function calculateDeterministicYield(
  area: string, 
  item: string, 
  year: number, 
  average_rain_fall_mm_per_year: number, 
  pesticides_tonnes: number, 
  avg_temp: number
): number {
  // This is a simplified deterministic calculation
  // It's still not as accurate as the XGBoost model, but at least it's consistent
  
  // Base yield values for different crops (very simplified)
  const cropBaseYields: {[key: string]: number} = {
    'Maize': 40000,
    'Potatoes': 50000,
    'Rice, paddy': 45000,
    'Sorghum': 35000,
    'Soybeans': 30000,
    'Wheat': 38000,
    'Cassava': 42000,
    'Sweet potatoes': 45000,
    'Plantains and others': 48000,
    'Yams': 46000
  };
  
  // Area factors (deterministic based on first letter of area name)
  const areaFactors: {[key: string]: number} = {
    'A': 1.05, 'B': 0.95, 'C': 1.10, 'D': 0.90, 'E': 1.15,
    'F': 0.85, 'G': 1.20, 'H': 0.80, 'I': 1.25, 'J': 0.75,
    'K': 1.30, 'L': 0.70, 'M': 1.35, 'N': 0.65, 'O': 1.40,
    'P': 0.60, 'Q': 1.45, 'R': 0.55, 'S': 1.50, 'T': 0.50,
    'U': 1.55, 'V': 0.45, 'W': 1.60, 'X': 0.40, 'Y': 1.65, 'Z': 0.35
  };
  
  // Get base yield for the selected crop, or use average if not found
  const baseYield = cropBaseYields[item] || 40000;
  
  // Get area factor based on first letter of area name
  const firstLetter = area.charAt(0).toUpperCase();
  const areaFactor = areaFactors[firstLetter] || 1.0;
  
  // Apply deterministic adjustments based on input parameters
  const yearFactor = (year - 2000) * 100; // Increase over time
  const rainFactor = average_rain_fall_mm_per_year * 10; // More rain generally means higher yield
  const tempFactor = Math.abs(avg_temp - 25) * 200; // Optimal temp around 25°C
  const pesticideFactor = pesticides_tonnes * 5; // More pesticides can increase yield
  
  // Calculate adjusted yield (deterministically)
  const adjustedYield = (baseYield + yearFactor + rainFactor - tempFactor + pesticideFactor) * areaFactor;
  
  // Ensure the yield is within reasonable bounds
  return Math.max(10000, Math.min(100000, Math.round(adjustedYield)));
}
