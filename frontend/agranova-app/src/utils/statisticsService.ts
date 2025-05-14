// statisticsService.ts
// This service manages application usage statistics and stores them in localStorage

export interface AppStatistics {
  cropsAnalyzed: number;
  waterSaved: number; // in liters
  yieldIncrease: number; // percentage
  diseasePrevented: number;
  lastUpdated: string;
  
  // Detailed statistics
  cropRecommendations: number;
  yieldPredictions: number;
  diseaseDetections: number;
  irrigationPredictions: number;
  
  // Most recent activities
  recentActivities: Array<{
    type: 'disease' | 'irrigation' | 'recommendation' | 'yield';
    timestamp: string;
    details: string;
  }>;
}

const STATS_STORAGE_KEY = 'agranova_statistics';

// Initialize default statistics if none exist
const initializeStatistics = (): AppStatistics => {
  return {
    cropsAnalyzed: 0,
    waterSaved: 0,
    yieldIncrease: 0,
    diseasePrevented: 0,
    lastUpdated: new Date().toISOString(),
    cropRecommendations: 0,
    yieldPredictions: 0,
    diseaseDetections: 0,
    irrigationPredictions: 0,
    recentActivities: []
  };
};

// Get current statistics from localStorage
export const getStatistics = (): AppStatistics => {
  if (typeof window === 'undefined') {
    return initializeStatistics();
  }
  
  const storedStats = localStorage.getItem(STATS_STORAGE_KEY);
  if (!storedStats) {
    const initialStats = initializeStatistics();
    localStorage.setItem(STATS_STORAGE_KEY, JSON.stringify(initialStats));
    return initialStats;
  }
  
  return JSON.parse(storedStats);
};

// Save statistics to localStorage
const saveStatistics = (stats: AppStatistics): void => {
  if (typeof window === 'undefined') return;
  
  stats.lastUpdated = new Date().toISOString();
  localStorage.setItem(STATS_STORAGE_KEY, JSON.stringify(stats));
};

// Add a new activity and update statistics
export const recordActivity = (
  type: 'disease' | 'irrigation' | 'recommendation' | 'yield',
  details: string
): void => {
  const stats = getStatistics();
  
  // Update activity counts
  stats.cropsAnalyzed++;
  
  switch (type) {
    case 'disease':
      stats.diseaseDetections++;
      stats.diseasePrevented++;
      break;
    case 'irrigation':
      stats.irrigationPredictions++;
      stats.waterSaved += Math.floor(Math.random() * 100) + 50; // Simulate water saved (50-150 liters)
      break;
    case 'recommendation':
      stats.cropRecommendations++;
      stats.yieldIncrease += 0.5; // Increase yield by 0.5% with each recommendation
      break;
    case 'yield':
      stats.yieldPredictions++;
      break;
  }
  
  // Add to recent activities
  const newActivity = {
    type,
    timestamp: new Date().toISOString(),
    details
  };
  
  stats.recentActivities.unshift(newActivity);
  
  // Keep only the 10 most recent activities
  if (stats.recentActivities.length > 10) {
    stats.recentActivities = stats.recentActivities.slice(0, 10);
  }
  
  saveStatistics(stats);
};

// Reset all statistics (for testing)
export const resetStatistics = (): void => {
  if (typeof window === 'undefined') return;
  localStorage.setItem(STATS_STORAGE_KEY, JSON.stringify(initializeStatistics()));
};

// Format a number with commas for thousands
export const formatNumber = (num: number): string => {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
};

// Format a percentage
export const formatPercentage = (num: number): string => {
  return num.toFixed(1) + '%';
};
