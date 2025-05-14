'use client';

import { useRouter } from 'next/navigation';
import Image from 'next/image';
import React, { useState, useEffect } from 'react';
import { getStatistics, formatNumber, formatPercentage, AppStatistics } from '@/utils/statisticsService';

// Define feature types
interface Feature {
  id: string;
  title: string;
  description: string;
  icon: string;
  path: string;
  color: string;
}

// Define component types
interface StatCard {
  title: string;
  value: string;
  change: string;
  isPositive: boolean;
  icon: string;
}

export default function Home() {
  const router = useRouter();
  // State for statistics only
  const [statistics, setStatistics] = useState<AppStatistics | null>(null);
  
  // Load statistics from local storage
  useEffect(() => {
    const stats = getStatistics();
    setStatistics(stats);
    
    // Set up interval to refresh statistics every 5 seconds
    const interval = setInterval(() => {
      setStatistics(getStatistics());
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  // Features data
  const features: Feature[] = [
    {
      id: 'disease',
      title: 'Crop Disease Detection',
      description: 'Identify plant diseases using AI-powered image analysis',
      icon: '🍃',
      path: '/disease-detection',
      color: '#4caf50'
    },
    {
      id: 'irrigation',
      title: 'Water Irrigation Prediction',
      description: 'Optimize water usage based on crop and location data',
      icon: '💧',
      path: '/water-irrigation',
      color: '#2196f3'
    },
    {
      id: 'recommendation',
      title: 'Crop Recommendation',
      description: 'Get ideal crop suggestions based on soil parameters',
      icon: '🌱',
      path: '/crop-recommendation',
      color: '#ff9800'
    },
    {
      id: 'yield',
      title: 'Crop Yield Prediction',
      description: 'Forecast crop yields using advanced AI models',
      icon: '📊',
      path: '/yield-prediction',
      color: '#9c27b0'
    }
  ];

  // Generate stats data from actual statistics
  const stats: StatCard[] = [
    {
      title: 'Crops Analyzed',
      value: statistics ? formatNumber(statistics.cropsAnalyzed) : '0',
      change: '+1 today',
      isPositive: true,
      icon: '🌾'
    },
    {
      title: 'Water Saved',
      value: statistics ? `${formatNumber(statistics.waterSaved)}L` : '0L',
      change: statistics && statistics.irrigationPredictions > 0 ? `+${statistics.irrigationPredictions} predictions` : 'No data',
      isPositive: true,
      icon: '💧'
    },
    {
      title: 'Yield Increase',
      value: statistics ? formatPercentage(statistics.yieldIncrease) : '0%',
      change: statistics && statistics.cropRecommendations > 0 ? `+${statistics.cropRecommendations} recommendations` : 'No data',
      isPositive: true,
      icon: '📈'
    },
    {
      title: 'Disease Prevention',
      value: statistics && statistics.diseasePrevented > 0 ? formatNumber(statistics.diseasePrevented) : '0',
      change: statistics && statistics.diseaseDetections > 0 ? `+${statistics.diseaseDetections} detections` : 'No data',
      isPositive: true,
      icon: '🛡️'
    }
  ];

  return (
    <div className="dashboard-container">
      {/* Dashboard Header */}
      <div className="dashboard-header">
        <div className="header-left">
          <h1>Dashboard</h1>
          <p>Welcome to AgraNova - Smart Agriculture Solutions</p>
        </div>
        <div className="header-right">
          <button className="btn-primary">New Analysis</button>
        </div>
      </div>

      {/* Dashboard content starts directly with stats */}

      {/* Stats Cards */}
      <div className="stats-grid">
        {stats.map((stat, index) => (
          <div className="stat-card" key={index}>
            <div className="stat-icon" style={{ backgroundColor: `rgba(${stat.isPositive ? '76, 175, 80' : '244, 67, 54'}, 0.1)` }}>
              <span>{stat.icon}</span>
            </div>
            <div className="stat-content">
              <h3>{stat.title}</h3>
              <div className="stat-value-container">
                <p className="stat-value">{stat.value}</p>
                <span className={`stat-change ${stat.isPositive ? 'positive' : 'negative'}`}>
                  {stat.change}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Features Grid */}
      <div className="features-section">
        <h2 className="section-title">Smart Agriculture Features</h2>
        <div className="features-grid">
          {features.map((feature) => (
            <div 
              className="feature-card" 
              key={feature.id}
              onClick={() => router.push(feature.path)}
            >
              <div className="feature-icon" style={{ backgroundColor: `${feature.color}20` }}>
                <span>{feature.icon}</span>
              </div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
              <button className="feature-button" style={{ backgroundColor: feature.color }}>
                Open
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Activity */}
      <div className="recent-activity">
        <h2 className="section-title">Recent Activity</h2>
        <div className="activity-list">
          {statistics && statistics.recentActivities.length > 0 ? (
            statistics.recentActivities.map((activity, index) => {
              // Determine icon and color based on activity type
              let icon = '🌱';
              let colorClass = 'green';
              let title = 'Activity';
              
              switch(activity.type) {
                case 'disease':
                  icon = '🍃';
                  colorClass = 'green';
                  title = 'Disease Detection';
                  break;
                case 'irrigation':
                  icon = '💧';
                  colorClass = 'blue';
                  title = 'Water Irrigation';
                  break;
                case 'recommendation':
                  icon = '🌱';
                  colorClass = 'orange';
                  title = 'Crop Recommendation';
                  break;
                case 'yield':
                  icon = '📊';
                  colorClass = 'purple';
                  title = 'Yield Prediction';
                  break;
              }
              
              // Format timestamp
              const timestamp = new Date(activity.timestamp);
              const now = new Date();
              const diffMs = now.getTime() - timestamp.getTime();
              const diffMins = Math.floor(diffMs / 60000);
              const diffHours = Math.floor(diffMins / 60);
              const diffDays = Math.floor(diffHours / 24);
              
              let timeAgo;
              if (diffMins < 1) {
                timeAgo = 'Just now';
              } else if (diffMins < 60) {
                timeAgo = `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
              } else if (diffHours < 24) {
                timeAgo = `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
              } else {
                timeAgo = `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
              }
              
              return (
                <div className="activity-item" key={index}>
                  <div className={`activity-icon ${colorClass}`}>{icon}</div>
                  <div className="activity-content">
                    <h4>{title}</h4>
                    <p>{activity.details}</p>
                    <span className="activity-time">{timeAgo}</span>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="empty-state">
              <p>No recent activities yet. Start using the app features to see your activity here!</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
