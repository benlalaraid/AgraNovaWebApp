'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';

// Mock disease detection results
const mockDiseaseResults = {
  status: 'Diseased',
  disease: 'Potato - Early Blight',
  details: 'Early blight is a common fungal disease that affects potato plants. It is caused by the fungus Alternaria solani and can affect leaves, stems, and tubers.',
  treatment: 'Remove infected leaves, ensure proper plant spacing for airflow, apply appropriate fungicides, and practice crop rotation.'
};

export default function DiseaseDetection() {
  const router = useRouter();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<any | null>(null);
  const [isOnline, setIsOnline] = useState(true);
  const [messages, setMessages] = useState<{ text: string; isUser: boolean }[]>([
    { text: 'Welcome to AI-Powered Leaf Disease Detection! Upload or capture a photo of a plant leaf to analyze.', isUser: false }
  ]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check online status
  useEffect(() => {
    setIsOnline(navigator.onLine);
    const handleOnlineStatusChange = () => {
      setIsOnline(navigator.onLine);
    };

    window.addEventListener('online', handleOnlineStatusChange);
    window.addEventListener('offline', handleOnlineStatusChange);

    return () => {
      window.removeEventListener('online', handleOnlineStatusChange);
      window.removeEventListener('offline', handleOnlineStatusChange);
    };
  }, []);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      
      reader.onload = (event) => {
        if (event.target && typeof event.target.result === 'string') {
          setSelectedImage(event.target.result);
          addMessage(`Image uploaded: ${file.name}`, true);
        }
      };
      
      reader.readAsDataURL(file);
    }
  };

  const captureImage = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const addMessage = (text: string, isUser: boolean) => {
    setMessages(prev => [...prev, { text, isUser }]);
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;
    
    setIsAnalyzing(true);
    addMessage('Analyzing image...', false);
    
    // Simulate processing time
    setTimeout(() => {
      // Local TensorFlow.js inference (mock)
      const localResult = {
        status: mockDiseaseResults.status,
        disease: mockDiseaseResults.disease
      };
      
      addMessage(`Analysis complete: ${localResult.status} - ${localResult.disease}`, false);
      
      // If online, get more detailed information from backend
      if (isOnline) {
        addMessage('Fetching detailed information from our database...', false);
        
        // Simulate API call
        setTimeout(() => {
          setResults(mockDiseaseResults);
          addMessage(`Disease: ${mockDiseaseResults.disease}`, false);
          addMessage(`Details: ${mockDiseaseResults.details}`, false);
          addMessage(`Treatment: ${mockDiseaseResults.treatment}`, false);
          setIsAnalyzing(false);
        }, 1000);
      } else {
        setResults(localResult);
        addMessage('You are offline. Connect to the internet for detailed information and treatment recommendations.', false);
        setIsAnalyzing(false);
      }
    }, 2000);
  };

  const resetAnalysis = () => {
    setSelectedImage(null);
    setResults(null);
    setMessages([
      { text: 'Welcome to AI-Powered Leaf Disease Detection! Upload or capture a photo of a plant leaf to analyze.', isUser: false }
    ]);
  };

  return (
    <div className="container">
      <div className="header">
        <button className="back-button" onClick={() => router.push('/')}>←</button>
        <h1>Leaf Disease Detection</h1>
        <div style={{ width: '24px' }}></div>
      </div>

      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((message, index) => (
            <div 
              key={index} 
              className={`message ${message.isUser ? 'user-message' : 'system-message'}`}
            >
              {message.text}
            </div>
          ))}
        </div>

        <div className="card" style={{ marginTop: 'auto' }}>
          {!selectedImage ? (
            <div className="image-upload-container">
              <p>Upload or capture a leaf image</p>
              <div style={{ display: 'flex', gap: '16px', marginTop: '16px' }}>
                <button onClick={captureImage}>
                  📷 Capture
                </button>
                <button onClick={captureImage}>
                  🖼️ Gallery
                </button>
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleImageUpload} 
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                />
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center' }}>
              <img 
                src={selectedImage} 
                alt="Selected leaf" 
                className="preview-image"
              />
              <div style={{ display: 'flex', gap: '16px', marginTop: '16px' }}>
                <button 
                  onClick={resetAnalysis}
                  style={{ backgroundColor: '#f44336' }}
                >
                  Cancel
                </button>
                <button 
                  onClick={analyzeImage} 
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? 'Analyzing...' : 'Analyze Leaf'}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {results && (
        <div className="result-container">
          <h2 style={{ marginBottom: '8px', color: results.status === 'Healthy' ? 'var(--primary-color)' : '#f44336' }}>
            {results.status}
          </h2>
          {results.disease && <p><strong>Disease:</strong> {results.disease}</p>}
          {results.details && <p style={{ marginTop: '8px' }}><strong>Details:</strong> {results.details}</p>}
          {results.treatment && <p style={{ marginTop: '8px' }}><strong>Treatment:</strong> {results.treatment}</p>}
        </div>
      )}
    </div>
  );
}
