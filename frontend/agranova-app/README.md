# AgraNova Mobile Application

A Next.js mobile application for the AgraNova project, part of the Algerian AI Olympiades. This application provides smart agriculture solutions to help farmers make data-driven decisions.

## Features

The application consists of 5 main pages:

1. **Home Page** - Navigation dashboard to access all features
2. **Crop Disease Detection** - AI-powered leaf disease detection using image analysis
3. **Water Irrigation Prediction** - Predict water requirements based on crop and location data
4. **Crop Recommendation** - Get crop recommendations based on soil and environmental parameters
5. **Crop Yield Prediction** - Predict crop yields based on location, crop type, and environmental factors

## Technologies Used

- Next.js 14
- React 18
- TypeScript
- TensorFlow.js (for local model inference)

## Getting Started

### Prerequisites

- Node.js 18.17 or later

### Installation

1. Clone the repository
```bash
git clone https://github.com/your-username/agranova-app.git
cd agranova-app
```

2. Install dependencies
```bash
npm install
```

3. Run the development server
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser to see the application

## Project Structure

```
agranova-app/
├── src/
│   ├── app/
│   │   ├── disease-detection/
│   │   │   └── page.tsx
│   │   ├── water-irrigation/
│   │   │   └── page.tsx
│   │   ├── crop-recommendation/
│   │   │   └── page.tsx
│   │   ├── yield-prediction/
│   │   │   └── page.tsx
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   └── ...
├── public/
├── package.json
├── tsconfig.json
└── ...
```

## Models

The application uses the following models:

- **Crop Disease Detection**: TensorFlow.js model for leaf disease classification
- **Water Irrigation Prediction**: Mock prediction model (to be replaced with actual model)
- **Crop Recommendation**: Mock recommendation model (to be replaced with actual model)
- **Crop Yield Prediction**: XGBoost model for yield prediction

## Future Improvements

- Integration with actual backend APIs
- Offline data storage
- User profiles and history
- Multi-language support
- Enhanced visualization of predictions

## License

This project is part of the Algerian AI Olympiades.
