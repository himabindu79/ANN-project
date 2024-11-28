# Customer Churn Prediction

A machine learning application that predicts customer churn probability using deep learning.

## Features
- Neural network model trained on customer data
- Interactive web interface using Streamlit
- Real-time predictions
- Detailed customer insights and recommendations

## Setup
1. Install dependencies:
```bash
pip install streamlit tensorflow scikit-learn pandas numpy pickle
```

2. Required files:
- `model.h5`: Trained neural network model
- `label_encoder_gender.pkl`: Gender label encoder
- `onehot_encoder_geo.pkl`: Geography one-hot encoder
- `scaler.pkl`: Feature scaler

## Usage
Run the application:
```bash
streamlit run app.py
```

## Input Features
- Geography (France/Germany/Spain)
- Gender
- Age (18-92)
- Account Balance
- Credit Score (300-850)
- Estimated Salary
- Tenure (0-10 years)
- Number of Products (1-4)
- Credit Card Status
- Membership Status

## Model Details
- Architecture: Deep Neural Network
- Input: 12 features (after encoding)
- Output: Churn probability (0-1)
- Preprocessing: Standard scaling, label encoding, one-hot encoding

## Files
- `app.py`: Main Streamlit application
- `model.h5`: Trained model
- `*.pkl`: Preprocessor files

## Prediction Output
- Churn probability percentage
- Risk assessment (High/Low)
- Recommended actions based on prediction
