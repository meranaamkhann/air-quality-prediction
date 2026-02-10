# Air Quality Prediction Model

This project predicts Air Quality Index (AQI) using machine learning based on
historical air pollution data.

## Dataset
The dataset contains pollutant concentrations such as PM2.5, PM10, CO, NO2,
SO2, and O3 collected over time.

# Features Used
- PM2.5
- PM10
- NO2
- SO2
- CO
- O3

## Visualizations
- Actual vs Predicted AQI plot
- Feature importance plot

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Flask

# Approach
- Data cleaning and preprocessing
- Exploratory data analysis
- Feature selection
- Model training and evaluation
- Model deployment using Flask API

## Model
- Random Forest Regressor
- R² Score: ~0.94
- MAE: ~14.8

## Future Improvements
- Improve predictions using deep learning models
- Deploy as a web application or API

## How to Run
- Install dependencies  
   pip install -r requirements.txt

- Run the Flask app  
   python app.py

- Open browser  
   http://127.0.0.1:5000/

- Use POST /predict to get AQI predictions
