# TripFare-Predicting-Urban-Taxi-Fare-with-Machine-Learning
# 🚕 NYC Taxi Fare Prediction

This project focuses on predicting taxi fares in New York City using machine learning regression models. The goal is to analyze fare patterns and build an optimized model to predict fare amounts based on ride characteristics.

## 📂 Project Structure

- `project3.ipynb`: Initial EDA, data preprocessing, and feature engineering.
- `taxi_fare_regression.ipynb`: Model training, hyperparameter tuning, and evaluation.

## 📊 Features Used

Some of the key features engineered and used for modeling include:

- Pickup and dropoff datetime
- Trip duration (minutes)
- Trip distance (km or miles)
- Fare per km, fare per minute
- Hour of day, AM/PM, rush hour indicators
- Payment type
- Passenger count

## 🧪 Exploratory Data Analysis (EDA)

Performed thorough EDA including:

- Distribution of fare amounts, trip distances, and durations
- Outlier detection and handling
- Bivariate analysis (e.g., fare vs. distance, fare vs. passenger count)
- Time-based patterns (rush hour, night trips)

## 🛠️ Machine Learning Models

Models explored and tuned:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor (final model)
- Lasso Regression
- Ridge regression
- KNeighborsRegressor

### ✔️ Model Evaluation Metrics

- R² Score
- MAE
- MSE
- RMSE (Root Mean Squared Error)

## 🔧 Hyperparameter Tuning

Used `GridSearchCV` to find the best combination of hyperparameters for the XGBoost model.

##Used Streamlit to present the model
![image](https://github.com/user-attachments/assets/1be3e7e7-d2fb-4efa-9fed-c7ddee0789ab)

