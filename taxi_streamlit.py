# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image


# Load trained model and feature order
with open("tuned_xgboost_model.pkl", "rb") as f:
    loaded_model, feature_order = pickle.load(f)
    
# Load data for EDA
data = pd.read_csv("C:\\Users\\Appu\\Desktop\\data science\\eda_data.csv")

st.set_page_config(layout="wide")
st.title("🚖Taxi Fare Prediction & EDA Dashboard")

option = st.sidebar.selectbox("Choose a section:", ["EDA","Regression Model","Predict Fare"])

# ===============================
# 📊 EDA SECTION
# ===============================

if option == "EDA":
    
# ===============================
    
    st.title("🔍 Exploratory Data Analysis")
    st.subheader("📈 Distance vs Total Fare")
    
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\fare_vs_distance.png"
    image = Image.open(image_path)
    st.image(image, caption='Distance vs Total Fare', use_column_width=True)
    st.write("""Each point represents one ride — plotted by how long the trip lasted versus how much the passenger paid in total.

    A positive trend (points rising to the right) suggests longer trips tend to cost more — which is expected.
    
    The spread around the trend indicates variability — perhaps due to differences in route efficiency, surge pricing, tolls, or tips.
    
    You might notice clusters (e.g., short trips under ~5 minutes all costing a few dollars) and outliers (e.g., short trips with high total amount, 
                                                                                                           or long trips with unusually low fares).""")  
# ===============================

    st.header("Fare vs. Passenger Count")
    
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\Average Fare by Passenger Count.png"
    image = Image.open(image_path)
    st.image(image, caption='Average Fare by Passenger Count', use_column_width=True)
    st.write("""Bars show the average fare for each passenger group (e.g., 1, 2, 3, 4+ passengers).

    If the average fare is similar whether there's 1, 2, 3, or 4 passengers, it suggests that adding more passengers does not significantly increase the price. 
    This often occurs because fare is primarily charged per mile/minute, not per passenger.
    
    Fare structure: Taxi fares are based on distance and duration, not number of riders.
    
    Trip characteristics: More passengers often ride together in situations like short group outings—these tend to have low fares.
    
    Shared rides: In services where multiple passengers share half the fare, per-person fare drops even if total ride cost stays similar.""")
# ===============================
    
    st.subheader("🚦 Average Fare by Passenger Count (Rush vs Non-Rush Hour)")

    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\Average Fare by Passenger Count (Rush vs Non-Rush Hour).png"
    image = Image.open(image_path)
    st.image(image, caption='Average Fare by Passenger Count (Rush vs Non-Rush Hour)', use_column_width=True)
    st.write("""Fare is largely independent of passenger count The fare structure in most taxi systems is based on distance and duration,
             not the number of passengers—so it doesn’t vary much if 1 or 3 people ride together even in the rush hour.""")        
# ===============================
   
    st.header("Outlier Detection")
    
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\Outlier Detection.png"
    image = Image.open(image_path)
    st.image(image, caption='Outlier Detection', use_column_width=True)
    st.markdown("""Each subplot shows:

    The middle line = median

    The box = interquartile range (25th–75th percentile)

    Whiskers = typically up to 1.5× IQR

    Dots outside whiskers = potential outliers

Outlier identification

    Points outside the whiskers are rides with unusually long trips, durations, or total costs—worth investigating or trimming for modeling.

Distribution insights

    A tall box = large IQR = high variability.

    Short whiskers = fewer extreme values.

    Symmetric boxes suggest a nearly normal distribution; skewed boxes indicate skew.

    You can easily see trip_duration_min is most prone to outliers""")
# ===============================        
    st.header("After treatment")
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\aftertreatment.png"
    image = Image.open(image_path)
    st.image(image, caption='After treatment', use_column_width=True)
    
# ===============================    
    st.header("Fare Variation by Time of Day")
    
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\Fare Variation by Time of Day.png"
    image = Image.open(image_path)
    st.image(image, caption='Fare Variation by Time of Day', use_column_width=True)
    
# ===============================
    st.header("Average Fare by Month")
    
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\Average Fare by Month.png"
    image = Image.open(image_path)
    st.image(image, caption='Average Fare by Month', use_column_width=True)
    st.write("""February–March 2016: A Challenging Season for NYC Taxis

In early 2016, both Uber and Lyft introduced around 15% fare cuts in New York City—notably around January 28, 2016.
 (https://www.thestreet.com/opinion/uber-and-lyft-fare-cuts-hit-nyc-taxis-and-their-lenders-hard-13589200?)

This led to:

A 27% decline in yellow taxi trips from their peak levels.


February 2016 saw a 11.7% year-over-year drop in taxi ridership, with related farebox revenue down by 11% compared to February 2015.

In March 2016, taxi trips and fare revenues continued to fall by 8.5% and 7.6% respectively, even though weather conditions were more 
favorable compared to March 2015

The dip in average fare from Feb to March 2016 in your dataset likely reflects:

Competitive pricing actions by Uber/Lyft, which drew passengers away from taxis.

Decline in taxi usage, as yellow cab ridership and revenues fell in early 2016.

Lower average fares, influenced by fewer trips and less demand-driven (surge) pricing in that window.
we observed decline in fares between February and March 2016 appears to link clearly to industry-wide 
disruptions—notably the fare cuts by ride-hailing apps—leading to reduced taxi demand and lower average fares.
 This isn’t merely a seasonal dip, but part of a broader market shift that year""")
# ===============================  
  
  
    st.header("corelation between the columns")
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\Correlation.png"
    image = Image.open(image_path)
    st.image(image, caption='corelation between the columns', use_column_width=True)
    
# ===============================
    
    st.header("studying the distribution of trip distances and pickup hours")
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\Trip Distance Distribution.png"
    image = Image.open(image_path)
    st.image(image, caption='Trip Distance Distribution', use_column_width=True)
    
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\Pickup Hour Distribution.png"
    image = Image.open(image_path)
    st.image(image, caption='Pickup Hour Distribution', use_column_width=True)
    st.write("""Trip Distance: heavily skewed right, most trips < 4 miles, long tail present

             Pickup Hour Trends: spikes in morning/evening rush, lowest around 3–5 AM""")
# ===============================

    st.header("📊 Fare-per-minute/km by Hour of Day")

    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\Fare per km by Hour of Day.png"
    image = Image.open(image_path)
    st.image(image, caption='Fare-per-minute/km by Hour of Day', use_column_width=True)
    st.markdown("""
Fare per km

    Early morning (0–5h): Around $4–4.5 per km.

    Morning to early afternoon (6–12h): Peaks around $6–6.5 per km, likely due to typical base fares plus rush-hour/idle traffic factors, 
    as defined by NYC taxi tariff rules (e.g. $3 initial charge + per-mile rates + surcharges). 

    Late afternoon to evening (13–20h): Declines to around $4 per km; traffic-heavy periods reduce effective mileage earnings.

    Night (21–23h): Rises again slightly (~$4.2), influenced by overnight surcharges ($1/night) in place. 

Fare per minute

    Montoored low (~1–2 $/min) — fairly consistent and stable.

    Rises slightly at night (~2–2.5 $/min) — indicating slower traffic, with drivers earning more per minute due to waiting or slow speeds.""")
# ===============================    
    
    st.header("Trips by Pickup Hour")
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\Trip Counts by Pickup Hour.png"
    image = Image.open(image_path)
    st.image(image, caption='Trips by Pickup Hour', use_column_width=True)


# ===============================  
    

    
    st.header("Trips by Day of Week")
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\Trip Counts by Day of the Week.png"
    image = Image.open(image_path)
    st.image(image, caption='Trip Counts by Day of the Week', use_column_width=True)
    

# ===============================
    
    st.header("Impact of Night Rides vs. Day Rides")
    
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\saved_plots\\Night vs Day.png"
    image = Image.open(image_path)
    st.image(image, caption='Fare Comparison: Night vs Day', use_column_width=True)
    st.markdown("""Observations from the Plot

Similar Medians

The median fare is roughly the same for both day and night—around $10–$12—indicating that typical trip fares don't vary much by time of day.
Greater Fare Variability at Night

The night-time box is taller, and the whiskers extend higher, indicating a wider spread of fares at night. 
This suggests that while most night fares are typical, there's a higher chance of encountering extreme fares—likely
 due to night surcharges, longer rides, or more traffic delays.
More High-End Outliers at Night

There are more night fares exceeding $200–$250 compared to day rides. This implies rare but costly trips happen more often at night.""")
# ===============================

    
elif option == "Regression Model":
    st.header("Comparison of regression models")
    
    st.subheader("Model Evaluation Metrics")
    # Load the CSV file
    df_results = pd.read_csv("C:\\Users\\Appu\\Desktop\\data science\\model_results.csv", index_col=0)
    
    # Display the table
    st.dataframe(df_results)
    
    # Optional: Show chart (e.g., R2 Score comparison)
    st.subheader("📊 R2 Score Comparison")
    st.bar_chart(df_results["R2 Score"])  

    st.subheader("got best result in XGBRegressor, so did GridSearchCV") 
    st.subheader("""Best Parameters: {
        'learning_rate': 0.1,
        'max_depth': 7, 
        'n_estimators': 200, 
        'subsample': 1}
        R2 Score: 0.9864508724679674""") 

              
    
# ===============================
# 💡 PREDICTION SECTION
# ===============================

elif option == "Predict Fare":
    st.header("💡 Predict Taxi Fare")
    st.markdown("Enter the ride details below to predict the **total fare amount**.")
    
    # --- Streamlit Input Widgets ---
    trip_duration = st.number_input("Trip duration (minutes)", min_value=1.0, format="%.2f")
    tip_amount = st.number_input("Tip amount ($)", min_value=0.0, format="%.2f")
    trip_distance = st.number_input("Trip distance (km)", min_value=0.1, format="%.2f")
    is_night = st.radio("Is it night?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    payment_type = st.selectbox("Payment type", options=[1, 2, 3], format_func=lambda x: {
        1: "Credit",
        2: "Cash",
        3: "No Charge"
    }[x])
    
    # --- Prediction Button ---
    if st.button("Predict Fare"):
    
        # Log transformations (same as training)
        log_trip_duration = np.log1p(trip_duration)
        log_tip_amount = np.log1p(tip_amount)
        log_trip_distance = np.log1p(trip_distance)
    
        # One-hot encode payment type
        payment_type_2 = 1 if payment_type == 2 else 0
        payment_type_3 = 1 if payment_type == 3 else 0
    
        # Create DataFrame
        input_data = pd.DataFrame([{
            'trip_duration': log_trip_duration,
            'trip_distance': log_trip_distance,
            'is_night': is_night,
            'payment_type_2': payment_type_2,
            'payment_type_3': payment_type_3,
            'tip_amount': log_tip_amount,
        }])
        # ✅ Ensure correct feature order
        feature_order = ['trip_duration','trip_distance','is_night', 'payment_type_2', 'payment_type_3','tip_amount']
        input_data = input_data[feature_order]
    
        # Make prediction
        log_pred = loaded_model.predict(input_data)
        predicted_total = np.expm1(log_pred[0])
    
        # Display result
        st.success(f"💰 Predicted Total Fare Amount: **${predicted_total:.2f}**")
        
