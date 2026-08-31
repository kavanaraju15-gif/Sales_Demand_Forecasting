import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Sales & Demand Forecasting",
    page_icon="📈",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("📈 AI-Based Sales & Demand Forecasting")

st.write(
    "This application uses Machine Learning to analyze "
    "historical sales and predict future demand."
)


# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("sales_data.csv")

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values("Date")


# ==========================================
# FEATURE ENGINEERING
# ==========================================

data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data["DayOfWeek"] = data["Date"].dt.dayofweek

data["Previous_Day_Sales"] = data["Sales"].shift(1)
data["Previous_Week_Sales"] = data["Sales"].shift(7)

data = data.dropna()


# ==========================================
# FEATURES
# ==========================================

features = [
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "Previous_Day_Sales",
    "Previous_Week_Sales"
]

X = data[features]
y = data["Sales"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# ==========================================
# TRAIN MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# MODEL METRICS
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)


# ==========================================
# DASHBOARD METRICS
# ==========================================

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", round(mae, 2))
col2.metric("RMSE", round(rmse, 2))
col3.metric("R² Score", round(r2, 2))


# ==========================================
# HISTORICAL SALES
# ==========================================

st.subheader("📈 Historical Sales")

st.line_chart(
    data.set_index("Date")["Sales"]
)


# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

st.subheader("🤖 Actual vs Predicted Sales")

comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions
})

st.line_chart(comparison)


# ==========================================
# FUTURE SALES PREDICTION
# ==========================================

st.subheader("🔮 Future Sales Prediction")

future_date = st.date_input(
    "Select a future date"
)


# Convert selected date
future_date = pd.to_datetime(future_date)


# Find previous sales values
previous_day_sales = data["Sales"].iloc[-1]

previous_week_sales = data["Sales"].iloc[-7]


# Create future features
future_features = pd.DataFrame({
    "Year": [future_date.year],
    "Month": [future_date.month],
    "Day": [future_date.day],
    "DayOfWeek": [future_date.dayofweek],
    "Previous_Day_Sales": [previous_day_sales],
    "Previous_Week_Sales": [previous_week_sales]
})


# Predict
future_prediction = model.predict(
    future_features
)


# Display prediction
st.success(
    f"Predicted Sales for {future_date.strftime('%Y-%m-%d')}: "
    f"{future_prediction[0]:.2f}"
)