import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("sales_data.csv")

# Convert Date to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Sort by date
data = data.sort_values("Date")


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

# Date features
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data["DayOfWeek"] = data["Date"].dt.dayofweek

# Previous day's sales
data["Previous_Day_Sales"] = data["Sales"].shift(1)

# Sales from 7 days ago
data["Previous_Week_Sales"] = data["Sales"].shift(7)


# ==========================================
# 3. REMOVE MISSING VALUES
# ==========================================

data = data.dropna()


# ==========================================
# 4. SELECT FEATURES
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
# 5. TRAIN-TEST SPLIT
# ==========================================

# Use the first 80% for training
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# ==========================================
# 6. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("Model training completed successfully!")


# ==========================================
# 7. PREDICTIONS
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)


print("\n========== MODEL EVALUATION ==========")

print("MAE      :", round(mae, 2))
print("MSE      :", round(mse, 2))
print("RMSE     :", round(rmse, 2))
print("R2 Score :", round(r2, 2))


# ==========================================
# 9. ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    data["Date"].iloc[split_index:],
    y_test,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    data["Date"].iloc[split_index:],
    predictions,
    marker="o",
    label="Predicted Sales"
)

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Actual vs Predicted Sales")

plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()