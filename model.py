import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('billing.csv')

# Removing the first row with 0 cost 
df = df[df['daily_cost'] != 0].copy()

# Converting usage_date to datetime
df['usage_date'] = pd.to_datetime(df['usage_date'])

# Adding more features
df['weekday'] = df['usage_date'].dt.dayofweek 
df['day_of_month'] = df['usage_date'].dt.day
df['month'] = df['usage_date'].dt.month
df['prev_day_cost'] = df['daily_cost'].shift(1)

# Removing the first row after shift which has NaN
df = df.dropna().copy()


features = ['weekday', 'day_of_month', 'month', 'prev_day_cost']
target = 'daily_cost'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initializing CatBoost model
model = CatBoostRegressor(verbose=0, random_state=42)

model.fit(X_train, y_train, cat_features=['weekday', 'month'])

predictions = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, predictions)

# Output results
print(f"Mean Absolute Error: {mae:.2f}")
print("Feature Importances:")
for name, importance in zip(features, model.feature_importances_):
    print(f"{name}: {importance:.2f}")


# Displaying sample predictions for review
print("\nSample Predictions:")
print("-" * 40)

num_samples = min(5, len(X_test))
for i in range(num_samples):
    actual_cost = y_test.iloc[i]
    predicted_cost = predictions[i]
    error = abs(actual_cost - predicted_cost)
    error_percentage = (error / actual_cost) * 100
    
    print(f"\n• Sample {i+1}:")
    print(f"  Actual: ${actual_cost:.2f}")
    print(f"  Predicted: ${predicted_cost:.2f}")
    print(f"  Error: ${error:.2f} ({error_percentage:.1f}%)")
