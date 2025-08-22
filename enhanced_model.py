import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('billing.csv')

df = df[df['daily_cost'] != 0].copy()

df['usage_date'] = pd.to_datetime(df['usage_date'])

df = df.sort_values('usage_date').reset_index(drop=True)


# Basic time features
df['weekday'] = df['usage_date'].dt.dayofweek 
df['day_of_month'] = df['usage_date'].dt.day
df['month'] = df['usage_date'].dt.month
df['quarter'] = df['usage_date'].dt.quarter
df['is_weekend'] = (df['weekday'] >= 5).astype(int)
df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)

# Multiple lag features for better historical context
for lag in [1, 2, 3, 7]:
    df[f'cost_lag_{lag}'] = df['daily_cost'].shift(lag)

# Rolling statistics for trend analysis
for window in [3, 7, 14, 30]:
    df[f'cost_avg_{window}d'] = df['daily_cost'].rolling(window=window).mean()
    df[f'cost_std_{window}d'] = df['daily_cost'].rolling(window=window).std()
    df[f'cost_max_{window}d'] = df['daily_cost'].rolling(window=window).max()
    df[f'cost_min_{window}d'] = df['daily_cost'].rolling(window=window).min()

# Trend indicators
df['cost_trend_7d'] = df['daily_cost'].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
df['cost_change_1d'] = df['daily_cost'].pct_change(1)
df['cost_change_7d'] = df['daily_cost'].pct_change(7)

# Cyclical encoding for better periodic pattern capture
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Remove rows with NaN values
df = df.dropna().copy()

# Smart Feature Selection
categorical_features = ['weekday', 'month', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end']
numerical_features = [col for col in df.columns if col.startswith(('cost_lag_', 'cost_avg_', 'cost_std_', 
                                                                   'cost_max_', 'cost_min_', 'cost_trend_', 
                                                                   'cost_change_'))]
cyclical_features = [col for col in df.columns if col.endswith(('_sin', '_cos'))]
basic_features = ['day_of_month']

features = categorical_features + numerical_features + cyclical_features + basic_features
print(f"Total features created: {len(features)}")

X = df[features]
y = df['daily_cost']

# Time-preserving split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Hyperparameter Tuning
param_grid = {
    'iterations': [300, 500, 800],
    'depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'l2_leaf_reg': [1, 3, 5]
}

# Use TimeSeriesSplit for proper time series validation
tscv = TimeSeriesSplit(n_splits=3)

model = CatBoostRegressor(verbose=0, random_state=42)

# Grid search with time series CV
grid_search = GridSearchCV(
    model, param_grid, cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train, cat_features=categorical_features)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# OPTIMIZATION 4: Comprehensive Evaluation
predictions = best_model.predict(X_test)

# Multiple evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

# Results
print("\n" + "="*60)
print("OPTIMIZED MODEL PERFORMANCE")
print("="*60)
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Square Error (RMSE): ${rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Feature importance analysis
feature_importance = dict(zip(features, best_model.feature_importances_))
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)


# Sample predictions with detailed analysis
print(f"\nDetailed Sample Predictions:")
print("-" * 50)

for i in range(min(5, len(X_test))):
    actual = y_test.iloc[i]
    predicted = predictions[i]
    error = abs(actual - predicted)
    error_pct = (error / actual) * 100
    
    print(f"\n• Sample {i+1}:")
    print(f"  Actual: ${actual:.2f}")
    print(f"  Predicted: ${predicted:.2f}")
    print(f"  Error: ${error:.2f} ({error_pct:.1f}%)")