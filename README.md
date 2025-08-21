# AWS Cost Forecast

A machine learning project that predicts daily AWS costs using historical billing data and the CatBoost algorithm.

## Overview

This project analyzes AWS billing patterns and builds a predictive model to forecast daily costs. It uses features like day of the week, day of the month, month, and previous day's cost to make predictions.

## Features

- **Time-based Features**: Leverages weekday, day of month, and month patterns
- **Cost History**: Uses previous day's cost as a key predictor
- **CatBoost Algorithm**: Employs gradient boosting for accurate predictions
- **Model Evaluation**: Provides Mean Absolute Error and feature importance analysis
- **Sample Predictions**: Shows actual vs predicted costs for verification

## Requirements

```
pandas
catboost
scikit-learn
```

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install pandas catboost scikit-learn
   ```

## Usage

1. Ensure your billing data is in `billing.csv` with columns:
   - `usage_date`: Date in YYYY-MM-DD format
   - `daily_cost`: Daily AWS cost in dollars

2. Run the model:
   ```bash
   python model.py
   ```

## Output

The script provides:

- **Mean Absolute Error**: Overall prediction accuracy
- **Feature Importances**: Which factors most influence cost predictions
- **Sample Predictions**: 5 test cases showing:
  - Actual cost vs predicted cost
  - Absolute error and percentage error


