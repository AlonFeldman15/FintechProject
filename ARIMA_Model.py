import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def get_business_day_data(ticker_symbol, start_date, end_date):
    """
    Fetches historical data for the given ticker symbol from Yahoo Finance,
    ensuring that only business days are considered.
    """
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    data = data.asfreq('B').dropna()  # Keep only business days
    return data

# Define the ticker symbols for the required data
symbols = {
    'Crude_Oil': 'CL=F',
    'SP500': '^GSPC',
    'USD_to_Euro': 'EURUSD=X',
    'Natural_Gas': 'NG=F'
}

# Define the date range
start_date = '2013-01-01'
end_date = date.today()

# Fetch data for each ticker symbol and keep only business days
data_frames = {}
for key, symbol in symbols.items():
    data_frames[key] = get_business_day_data(symbol, start_date, end_date)

# Combine the relevant 'Close' columns into a single DataFrame
data = pd.DataFrame({
    'Crude_Oil_Close': data_frames['Crude_Oil']['Close'],
    'SP500_Close': data_frames['SP500']['Close'],
    'USD_to_Euro_Close': data_frames['USD_to_Euro']['Close'],
    'Natural_Gas_Close': data_frames['Natural_Gas']['Close'],
})

# Handle missing data by forward filling and then dropping any remaining NaNs
data = data.ffill().dropna()

# Define features and target (only using close prices for ARIMA)
X = data.drop(columns=['Natural_Gas_Close'])
y = data['Natural_Gas_Close']

# Split data into train and test sets
train_size = int(len(data) * 0.9)  # 90% training data
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Standardize the target
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Fit the ARIMA model on the training set
arima_order = (5, 1, 0)  # Example ARIMA order, you can optimize this
arima_model = ARIMA(y_train_scaled, order=arima_order)
arima_model_fit = arima_model.fit()

# Forecast the values for the test set
forecast_scaled = arima_model_fit.forecast(steps=len(y_test))

# Convert forecasts back to original scale
forecast_unscaled = pd.Series(scaler_y.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten(),
                              index=y_test.index)

# Calculate and print performance metrics
mae = mean_absolute_error(y_test, forecast_unscaled)
rmse = np.sqrt(mean_squared_error(y_test, forecast_unscaled))
r2 = r2_score(y_test, forecast_unscaled)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")

# Calculate returns and volatility
true_returns = np.log(y_test / y_test.shift(1)).dropna()
predicted_returns = np.log(forecast_unscaled / forecast_unscaled.shift(1)).dropna()

true_volatility = true_returns.std()
predicted_volatility = predicted_returns.std()

print(f"True Volatility: {true_volatility}")
print(f"Predicted Volatility: {predicted_volatility}")

# Calculate Value at Risk (VaR)
confidence_level = 0.95
VaR_true = np.percentile(true_returns, (1 - confidence_level) * 100)
VaR_predicted = np.percentile(predicted_returns, (1 - confidence_level) * 100)

print(f"True Value at Risk (VaR): {VaR_true}")
print(f"Predicted Value at Risk (VaR): {VaR_predicted}")