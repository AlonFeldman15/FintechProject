import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Combine the relevant 'Close' and 'Open' columns into a single DataFrame
data = pd.DataFrame({
    'Crude_Oil_Close': data_frames['Crude_Oil']['Close'],
    'SP500_Close': data_frames['SP500']['Close'],
    'USD_to_Euro_Close': data_frames['USD_to_Euro']['Close'],
    'Natural_Gas_Close': data_frames['Natural_Gas']['Close'],
    'Crude_Oil_Open': data_frames['Crude_Oil']['Open'],
    'SP500_Open': data_frames['SP500']['Open'],
    'USD_to_Euro_Open': data_frames['USD_to_Euro']['Open'],
    'Natural_Gas_Open': data_frames['Natural_Gas']['Open']
})

# Handle missing data by forward filling and then dropping any remaining NaNs
data = data.ffill().dropna()

# Define the number of lags to use
num_lags = 5  # You can choose a different number of lags based on model performance

# Create lagged features
for i in range(1, num_lags + 1):
    data[f'Crude_Oil_Close_Lag_{i}'] = data['Crude_Oil_Close'].shift(i)
    data[f'SP500_Close_Lag_{i}'] = data['SP500_Close'].shift(i)
    data[f'USD_to_Euro_Close_Lag_{i}'] = data['USD_to_Euro_Close'].shift(i)
    data[f'Crude_Oil_Open_Lag_{i}'] = data['Crude_Oil_Open'].shift(i)
    data[f'SP500_Open_Lag_{i}'] = data['SP500_Open'].shift(i)
    data[f'USD_to_Euro_Open_Lag_{i}'] = data['USD_to_Euro_Open'].shift(i)

# Drop rows with NaN values (which will be created by lagging)
data = data.dropna()

# Separate features and target
X = data.drop(columns=['Natural_Gas_Close'])
y = data['Natural_Gas_Close']

# Split data into train and test sets
train_size = int(len(data) * 0.9)  # 90% training data
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Standardize the features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Combine training data into a single DataFrame for VAR model
train_data = pd.DataFrame(X_train_scaled, columns=X.columns)
train_data['Natural_Gas_Close'] = y_train_scaled

# Fit the VAR model
model = VAR(train_data)
lag_order = model.select_order(maxlags=15)
var_model = model.fit(lag_order.aic)

# Forecast the values for the test set
test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
test_data['Natural_Gas_Close'] = np.nan  # Placeholder for actual values

forecast_scaled = var_model.forecast(train_data.values[-var_model.k_ar:], steps=len(X_test))

# Convert forecasts back to original scale
forecast_unscaled = pd.DataFrame(scaler_y.inverse_transform(forecast_scaled[:, -1].reshape(-1, 1)),
                                 index=y_test.index,
                                 columns=['Natural_Gas'])

# Calculate and print performance metrics
mae = mean_absolute_error(y_test, forecast_unscaled)
rmse = np.sqrt(mean_squared_error(y_test, forecast_unscaled))
r2 = r2_score(y_test, forecast_unscaled)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")

# Calculate returns and volatility
true_returns = np.log(y_test / y_test.shift(1)).dropna()
predicted_returns = np.log(forecast_unscaled['Natural_Gas'] / forecast_unscaled['Natural_Gas'].shift(1)).dropna()

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
