import pandas as pd
import numpy as np
from datetime import date, datetime
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from meteostat import Stations, Daily
from geopy.geocoders import Nominatim
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt

def get_business_day_data(ticker_symbol, start_date, end_date):
    """
    Fetches historical data for the given ticker symbol from Yahoo Finance,
    ensuring that only business days are considered.
    """
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    data = data.asfreq('B').dropna()  # Keep only business days
    return data

def get_weather_data(states, start_date, end_date):
    """
    Fetches historical weather data for the specified states
    between start_date and end_date using the Meteostat library.
    Aggregates the data by averaging across selected stations in each state.
    """
    # Initialize geolocator
    geolocator = Nominatim(user_agent="natural_gas_weather_app")

    # Define a dictionary to hold state-wise average weather data
    weather_data = pd.DataFrame()

    for state, city in states.items():
        print(f"Fetching weather data for {state} ({city})")
        # Geocode the city to get latitude and longitude
        location = geolocator.geocode(city + ", " + state)
        if not location:
            print(f"Could not geocode city: {city}, {state}")
            continue
        lat, lon = location.latitude, location.longitude

        # Find nearby weather stations
        stations = Stations()
        stations = stations.nearby(lat, lon)
        stations = stations.inventory('daily', (datetime.strptime(start_date, "%Y-%m-%d"),
                                               datetime.strptime(end_date, "%Y-%m-%d")))
        station = stations.fetch(1)

        if station.empty:
            print(f"No weather stations found for {city}, {state}")
            continue

        station_id = station.index[0]

        # Fetch daily weather data
        data = Daily(station_id, start=datetime.strptime(start_date, "%Y-%m-%d"),
                    end=datetime.strptime(end_date, "%Y-%m-%d"))
        data = data.fetch()

        if data.empty:
            print(f"No weather data found for station {station_id} in {state}")
            continue

        # Define the required columns
        required_columns = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres', 'rhum']

        # Check which required columns are present
        present_columns = [col for col in required_columns if col in data.columns]
        if not present_columns:
            print(f"No required weather data available for station {station_id} in {state}")
            continue

        # Select relevant columns
        selected_data = data[present_columns].copy()

        # Rename all present columns by prefixing with the state name to ensure uniqueness
        rename_dict = {col: f'{state}_{col}' for col in present_columns}
        selected_data = selected_data.rename(columns=rename_dict)

        # Handle missing data
        selected_data = selected_data.fillna(method='ffill').fillna(method='bfill')

        # Aggregate by date
        if weather_data.empty:
            weather_data = selected_data.copy()
        else:
            weather_data = weather_data.join(
                selected_data, how='outer'
            )

    # After fetching all states, aggregate the features by taking the mean across states
    # Identify averaged features based on renamed columns
    temp_cols = [col for col in weather_data.columns if col.endswith('_tavg') or col.endswith('_tmin') or col.endswith('_tmax')]
    precip_cols = [col for col in weather_data.columns if col.endswith('_prcp')]
    humid_cols = [col for col in weather_data.columns if col.endswith('_rhum')]
    wspd_cols = [col for col in weather_data.columns if col.endswith('_wspd')]
    pres_cols = [col for col in weather_data.columns if col.endswith('_pres')]

    # Calculate average for each feature category if columns exist
    if temp_cols:
        weather_data['Avg_Temp'] = weather_data[temp_cols].mean(axis=1)
    if precip_cols:
        weather_data['Avg_Precipitation'] = weather_data[precip_cols].mean(axis=1)
    if humid_cols:
        weather_data['Avg_Humidity'] = weather_data[humid_cols].mean(axis=1)
    if wspd_cols:
        weather_data['Avg_Wind_Speed'] = weather_data[wspd_cols].mean(axis=1)
    if pres_cols:
        weather_data['Avg_Pressure'] = weather_data[pres_cols].mean(axis=1)

    # Keep only the averaged features that were created
    averaged_features = []
    if temp_cols:
        averaged_features.append('Avg_Temp')
    if precip_cols:
        averaged_features.append('Avg_Precipitation')
    if humid_cols:
        averaged_features.append('Avg_Humidity')
    if wspd_cols:
        averaged_features.append('Avg_Wind_Speed')
    if pres_cols:
        averaged_features.append('Avg_Pressure')

    weather_data = weather_data[averaged_features]

    # Handle any remaining missing data
    weather_data = weather_data.fillna(method='ffill').fillna(method='bfill')
    return weather_data

# Define the ticker symbols for the required data
symbols = {
    'Crude_Oil': 'CL=F',
    # 'SP500': '^GSPC',
    'USD_to_Euro': 'EURUSD=X',
    'Natural_Gas': 'NG=F'
}

# Define the date range
start_date = '2013-01-01'
end_date = date.today().strftime("%Y-%m-%d")

# Fetch financial data for each ticker symbol and keep only business days
data_frames = {}
for key, symbol in symbols.items():
    print(f"Downloading data for {key} ({symbol})")
    data_frames[key] = get_business_day_data(symbol, start_date, end_date)

# Combine the relevant 'Close' columns into a single DataFrame
data = pd.DataFrame({
    'Crude_Oil_Close': data_frames['Crude_Oil']['Close'],
    # 'SP500_Close': data_frames['SP500']['Close'],
    'USD_to_Euro_Close': data_frames['USD_to_Euro']['Close'],
    'Natural_Gas_Close': data_frames['Natural_Gas']['Close'],
    'Natural_Gas_Open': data_frames['Natural_Gas']['Open'],
})

# Handle missing data by forward filling and then dropping any remaining NaNs
data = data.ffill().dropna()

# Define the states and their representative cities
states = {
    'Texas': 'Houston',          # Major natural gas production and export hub
    'Louisiana': 'New Orleans',  # Significant natural gas production and trading
    'Oklahoma': 'Oklahoma City',  # Important production area
    'New York': 'New York City',  # Major consumer market
    'Pennsylvania': 'Pittsburgh', # Key region for Marcellus Shale gas production
    'Colorado': 'Denver',         # Important gas production state
    'Alaska': 'Anchorage',        # Major supplier of natural gas
    'California': 'Los Angeles',  # Large consumer market with import facilities
    'Illinois': 'Chicago',        # Major market for natural gas
    'Ohio': 'Cleveland',          # Emerging natural gas production area
    'West Virginia': 'Charleston', # Significant producer in the Appalachian region
    'Kentucky': 'Louisville',     # Natural gas production and consumption
    'Florida': 'Miami',           # Major consumer with import facilities
    'Alabama': 'Birmingham',      # Consumer and transporter
    'Michigan': 'Detroit',        # Significant natural gas consumer
    'Maryland': 'Baltimore',      # Consumption and distribution hub
    'Virginia': 'Richmond',       # Important for gas distribution
    'Tennessee': 'Nashville'      # Consumption and transportation hub
    # Add more states and cities as needed
}


# Fetch weather data
weather_data = get_weather_data(states, start_date, end_date)

# Merge weather data with financial data
# Ensure that the weather_data index is in datetime format and matches the financial data index
weather_data.index = pd.to_datetime(weather_data.index)
data.index = pd.to_datetime(data.index)

# Resample weather data to business days by forward filling
weather_data = weather_data.asfreq('B').fillna(method='ffill')

# Merge on the index (dates)
data = data.join(weather_data, how='left')

# Handle any missing weather data
data = data.fillna(method='ffill').fillna(method='bfill')

# Define features and target
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

# Standardize the target
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Fit the ARIMA model on the training set with exogenous variables
arima_order = (4, 1, 0)  # Example ARIMA order, you can optimize this
arima_model = ARIMA(y_train_scaled, order=arima_order, exog=X_train_scaled)
arima_model_fit = arima_model.fit()
# Forecast the values for the test set
forecast_scaled = arima_model_fit.forecast(steps=len(y_test), exog=X_test_scaled)

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

# Bonus:
# Existing imports...

# Additional imports for trading strategy calculations
from scipy.stats import norm


# Define trading strategy based on forecast
def trading_strategy(predicted_prices, actual_prices):
    """
    Implements a simple trading strategy based on the forecasted natural gas prices.
    Buy if the forecast price is higher than the current price, sell otherwise.
    """
    # Generate signals: 1 for buy, -1 for sell, 0 for hold
    signals = np.where(predicted_prices > actual_prices.shift(1), 1, -1)

    # Calculate returns based on signals
    daily_returns = actual_prices.pct_change().shift(-1)  # Shift to align with signals
    strategy_returns = signals * daily_returns  # Multiply signals by returns

    return strategy_returns.dropna()


# Calculate risk-return metrics
def calculate_metrics(strategy_returns, risk_free_rate=0.0):
    """
    Calculates performance metrics for the trading strategy.
    """
    mean_return = strategy_returns.mean()  # Average return
    volatility = strategy_returns.std()  # Standard deviation of returns

    # Calculate Sharpe Ratio
    sharpe_ratio = (mean_return - risk_free_rate) / volatility

    # Calculate Sortino Ratio
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_volatility = downside_returns.std()
    sortino_ratio = (mean_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else np.nan

    return sharpe_ratio, sortino_ratio


# Create a DataFrame for actual prices and predicted prices
predicted_prices = forecast_unscaled
actual_prices = y_test

# Generate strategy returns
strategy_returns = trading_strategy(predicted_prices, actual_prices)

# Calculate risk-return metrics
sharpe_ratio, sortino_ratio = calculate_metrics(strategy_returns)

# Print results
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Sortino Ratio: {sortino_ratio}")

# Compare empirical values
empirical_returns = y_test.pct_change().dropna()
empirical_sharpe_ratio, empirical_sortino_ratio = calculate_metrics(empirical_returns)

print(f"Empirical Sharpe Ratio: {empirical_sharpe_ratio}")
print(f"Empirical Sortino Ratio: {empirical_sortino_ratio}")

# Graph Actual vs Predicted Natural Gas Prices in test-set

# Plot actual vs predicted prices over time
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(forecast_unscaled.index, forecast_unscaled, label='Predicted Prices', color='orange', linestyle='--')

# Formatting the x-axis to show months and years
plt.xlabel('Date')
plt.ylabel('Natural Gas Prices')
plt.title('Actual vs Predicted Natural Gas Prices in test-set')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()

# Display the plot
plt.show()


