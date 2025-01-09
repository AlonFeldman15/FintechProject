# Natural Gas Price Forecasting - Fintech Project 

## Overview
This project focuses on forecasting the opening price of natural gas in the stock market. 
Natural gas prices are a key component of the global energy market, impacting traders, investors, and energy companies. 
Accurate price predictions can influence investment decisions and improve understanding of market dynamics.

## Market
The energy market includes traditional sources like crude oil and natural gas, renewable sources like wind and solar, and nuclear energy. 
Price fluctuations in this market affect global economies, government policies, financial markets, industries, transportation, and the environment.

## Features
The models utilize key features influencing the energy market:
- **Crude Oil Price:** As a substitute for natural gas, crude oil price changes can impact natural gas prices.
- **USD Exchange Rate:** Energy prices are traded in USD, making exchange rate fluctuations significant.
- **S&P 500 Index:** An indicator of overall market confidence and economic conditions.
- **Weather:** Extreme temperatures and winter season influence demand for natural gas. Additionally, weather conditions in natural gas-exporting and processing regions impact prices.

## Timeframe
Daily predictions for the stock market's opening price of natural gas.

## Data Sources
- [Yahoo Finance](https://finance.yahoo.com/)
- [Meteostat](https://meteostat.net/)

## Data Engineering
- **Data Cleaning:** Removal of missing values.
- **Normalization:** Standardizing features to comparable scales.
- **Lag Features:** Utilizing historical data for prediction.
- **Trading Days Only:** Analysis focuses solely on stock market trading days.

## Forecasting Models
1. **VAR:** Captures relationships between multiple variables over time.
2. **Linear Regression:** A basic model relying on linear relationships.
3. **ARIMA:** Analyzes and predicts time series data.
4. **ARIMA with Weather:** Includes weather data from selected U.S. cities like Houston (export/processing hub) and NYC (high consumption).

## Evaluation Metrics
- **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values.
- **Root Mean Squared Error (RMSE):** Square root of the mean squared errors; sensitive to large errors.
- **R-squared (R²):** Measures how well the model explains the variance. A positive R² indicates better performance.
- **Volatility:** Standard deviation of returns.
- **Value at Risk (VaR):** Estimates potential financial risk.
- **Sharpe Ratio:** Evaluates return per unit of risk.
- **Sortino Ratio:** Focuses on downside risk-adjusted returns.

## Results
1. **VAR:** Low R² and poor overall performance.
   - **MAE:** 1.48  
   - **RMSE:** 1.67  
   - **R² Score:** -11.49  

2. **Linear Regression:** Lower performance as expected.
   - **MAE:** 2.25  
   - **RMSE:** 2.32  
   - **R² Score:** -23.24  

3. **ARIMA:** Best results among the models, though R² is negative.
   - **MAE:** 0.45  
   - **RMSE:** 0.57  
   - **R² Score:** -0.48  

4. **ARIMA with Weather:** Improved results with positive R².
   - **MAE:** 0.14  
   - **RMSE:** 0.17  
   - **R² Score:** 0.87

   Graph For ARIMA with Weather:
![image](https://github.com/user-attachments/assets/7dbd695e-b605-4401-accb-ef3fb6c16bfb)

The addition of weather data improved performance and reduced reliance on the S&P 500 index as a demand indicator.

---
## Conclusion
The results from the final model, which combines ARIMA with weather data, show promising performance. 
We achieved low error metrics, indicating the model handles outliers reasonably well. 
The Root Mean Squared Error (RMSE) is slightly higher, suggesting that large errors occur infrequently.
The R² value close to 1 (86.5%) indicates that the model explains most of the variance in the data, a significant improvement compared to models without weather data, which showed negative R² values.
The predicted volatility is slightly lower than the actual volatility, implying that the model may slightly underestimate risk. However, the close alignment suggests the model captures the general volatility trend well.
The predicted Value at Risk (VaR) is less negative than the actual VaR, indicating the model provides a slightly conservative risk estimate (lower extreme loss forecast). 
While this is not ideal for precise risk management, the forecast remains reasonable and cautious.
The Sharpe and Sortino ratios are both positive, with the Sortino ratio slightly higher, suggesting the model performs better when considering only negative volatility. 
However, empirical results are lower than theoretical values, which may imply that the actual returns, adjusted for volatility, could be lower under real market conditions. 
This could be due to the model’s conservative risk estimates or external factors not accounted for in the model.
Overall, the model provides a cautious yet reasonable forecast, balancing accuracy and risk estimation, with some room for improvement in capturing extreme events and real market conditions.

## Code
The repository includes:
- Scripts for each model (VAR, Linear Regression, ARIMA, ARIMA with Weather).
- Historical data from 01/01/2013 to the present.

## Contact
**Alon Feldman**  
Email: alon.feldman5@gmail.com  
Linkedin: [Alon Feldman](https://www.linkedin.com/in/alon-feldman5)  
Special Thanks: [Hagit Lev Feldman](https://github.com/hagitLev)  
Technion - Spring 2024  

