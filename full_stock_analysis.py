import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and Prepare Data
df = pd.read_csv('stocks.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date'])

print('--- Data Info ---')
print(df.info())
print('\n--- Null Values ---')
print(df.isnull().sum())

# 2. EDA: Visualize closing prices and volume
plt.figure(figsize=(12, 6))
for ticker in df['Ticker'].unique():
    plt.plot(df[df['Ticker'] == ticker]['Date'], df[df['Ticker'] == ticker]['Close'], label=ticker)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Closing Prices Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for ticker in df['Ticker'].unique():
    plt.plot(df[df['Ticker'] == ticker]['Date'], df[df['Ticker'] == ticker]['Volume'], label=ticker)
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Volume Traded Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Feature Engineering: Moving averages and volatility
def add_rolling_features(group, windows=[7, 30]):
    for window in windows:
        group[f'MA_{window}'] = group['Close'].rolling(window).mean()
        group[f'Volatility_{window}'] = group['Close'].rolling(window).std()
    return group

df = df.groupby('Ticker').apply(add_rolling_features)

# 4. Correlation analysis
pivot_close = df.pivot(index='Date', columns='Ticker', values='Close')
correlation_matrix = pivot_close.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Closing Prices')
plt.show()

# 5. Predictive Modeling (example: predict next day's close for AAPL)
aapl = df[df['Ticker'] == 'AAPL'].copy()
aapl['Close_next'] = aapl['Close'].shift(-1)
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_7', 'MA_30', 'Volatility_7', 'Volatility_30']
aapl = aapl.dropna(subset=features + ['Close_next'])
X = aapl[features]
y = aapl['Close_next']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('\n--- Predictive Modeling Results (AAPL) ---')
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R^2:', r2_score(y_test, y_pred))
plt.figure(figsize=(10, 5))
plt.plot(aapl['Date'].iloc[-len(y_test):], y_test, label='Actual')
plt.plot(aapl['Date'].iloc[-len(y_test):], y_pred, label='Predicted')
plt.title('AAPL: Actual vs Predicted Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print('\nAnalysis Complete! You can expand this for other tickers or more advanced models as needed.')
