import pandas as pd
import numpy as np

def load_and_prepare_data(csv_path='stocks.csv'):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date'])
    return df

def print_data_info(df):
    print('--- Data Info ---')
    print(df.info())
    print('\n--- Null Values ---')
    print(df.isnull().sum())

def add_rolling_features(df, windows=[7, 30]):
    def add_features(group):
        for window in windows:
            group[f'MA_{window}'] = group['Close'].rolling(window).mean()
            group[f'Volatility_{window}'] = group['Close'].rolling(window).std()
        return group
    return df.groupby('Ticker', group_keys=False).apply(add_features)

def split_features_targets(df, ticker='AAPL'):
    stock = df[df['Ticker'] == ticker].copy()
    stock['Close_next'] = stock['Close'].shift(-1)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_7', 'MA_30', 'Volatility_7', 'Volatility_30']
    stock = stock.dropna(subset=features + ['Close_next'])
    X = stock[features]
    y = stock['Close_next']
    dates = stock['Date']
    return X, y, dates
