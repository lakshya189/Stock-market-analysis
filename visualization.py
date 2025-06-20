import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_closing_prices(df):
    plt.figure(figsize=(12, 6))
    for ticker in df['Ticker'].unique():
        plt.plot(df[df['Ticker'] == ticker]['Date'], df[df['Ticker'] == ticker]['Close'], label=ticker)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Closing Prices Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/closing_prices.png')
    plt.show()
    plt.close()

def plot_volume(df):
    plt.figure(figsize=(12, 6))
    for ticker in df['Ticker'].unique():
        plt.plot(df[df['Ticker'] == ticker]['Date'], df[df['Ticker'] == ticker]['Volume'], label=ticker)
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('Volume Traded Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/volume.png')
    plt.show()
    plt.close()

def plot_moving_averages_volatility(df, windows=[7, 30]):
    os.makedirs('images', exist_ok=True)
    for ticker in df['Ticker'].unique():
        company = df[df['Ticker'] == ticker]
        plt.figure(figsize=(14, 6))
        plt.plot(company['Date'], company['Close'], label='Close', alpha=0.5)
        for window in windows:
            plt.plot(company['Date'], company[f'MA_{window}'], label=f'MA_{window}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title(f'{ticker} - Moving Averages')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('images/moving_averages_volatility.png')
        plt.show()
        plt.close()
        plt.figure(figsize=(14, 6))
        for window in windows:
            plt.plot(company['Date'], company[f'Volatility_{window}'], label=f'Volatility_{window}')
        plt.xlabel('Date')
        plt.ylabel('Volatility (Std Dev)')
        plt.title(f'{ticker} - Rolling Volatility')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Optionally, save volatility plot separately if needed
        plt.show()
        plt.close()

def plot_correlation_heatmap(df):
    pivot_close = df.pivot(index='Date', columns='Ticker', values='Close')
    correlation_matrix = pivot_close.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Closing Prices')
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/correlation_heatmap.png')
    plt.show()
    plt.close()

def plot_actual_vs_predicted(dates, y_true, y_pred, ticker='AAPL'):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, label='Actual')
    plt.plot(dates, y_pred, label='Predicted')
    plt.title(f'{ticker}: Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/actual_vs_predicted.png')
    plt.show()
    plt.close()
