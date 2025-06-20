import pandas as pd
from data_utils import load_and_prepare_data, print_data_info, add_rolling_features, split_features_targets
from visualization import (
    plot_closing_prices, plot_volume, plot_moving_averages_volatility,
    plot_correlation_heatmap, plot_actual_vs_predicted
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

if __name__ == '__main__':
    # Data Preparation
    df = load_and_prepare_data('stocks.csv')
    print_data_info(df)
    
    # Feature Engineering
    df = add_rolling_features(df)

    # EDA Visualizations
    plot_closing_prices(df)
    plot_volume(df)
    plot_moving_averages_volatility(df)
    plot_correlation_heatmap(df)

    # Predictive Modeling Example (AAPL)
    X, y, dates = split_features_targets(df, ticker='AAPL')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('\n--- Predictive Modeling Results (AAPL) ---')
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R^2:', r2_score(y_test, y_pred))
    plot_actual_vs_predicted(dates.iloc[-len(y_test):], y_test, y_pred, ticker='AAPL')
    print('\nAnalysis Complete! You can expand this for other tickers or more advanced models as needed.')
