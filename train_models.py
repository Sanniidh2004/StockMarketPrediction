import matplotlib
matplotlib.use("Agg")  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def fetch_stock_data(ticker):
    import yfinance as yf 

    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")

    if df.empty:
        return df

    return df[['Close']]


def linear_regression_prediction(df, years=3):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    future_days = 365 * years
    future_X = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
    future_prices = model.predict(future_X)

    return future_prices


def lstm_prediction(df, years=3):
    data = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    seq_len = 60
    X, y = [], []

    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    future_days = 365 * years
    last_sequence = scaled_data[-seq_len:]
    future_predictions = []

    for _ in range(future_days):
        input_seq = last_sequence.reshape((1, seq_len, 1))
        next_val = model.predict(input_seq, verbose=0)[0][0]
        future_predictions.append(next_val)
        last_sequence = np.append(last_sequence[1:], [[next_val]], axis=0)

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    return future_predictions.flatten()


def plot_stock_with_prediction(df, future_prices, company, model_name):
    plt.figure(figsize=(10, 5))

    plt.plot(df.index, df['Close'], label="Historical Data")

    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date,
        periods=len(future_prices) + 1,
        freq="D"
    )[1:]

    plt.plot(
        future_dates,
        future_prices,
        "--",
        label=f"{model_name} Prediction"
    )

    plt.title(f"{company.capitalize()} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    plt.savefig("static/stock_plot.png")
    plt.close()
