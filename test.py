import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('ITC.csv')

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

df['LOW'] = pd.to_numeric(df['LOW'], errors='coerce')
df['OPEN'] = pd.to_numeric(df['OPEN'], errors='coerce')
df['HIGH'] = pd.to_numeric(df['HIGH'], errors='coerce')
df['close'] = pd.to_numeric(df['close'], errors='coerce')

df.dropna(inplace=True)

close_prices = df[['close']].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

def create_lstm_data(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_lstm_data(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32)

test_predictions = model.predict(X_test)
test_predictions_rescaled = scaler.inverse_transform(test_predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

mape = np.mean(np.abs((y_test_rescaled - test_predictions_rescaled) / y_test_rescaled)) * 100
print("LSTM Model MAPE:", mape, "%")

def predict_next_days(model, last_sequence, days_to_predict=5):
    predictions = []
    input_sequence = last_sequence

    for _ in range(days_to_predict):
        prediction = model.predict(input_sequence.reshape(1, -1, 1))
        predictions.append(prediction[0, 0])
        input_sequence = np.append(input_sequence[1:], prediction)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

last_sequence = scaled_data[-time_step:]
next_5_days = predict_next_days(model, last_sequence)

print("Predicted prices for the next 5 days:")
print(next_5_days)

plt.figure(figsize=(16, 8))
plt.title('LSTM Model: Actual Prices and Next 5 Days Prediction')
plt.xlabel('Days')
plt.ylabel('Close Price ₹INR')

plt.plot(df['Date'][-len(y_test):], y_test_rescaled, label='Actual Price', color='black')

plt.plot(df['Date'][-len(y_test):], test_predictions_rescaled, label='Test Predictions', linestyle='--')

future_dates = pd.date_range(df['Date'].iloc[-1], periods=6)[1:]
plt.plot(future_dates, next_5_days, label='Next 5 Days Predictions', marker='o')

plt.legend()
plt.xticks(rotation=45)
plt.show()
