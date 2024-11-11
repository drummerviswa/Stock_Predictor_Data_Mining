import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.dates as mdates
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data.csv')  # Load the ITC stock data

# Check for missing data
print(pd.isnull(df).sum())

# Check for non-null data
print(pd.notnull(df).sum())

# Check for duplicate records
print(df.duplicated().sum())

# Manage data types
df.info()

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
# Extract 'Day', 'Month', and 'Year' from 'Date' column
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Check the updated dataframe
print(df.head())

# Remove commas and convert columns to numeric types (e.g., VOLUME, VALUE)
df['VOLUME'] = df['VOLUME'].replace({',': ''}, regex=True).astype(float)
df['VALUE'] = df['VALUE'].replace({',': ''}, regex=True).astype(float)

# Ensure that all necessary columns are in the correct numeric type
df['LOW'] = pd.to_numeric(df['LOW'], errors='coerce')
df['OPEN'] = pd.to_numeric(df['OPEN'], errors='coerce')
df['HIGH'] = pd.to_numeric(df['HIGH'], errors='coerce')
df['close'] = pd.to_numeric(df['close'], errors='coerce')

# Select relevant columns for modeling (remove 'Date' column, as it's not needed for regression)
df = df[['Date', 'LOW', 'OPEN', 'VOLUME', 'HIGH', 'close', 'Day', 'Month', 'Year']]

df.dropna(inplace=True)  # Drop any rows with missing values

# Visualizing the close price
plt.figure(figsize=(16, 8))
plt.title("ITC Stock Close Price")
plt.xlabel('Days')
plt.ylabel('Close Price in ₹INR')
plt.plot(df['Date'], df['close'])  # Use 'Date' column for x-axis
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
plt.show()

# Create a new column 'Prediction' for future prediction
rows = len(df.axes[0])  # Get the number of rows
future_days = rows // 2  # Assuming 20% of data is used for prediction
df['Prediction'] = df[['close']].shift(-future_days)  # Shift the 'close' column for prediction

# Prepare data for training and testing
F = np.array(df.drop(['Prediction', 'Date'], axis=1))[:-future_days]  # Dropping 'Date' column
T = np.array(df['Prediction'])[:-future_days]

# Split data into training (80%) and testing (20%)
f_train, f_test, t_train, t_test = train_test_split(F, T, test_size=0.2, shuffle=False)  # Ensure time series order
print(f_train.shape, f_test.shape, t_train.shape, t_test.shape)

# Apply MinMaxScaler to scale features
scaler = MinMaxScaler(feature_range=(0, 1))
f_train_scaled = scaler.fit_transform(f_train)
f_test_scaled = scaler.transform(f_test)

# DecisionTreeRegressor Model
tree = DecisionTreeRegressor(max_depth=150, min_samples_leaf=20)
tree.fit(f_train_scaled, t_train)

# LinearRegression Model
lr = LinearRegression().fit(f_train_scaled, t_train)

# RandomForestRegressor Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(f_train_scaled, t_train)

# Prepare the future dataset for prediction
n_future = df.drop(['Prediction', 'Date'], axis=1)[:-future_days]  # Dropping 'Date' column
n_future = n_future.tail(future_days)
n_future_scaled = scaler.transform(np.array(n_future))  # Scaling future data

# Predictions using DecisionTreeRegressor
tree_prediction = tree.predict(n_future_scaled)

# Predictions using LinearRegression
lr_prediction = lr.predict(n_future_scaled)

# Predictions using RandomForestRegressor
rf_prediction = rf.predict(n_future_scaled)

# Create a validation set (last 'future_days' rows for validation)
valid = df[-future_days:].copy()
valid['Predictions_DecisionTree'] = tree_prediction
valid['Predictions_LR'] = lr_prediction
valid['Predictions_RF'] = rf_prediction

# Visualizing the Predictions using DecisionTreeRegressor
plt.figure(figsize=(16, 8))
plt.title('Decision Tree Regressor Model')
plt.xlabel('Date')
plt.ylabel('Close Price ₹INR')
plt.plot(df['Date'], df['close'], label='Original')
plt.plot(valid['Date'], valid['Predictions_DecisionTree'], label='Predicted (Decision Tree)', linestyle='--')
plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
plt.show()

# Visualizing the Predictions using LinearRegression
plt.figure(figsize=(16, 8))
plt.title('Linear Regression Model')
plt.xlabel('Date')
plt.ylabel('Close Price ₹INR')
plt.plot(df['Date'], df['close'], label='Original')
plt.plot(valid['Date'], valid['Predictions_LR'], label='Predicted (Linear Regression)', linestyle='--')
plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
plt.show()

# Visualizing the Predictions using RandomForestRegressor
plt.figure(figsize=(16, 8))
plt.title('Random Forest Regressor Model')
plt.xlabel('Date')
plt.ylabel('Close Price ₹INR')
plt.plot(df['Date'], df['close'], label='Original')
plt.plot(valid['Date'], valid['Predictions_RF'], label='Predicted (Random Forest)', linestyle='--')
plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
plt.show()

# Calculate the performance metrics
mse_tree = mean_squared_error(t_test, tree.predict(f_test_scaled))
mae_tree = mean_absolute_error(t_test, tree.predict(f_test_scaled))
print(f"Decision Tree MSE: {mse_tree}, MAE: {mae_tree}")

mse_lr = mean_squared_error(t_test, lr.predict(f_test_scaled))
mae_lr = mean_absolute_error(t_test, lr.predict(f_test_scaled))
print(f"Linear Regression MSE: {mse_lr}, MAE: {mae_lr}")

mse_rf = mean_squared_error(t_test, rf.predict(f_test_scaled))
mae_rf = mean_absolute_error(t_test, rf.predict(f_test_scaled))
print(f"Random Forest MSE: {mse_rf}, MAE: {mae_rf}")

# Stock Market Buy and Sell Classification
df['Return'] = df['close'].pct_change(90).shift(-90)  # 90-day return
df['Signal'] = np.where(df['Return'] > 0, 1, 0)  # Buy (1) or Sell (0) signal

# Select features for classification model
features = ['Day', 'Month', 'Year', 'LOW', 'OPEN', 'VOLUME', 'HIGH', 'close']
X = df[features]
y = df['Signal']

# Split data for training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=550, shuffle=False)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Decision Tree Classifier for Buy/Sell Signal
treeClassifier = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5)
treeClassifier.fit(X_train, y_train)

# Make predictions
y_pred = treeClassifier.predict(X_test)

# Performance Evaluation using classification report
report = classification_report(y_test, y_pred)
print(report)

# Debugging: Checking the last few rows of prediction data
print(f"Last few rows of predictions:\n {valid.tail()}")

# Checking the range of prediction dates
print(f"Prediction date range:\n {valid['Date'].min()} to {valid['Date'].max()}")

# Split data into features and target
df_lstm = df[['close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_lstm)

# Create dataset for LSTM
def create_lstm_data(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_lstm_data(df_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data for training and testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Prepare features for other models
features = ['Day', 'Month', 'Year', 'LOW', 'OPEN', 'VOLUME', 'HIGH', 'close']
F = np.array(df[features])[:-60]  # Adjust based on time_step for LSTM
T = np.array(df['close'])[:-60]

f_train, f_test, t_train, t_test = train_test_split(F, T, test_size=0.2, shuffle=False)

# Apply MinMaxScaler to scale features
scaler_features = MinMaxScaler(feature_range=(0, 1))
f_train_scaled = scaler_features.fit_transform(f_train)
f_test_scaled = scaler_features.transform(f_test)

# DecisionTreeRegressor Model
tree = DecisionTreeRegressor(max_depth=150, min_samples_leaf=20)
tree.fit(f_train_scaled, t_train)
tree_prediction = tree.predict(f_test_scaled)

# LinearRegression Model
lr = LinearRegression().fit(f_train_scaled, t_train)
lr_prediction = lr.predict(f_test_scaled)

# RandomForestRegressor Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(f_train_scaled, t_train)
rf_prediction = rf.predict(f_test_scaled)

# LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)
lstm_prediction = model.predict(X_test)

# Inverse transform LSTM predictions
lstm_prediction_rescaled = scaler.inverse_transform(lstm_prediction)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate performance metrics
mse_tree = mean_squared_error(t_test, tree_prediction)
mae_tree = mean_absolute_error(t_test, tree_prediction)

mse_lr = mean_squared_error(t_test, lr_prediction)
mae_lr = mean_absolute_error(t_test, lr_prediction)

mse_rf = mean_squared_error(t_test, rf_prediction)
mae_rf = mean_absolute_error(t_test, rf_prediction)

mse_lstm = mean_squared_error(y_test_rescaled, lstm_prediction_rescaled)
mae_lstm = mean_absolute_error(y_test_rescaled, lstm_prediction_rescaled)

# Print performance metrics for comparison
print("Decision Tree Regressor MSE:", mse_tree, "MAE:", mae_tree)
print("Linear Regression MSE:", mse_lr, "MAE:", mae_lr)
print("Random Forest Regressor MSE:", mse_rf, "MAE:", mae_rf)
print("LSTM Model MSE:", mse_lstm, "MAE:", mae_lstm)

# Visualizing the predictions
plt.figure(figsize=(16, 8))
plt.title('Model Comparison')
plt.xlabel('Date')
plt.ylabel('Close Price ₹INR')

# Plot actual prices
plt.plot(df['Date'][-len(y_test):], y_test_rescaled, label='Actual Price', color='black')

# Plot predictions of each model
plt.plot(df['Date'][-len(y_test):], lstm_prediction_rescaled, label='LSTM Predicted Price', linestyle='--')
plt.plot(df['Date'][-len(t_test):], tree_prediction, label='Decision Tree Predicted Price', linestyle='-.')
plt.plot(df['Date'][-len(t_test):], lr_prediction, label='Linear Regression Predicted Price', linestyle=':')
plt.plot(df['Date'][-len(t_test):], rf_prediction, label='Random Forest Predicted Price', linestyle='-.')

plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
plt.show()

# Best Model Selection (Based on MSE or MAE)
best_model = None
best_mse = float('inf')
best_mae = float('inf')

models = {
    'Decision Tree': (mse_tree, mae_tree, tree_prediction),
    'Linear Regression': (mse_lr, mae_lr, lr_prediction),
    'Random Forest': (mse_rf, mae_rf, rf_prediction),
    'LSTM': (mse_lstm, mae_lstm, lstm_prediction_rescaled)
}

# Correct comparison and selection logic
for model_name, (mse, mae, _) in models.items():
    if mse < best_mse:
        best_mse = mse
        best_model = model_name

print(f"\nBest Model based on MSE: {best_model}")

# Get the best model's predictions
best_predictions = models[best_model][2]

# Compare the best model with actual prices (plotting)
plt.figure(figsize=(16, 8))
plt.title(f"Best Model: {best_model} Comparison with Actual Price")
plt.xlabel('Date')
plt.ylabel('Close Price ₹INR')

# Plot actual prices
plt.plot(df['Date'][-len(y_test):], y_test_rescaled, label='Actual Price', color='black')

# Plot the best model's predictions
plt.plot(df['Date'][-len(y_test):], best_predictions, label=f'{best_model} Predicted Price', linestyle='--')

plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
plt.show()