import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf 
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 
import tensorflow as tf 

data = yf.download('AAPL', start='2010-01-01', end='2023-01-01') 
data = data[['Close']] 

scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1)) 

window_size = 60 

train_size = int(len(scaled_data) * 0.8) 
train_data = scaled_data[:train_size] 
test_data = scaled_data[train_size:]
 
def create_dataset(data, window_size): 
    X, y = [], [] 
    for i in range(window_size, len(data)): 
        X.append(data[i-window_size:i, 0]) 
        y.append(data[i, 0]) 
    return np.array(X), np.array(y) 
 
X_train, y_train = create_dataset(train_data, window_size) 
X_test, y_test = create_dataset(test_data, window_size) 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 

def create_model(): 
    model = Sequential() 
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1))) 
    model.add(LSTM(units=50, return_sequences=False)) 
    model.add(Dense(units=25)) 
    model.add(Dense(units=1)) 
    model.compile(optimizer='adam', loss='mean_squared_error') 
return model 

model = create_model() 
history_normal = model.fit(X_train, y_train, batch_size=1, epochs=5, validation_split=0.2) 
model.save('lstm_model_normal.keras') 

from tensorflow.keras.models import load_model 
model = load_model('lstm_model_normal.keras') 

predictions = model.predict(X_test) 
predictions = scaler.inverse_transform(predictions) 

plt.figure(figsize=(16,8)) 
plt.plot(data.index[train_size+window_size:], data['Close'][train_size+window_size:], color='blue', 
label='Фактические значения') 
plt.plot(data.index[train_size+window_size:], 
значения') 
plt.xlabel('Дата') 
plt.ylabel('Цена закрытия') 
plt.legend() 
plt.show() 

# Распределенное обучение 
predictions, 
color='red', 
label='Прогнозные 
strategy = tf.distribute.MirroredStrategy() 
with strategy.scope(): 
model_distributed = create_model() 
history_distributed = model_distributed.fit(X_train, y_train, batch_size=1, epochs=5, 
validation_split=0.2) 
model_distributed.save('lstm_model_distributed.keras') 

def predict_and_plot(model, model_name): 
    predictions = model.predict(X_test) 
    predictions = scaler.inverse_transform(predictions) 
    plt.figure(figsize=(16,8)) 
    plt.plot(data.index[train_size+window_size:], data['Close'][train_size+window_size:], 
color='blue', label='Фактические значения') 
    plt.plot(data.index[train_size+window_size:], predictions, color='red', label='Прогнозные 
значения') 
    plt.xlabel('Дата') 
    plt.ylabel('Цена закрытия') 
    plt.title(f'Прогнозы модели {model_name}') 
    plt.legend() 
    plt.show() 
 
predict_and_plot(model_distributed, 'Распределенная') 
 
# Сравнение историй обучения
plt.figure(figsize=(12, 6)) 
plt.plot(history_normal.history['loss'], label='Обычная модель (loss)') 
plt.plot(history_normal.history['val_loss'], label='Обычная модель (val_loss)') 
plt.plot(history_distributed.history['loss'], label='Распределенная модель (loss)') 
plt.plot(history_distributed.history['val_loss'], label='Распределенная модель (val_loss)') 
plt.xlabel('Эпохи') 
plt.ylabel('Loss') 
plt.title('Сравнение обучения') 
plt.legend() 
plt.show()
