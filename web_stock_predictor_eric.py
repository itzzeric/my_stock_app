import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("Stock Forecasting Web App")

stock = st.text_input('Enter Stock Symbol', 'GOOG')

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

google_data = yf.download(stock, start, end)

if google_data.empty:
    st.error(f"Failed to fetch data for {stock}. Please check the symbol.")
    st.stop()  

if google_data.isnull().sum().any():
    st.write("Missing values found in the data, dropping rows with missing values...")
    google_data = google_data.dropna()

st.subheader("Select Date Range for Volume Forecasting")
start_date = st.date_input('Start Date', min_value=google_data.index.min(), max_value=google_data.index.max(), value=google_data.index.min())
end_date = st.date_input('End Date', min_value=google_data.index.min(), max_value=google_data.index.max(), value=google_data.index.max())

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered_data = google_data[(google_data.index >= start_date) & (google_data.index <= end_date)]

st.write(f"Displaying data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
st.write(filtered_data)

st.subheader("Stock Volume for the Selected Date Range")
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(filtered_data.index, filtered_data['Volume'], label='Volume', color='orange')
ax.set_xlabel('Date')
ax.set_ylabel('Volume')
ax.set_title(f"Volume of {stock} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
ax.legend()
st.pyplot(fig)

model = load_model("Latest_stock_price_model_byERIC.keras")

splitting_len = int(len(google_data) * 0.7)

x_test = google_data[['Close']].iloc[splitting_len:]

if 'Close' not in x_test.columns:
    st.error("The 'Close' column is missing in x_test!")
    st.stop()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len + 100:]
)

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

def plot_graph(figsize, values, full_data, extra_data=None, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange')
    plt.plot(full_data.Close, 'b')
    if extra_data is not None:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, extra_data=1, extra_dataset=google_data['MA_for_250_days']))

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data['Close'][:splitting_len + 100], ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test Data", "Predicted Test Data"])
st.pyplot(fig)
