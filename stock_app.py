import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- Load Model & Scaler ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/lstm_model.h5")
    scaler = joblib.load("scaler.gz")
    return model, scaler

model, scaler = load_model()

# --- App UI ---
st.title("📈 Stock Price Prediction using LSTM")
ticker = st.text_input("Enter Stock Ticker Symbol", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
# end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))
today = pd.to_datetime("today").date()
end_date = st.date_input("End Date", value=today, max_value=today)

if st.button("Predict"):

    try:
        # --- Load Data ---
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found. Please check the ticker or date range.")
        else:
            close_data = df[['Close']].dropna()
            st.success(f"✅ Loaded {len(close_data)} rows of stock data")
        
            # --- Show current stock price ---
            # current_price = close_data['Close'].iloc[-1]
            # current_price = float(close_data.iloc[-1]['Close'])
            # st.metric(label=f"📌 Current Closing Price of {ticker}", value=f"${current_price:.2f}")

            # Get current stock price
            current_price = float(close_data.iloc[-1]['Close'])

            # Try to get currency
            try:
                stock_info = yf.Ticker(ticker).info
                currency = stock_info.get("currency", "USD")
            except:
                currency = "USD"

            # Format symbol
            currency_symbols = {
                "USD": "$",
                "INR": "₹",
                "EUR": "€",
                "GBP": "£",
                "JPY": "¥"
            }
            symbol = currency_symbols.get(currency, currency)

            # Display
            st.metric(label=f"📌 Current Closing Price of {ticker}",
                    value=f"{symbol}{current_price:,.2f} ({currency})")

            # --- Preprocessing ---
            scaled_data = scaler.transform(close_data)
            sequence_length = 60

            def create_sequences(data, seq_len=60):
                x, y = [], []
                for i in range(seq_len, len(data)):
                    x.append(data[i-seq_len:i])
                    y.append(data[i])
                return np.array(x), np.array(y)

            X, y_true = create_sequences(scaled_data, sequence_length)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # --- Predict on historical data (for evaluation) ---
            y_pred = model.predict(X, verbose=0)
            y_true_inv = scaler.inverse_transform(y_true)
            y_pred_inv = scaler.inverse_transform(y_pred)

            # --- Show Actual vs Predicted Plot ---
            st.subheader("🔁 Actual vs Predicted (Training Data)")
            fig, ax = plt.subplots()
            ax.plot(y_true_inv, label="Actual", color="blue")
            ax.plot(y_pred_inv, label="Predicted", color="orange")
            ax.set_title("Actual vs Predicted Closing Prices")
            ax.legend()
            st.pyplot(fig)

            # --- Predict future (7, 15, 30 days) ---
            st.subheader("🔮 Future Predictions")
            def predict_future(last_sequence, days):
                temp_seq = last_sequence.copy()
                future_preds = []
                for _ in range(days):
                    pred = model.predict(temp_seq.reshape(1, sequence_length, 1), verbose=0)[0][0]
                    future_preds.append(pred)
                    temp_seq = np.append(temp_seq[1:], [[pred]], axis=0)
                return scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

            last_seq = scaled_data[-sequence_length:]
            for days in [7, 15, 30]:
                future_pred = predict_future(last_seq, days)
                st.subheader(f"📆 Next {days} Days")
                st.line_chart(pd.DataFrame(future_pred, columns=["Predicted Close"]))

    except Exception as e:
        st.error(f"❌ Error: {e}")
