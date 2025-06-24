# 📈 Stock Price Prediction using LSTM

This project predicts future stock prices using a Long Short-Term Memory (LSTM) neural network. It includes:

- Real-time data fetching from Yahoo Finance (`yfinance`)
- Deep learning model with Keras/TensorFlow
- Future stock price predictions for 7, 15, and 30 days
- Actual vs Predicted trend plot
- Interactive web app built using **Streamlit**

---

## 🚀 Demo

![Screenshot 2025-06-25 000051](https://github.com/user-attachments/assets/8f8f6c76-9703-4540-8970-d5913daec575)
![Screenshot 2025-06-25 000106](https://github.com/user-attachments/assets/3dbde496-5237-4c65-a487-0f02e8dbaf7a)
![Screenshot 2025-06-25 000154](https://github.com/user-attachments/assets/38f4d189-32c4-4457-8a39-2a9b9921713b)
![Screenshot 2025-06-25 000126](https://github.com/user-attachments/assets/8488a2b2-26a1-4a2f-9cf4-f206431bb0a2)

---

## 🧠 Tech Stack

- Python 🐍
- yfinance 📉
- TensorFlow/Keras 🧠
- Pandas, Numpy, Matplotlib 📊
- Scikit-learn 🔬
- Streamlit 🌐

---

## 📦 Features

- ✅ Real-time data fetching from Yahoo Finance
- ✅ Train LSTM on historical data (e.g. AAPL, TATAMOTORS.NS, etc.)
- ✅ Actual vs Predicted graph
- ✅ Predict next 7, 15, 30 days stock prices
- ✅ Show current closing price and currency
- ✅ Fully interactive Streamlit web app

---

## 📂 Project Structure
stock-price-lstm/  
│  
├── notebook  
    ├── train_model.ipynb # Jupyter Notebook to train and save LSTM model  
├── models  
    ├── lstm_model.h5 # Trained LSTM model  
    ├── scaler.gz # MinMaxScaler used for normalization  
├── app.py # Streamlit app for prediction and visualization  
├── requirements.txt # All required Python packages  
└── README.md  

---

## 💡 Ticker Symbol Examples
| Company         | Exchange | Ticker        |
| --------------- | -------- | ------------- |
| Apple           | NASDAQ   | AAPL          |
| Tata Motors     | NSE      | TATAMOTORS.NS |
| Tesla           | NASDAQ   | TSLA          |
| Reliance        | NSE      | RELIANCE.NS   |
| Tata Motors ADR | NYSE     | TTM           |

---

## 🤝 Contributing
Pull requests are welcome! If you find a bug or want a feature, open an issue or PR.

---

## 📬 Contact
Created by **Vishwajeet Kumar** – feel free to reach out!  
Email : mailme.wivk@gmail.com
