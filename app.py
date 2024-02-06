import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

# Fetch stock data from Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Train ARIMA model and make predictions
def train_arima_model(data):
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit

# Evaluate model accuracy
def evaluate_model(model_fit, test_data):
    predictions = model_fit.predict(start=len(test_data), end=len(test_data)+len(test_data)-1)
    mse = ((predictions - test_data) ** 2).mean()
    rmse = mse ** 0.5
    return rmse

# Streamlit Dashboard
def main():
    st.title("Stock Price Prediction Dashboard")
    
    # User input for stock symbol and date range
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):")
    start_date = st.date_input("Enter Start Date:")
    end_date = st.date_input("Enter End Date:")
    
    if symbol and start_date and end_date:
        # Fetch stock data
        stock_data = fetch_stock_data(symbol, start_date, end_date)
        
        if not stock_data.empty:
            st.subheader("Stock Data")
            st.write(stock_data)
            
            # Train ARIMA model
            model_fit = train_arima_model(stock_data['Close'])
            
            # Make predictions
            predictions = model_fit.predict(start=len(stock_data), end=len(stock_data)+30)
            
            # Evaluate model accuracy
            rmse = evaluate_model(model_fit, stock_data['Close'])
            st.subheader("Model Accuracy")
            st.write("Root Mean Squared Error (RMSE):", rmse)
            
            # Visualize stock data and predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Actual'))
            fig.add_trace(go.Scatter(x=predictions.index, y=predictions, name='Predicted'))
            fig.update_layout(title="Stock Price Prediction", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig)
            
            
            # Add code for additional visualizations or predictions here
            forecast_days = st.number_input("Enter the number of days to forecast:", min_value=1, max_value=365, value=30)

            # Make predictions for the specified number of days
            future_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=forecast_days)
            future_predictions = model_fit.predict(start=len(stock_data), end=len(stock_data) + forecast_days - 1)

            # Visualize future predictions
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Actual'))
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_predictions, name='Future Predictions'))
            fig_future.update_layout(title="Stock Price Prediction (Including Future)", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_future)

if __name__ == "__main__":
    main()
