import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, set_random_seed
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error

# Set random seed
set_random_seed(42)

# Sidebar Inputs
st.sidebar.title("🛠️ Stock Forecasting Controls")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., INFY.NS, TCS.NS)", value="INFY.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-10-01"))
predict_button = st.sidebar.button("📊 Predict Stock")

# App Title
st.title(f"📈 NeuralProphet Forecasting App for '{ticker.upper()}'")

# Function to load data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']].reset_index()
    df.columns = ['ds', 'y']
    return df

# Main prediction flow
if predict_button:
    data = load_data(ticker, start_date, end_date)

    if data.empty:
        st.warning("⚠️ No data found for this ticker in the selected date range.")
        st.stop()

    # Display sample data
    st.subheader("📊 Sample Stock Data")
    st.dataframe(data.tail())

    # Train-Test Split
    train_size = int(len(data) * 0.8)
    train_df = data.iloc[:train_size]
    test_df = data.iloc[train_size:]

    # Configure NeuralProphet Model
    model = NeuralProphet(
        n_lags=60,
        n_forecasts=30,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

    # Train Model
    with st.spinner("🔄 Training NeuralProphet model..."):
        model.fit(train_df, freq="D")

    # Predict on Test Set
    st.subheader("🔁 Rolling Forecast (Multiple Origins)")
    forecast_df = model.predict(test_df)

    # Plot actual vs forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y'], mode='markers', name='Actual', marker=dict(color='black', size=4)))

    for col in forecast_df.columns:
        if col.startswith('yhat'):
            fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df[col], mode='lines', name=col, opacity=0.3, line=dict(width=1)))

    fig.update_layout(
        title=f"Rolling Forecast - {ticker.upper()}",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600
    )
    st.plotly_chart(fig)

    # Evaluation Metrics
    st.subheader("📈 Model Evaluation")
    merged = test_df.merge(forecast_df[['ds', 'yhat1']], on='ds', how='inner').dropna()

    r2 = r2_score(merged['y'], merged['yhat1'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat1']))
    mape = np.mean(np.abs((merged['y'] - merged['yhat1']) / merged['y'])) * 100

    st.markdown(f"**R² Score**: `{r2:.4f}`")
    st.markdown(f"**RMSE**: `{rmse:.2f}`")
    st.markdown(f"**MAPE**: `{mape:.2f}%`")

    # Future Forecast
    st.subheader("🔮 Forecast Next 30 Days")
    future_df = model.make_future_dataframe(data, periods=30)
    forecast_future = model.predict(future_df)

    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines+markers', name='Historical'))
    fig_future.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat1'], mode='lines', name='Forecast'))

    fig_future.update_layout(
        title=f"{ticker.upper()} - 30 Day Future Forecast",
        xaxis_title='Date',
        yaxis_title='Price'
    )
    st.plotly_chart(fig_future)

else:
    st.info("👈 Enter details on the sidebar and click **Predict Stock** to begin.")
