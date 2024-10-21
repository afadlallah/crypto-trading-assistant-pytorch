#!/usr/bin/env python3

from analysis import SentimentAnalysis, TechnicalAnalysis
from data.collector import DataCollector
from dotenv import load_dotenv
from ml.predictor import PricePredictor
from trading.signal_generator import TradingSignalGenerator
import logging
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the root directory of the project (two levels up from the current file)
project_root = os.path.dirname(os.path.dirname(current_file_path))

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sentiment_chart(df):
    fig = go.Figure(data=[go.Scatter(x=df.index, y=df, mode='lines')])
    fig.update_layout(
        title='Daily Sentiment Score',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        height=800,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def create_candlestick_chart(df, predictions=None, height=600):
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])

    if predictions is not None:
        fig.add_trace(go.Scatter(x=df.index[-len(predictions):], y=predictions.flatten(), mode='lines', name='Prediction', line=dict(color='red', dash='dash')))

    fig.update_layout(
        title='Bitcoin Price',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=800,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def create_line_chart(df, title, y_axis_title):
    if df.empty:
        logger.warning(f'Empty DataFrame for {title}')
        return go.Figure()

    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))

    fig.update_layout(
        title=title,
        xaxis_title='Epoch',
        yaxis_title=y_axis_title,
        height=800,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def create_signal_chart(signals):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=signals.index, y=signals['Signal_Strength'], mode='lines', name='Signal Strength'))
    fig.add_trace(go.Scatter(x=signals.index, y=[50] * len(signals), mode='lines', name='Buy Threshold', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=signals.index, y=[-50] * len(signals), mode='lines', name='Sell Threshold', line=dict(color='red', dash='dash')))

    fig.update_layout(
        title='Trading Signal Strength Over Time',
        xaxis_title='Date',
        yaxis_title='Signal Strength',
        yaxis_range=[-100, 100],
        height=800,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def main():
    st.set_page_config(layout='wide')
    st.title('Bitcoin Trading Assistant')

    # Load environment variables
    load_dotenv()
    newsapi_key = os.getenv('NEWSAPI_KEY')

    exchange_proper = st.selectbox('Select Exchange', ['Bitstamp', 'Coinbase', 'Gemini', 'Kraken'])
    exchange = exchange_proper.lower()
    collector = DataCollector(exchange, newsapi_key=newsapi_key)

    # Initialize predictions to None
    predictions = None

    # Define timeframes for each exchange
    timeframes = {
        'bitstamp': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'],
        'coinbase': ['1m', '5m', '15m', '30m', '1h', '2h', '6h', '1d'],
        'gemini': ['1m', '5m', '15m', '30m', '1h', '6h', '1d'],
        'kraken': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '2w']
    }

    # Fetch OHLCV data
    timeframe = st.selectbox('Select Timeframe', timeframes[exchange], index=timeframes[exchange].index('1d'))
    ohlcv_data = collector.fetch_ohlcv(timeframe=timeframe)

    # Calculate technical indicators
    ta = TechnicalAnalysis(ohlcv_data)
    ohlcv_data_with_indicators = ta.calculate_all_indicators()

    # Add model settings
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        batch_size = st.slider('Batch Size', min_value=16, max_value=128, value=32, step=16)
        num_layers = st.slider('Number of Layers', min_value=1, max_value=5, value=2, step=1)
    with col2:
        hidden_dim = st.slider('Hidden Dimension', min_value=50, max_value=500, value=100, step=50)
        prediction_horizon = st.slider('Prediction Horizon (# Timeframes)', min_value=1, max_value=30, value=7, step=1)
    with col3:
        learning_rate = st.slider('Learning Rate', min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format='%.4f')
        epochs = st.slider('Training Epochs', min_value=10, max_value=1000, value=100, step=10)
    with col4:
        lookback_window = st.slider('Lookback Window (# Timeframes)', min_value=30, max_value=180, value=60, step=1)
        val_split = st.slider('Validation Split', min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    col5, col6 = st.columns(2)
    with col5:
        loss_fn = st.selectbox('Loss Function', ['mae', 'mse', 'huber'], index=1)
    with col6:
        optimizer = st.selectbox('Optimizer', ['adam', 'rmsprop', 'sgd'], index=0)

    # Train price prediction model
    if st.button('Train Price Prediction Model'):
        predictor = PricePredictor(
            lookback=lookback_window,
            prediction_horizon=prediction_horizon,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            val_split=val_split,
            optimizer=optimizer,
            loss_fn=loss_fn
        )
        with st.spinner('Training model...'):
            history = predictor.train(ohlcv_data_with_indicators)
        st.success('Model trained successfully!')

        # Make predictions
        predictions = predictor.predict(ohlcv_data_with_indicators)

        # Display candlestick chart with predictions
        st.plotly_chart(create_candlestick_chart(ohlcv_data, predictions), use_container_width=True)

        # Display model performance
        st.subheader('Model Performance')
        st.plotly_chart(create_line_chart(pd.DataFrame(history), 'Model Performance', 'Loss'), use_container_width=True)
    else:
        # Display candlestick chart without predictions
        st.plotly_chart(create_candlestick_chart(ohlcv_data), use_container_width=True)

    # Display technical indicators
    st.subheader('Technical Indicators')

    # Define new column names (excluding Timestamp)
    column_names = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA (20)', 'SMA (50)', 'SMA (100)', 'SMA (200)',
        'EMA (20)', 'RSI',
        'Bollinger Middle', 'Bollinger Upper', 'Bollinger Lower',
        'MACD', 'MACD Signal', 'MACD Histogram',
        'Stochastic %K', 'Stochastic %D'
    ]

    # Rename the columns
    formatted_data = ohlcv_data_with_indicators.copy()
    formatted_data.columns = column_names

    # Capitalize the index name (Timestamp)
    formatted_data.index.name = 'Timestamp'

    # Display the formatted dataframe
    st.dataframe(formatted_data.tail())

    # Display order book and recent trades side by side
    st.subheader('Market Data')
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        order_book = collector.fetch_order_book()
        st.write('Order Book - Bids')
        bids_df = order_book['bids'].copy()
        bids_df.index.name = 'Timestamp'
        bids_df.columns = bids_df.columns.str.capitalize()
        st.dataframe(bids_df)

    with col2:
        st.write('Order Book - Asks')
        asks_df = order_book['asks'].copy()
        asks_df.index.name = 'Timestamp'
        asks_df.columns = asks_df.columns.str.capitalize()
        st.dataframe(asks_df)

    with col3:
        st.write('Recent Trades')
        recent_trades = collector.fetch_recent_trades()
        st.dataframe(recent_trades.head())

    # Fetch and display on-chain data
    st.subheader('On-chain Metrics')
    on_chain_metrics = collector.fetch_multiple_on_chain_metrics()
    if not on_chain_metrics:
        st.warning('No on-chain metrics data available. Please check your internet connection or try again later.')
    else:
        for metric, data in on_chain_metrics.items():
            fig = create_line_chart(data, f'{metric.replace("-", " ").title()} Over Time', metric.replace("-", " ").title())
            if not fig.data:
                st.warning(f'No data available for {metric}')
            else:
                st.plotly_chart(fig, use_container_width=True)

    # Fetch and analyze news sentiment
    st.subheader('News Sentiment Analysis')
    news_data = collector.fetch_news(days_back=14)  # Fetch news for the last 14 days
    sentiment_analyzer = SentimentAnalysis()
    news_data = sentiment_analyzer.analyze_dataframe(news_data, 'title')
    daily_sentiment = sentiment_analyzer.get_daily_sentiment(news_data, 'publishedAt', 'sentiment')

    if daily_sentiment.empty:
        st.warning('No sentiment data available. Please check your NewsAPI key and internet connection.')
    else:
        st.plotly_chart(create_sentiment_chart(daily_sentiment), use_container_width=True)

        # Display raw sentiment data with renamed column
        st.subheader('Raw Sentiment Data')
        display_sentiment = daily_sentiment.copy()
        display_sentiment.index.name = 'Published'
        display_sentiment = display_sentiment.rename('Sentiment Score')
        st.dataframe(display_sentiment)

    # Generate trading signals only if predictions are available
    if predictions is not None and len(predictions) > 0:
        signal_generator = TradingSignalGenerator(ohlcv_data, ohlcv_data_with_indicators, daily_sentiment, predictions, timeframe)
        signals = signal_generator.generate_signals()
        signals = signal_generator.calculate_signal_strength(signals)

        # Display trading signals
        st.subheader('Trading Signals')
        latest_signal = signal_generator.get_latest_signal(signals)
        st.write(f'Latest Trading Signal: {latest_signal}')
        st.plotly_chart(create_signal_chart(signals), use_container_width=True)
    else:
        st.warning('No predictions available. Please train the model first.')


if __name__ == '__main__':
    main()
