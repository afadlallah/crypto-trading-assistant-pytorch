from datetime import timedelta
import numpy as np
import pandas as pd


class TradingSignalGenerator:
    def __init__(self, price_data, technical_indicators, sentiment_data, prediction_data, timeframe):
        self.price_data = price_data
        self.technical_indicators = technical_indicators
        self.sentiment_data = sentiment_data
        self.prediction_data = prediction_data
        self.timeframe = timeframe
        self.timeframe_params = self._get_timeframe_params()

    def _get_timeframe_params(self):
        # Convert timeframe string to timedelta
        tf_dict = {'m': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}
        num = int(''.join(filter(str.isdigit, self.timeframe)))
        unit = ''.join(filter(str.isalpha, self.timeframe))
        tf_delta = timedelta(**{tf_dict[unit]: num})

        # Adjust parameters based on timeframe
        if tf_delta <= timedelta(minutes=15):
            return {'sma_fast': 20, 'sma_slow': 50, 'rsi_window': 14, 'sentiment_window': 12}
        elif timedelta(minutes=15) < tf_delta <= timedelta(hours=4):
            return {'sma_fast': 50, 'sma_slow': 100, 'rsi_window': 14, 'sentiment_window': 24}
        else:  # 1d and above
            return {'sma_fast': 50, 'sma_slow': 200, 'rsi_window': 14, 'sentiment_window': 7}

    def generate_signals(self):
        signals = pd.DataFrame(index=self.price_data.index)

        # Signal based on SMA crossover
        signals['SMA_Signal'] = np.where(
            self.technical_indicators[f'SMA_{self.timeframe_params["sma_fast"]}'] >
            self.technical_indicators[f'SMA_{self.timeframe_params["sma_slow"]}'], 1, 0)
        signals['SMA_Signal'] = np.where(
            self.technical_indicators[f'SMA_{self.timeframe_params["sma_fast"]}'] <
            self.technical_indicators[f'SMA_{self.timeframe_params["sma_slow"]}'], -1, signals['SMA_Signal'])

        # Signal based on RSI
        signals['RSI_Signal'] = np.where(self.technical_indicators['RSI'] < 30, 1, 0)  # Oversold
        signals['RSI_Signal'] = np.where(self.technical_indicators['RSI'] > 70, -1, signals['RSI_Signal'])  # Overbought

        # Signal based on Bollinger Bands
        signals['BB_Signal'] = np.where(self.price_data['close'] < self.technical_indicators['BB_lower'], 1, 0)  # Price below lower band
        signals['BB_Signal'] = np.where(self.price_data['close'] > self.technical_indicators['BB_upper'], -1, signals['BB_Signal'])  # Price above upper band

        # Signal based on sentiment
        self.sentiment_data = self.sentiment_data.reindex(self.price_data.index, method='ffill')
        rolling_mean = self.sentiment_data.rolling(window=self.timeframe_params['sentiment_window']).mean().fillna(self.sentiment_data)
        signals['Sentiment_Signal'] = np.where(self.sentiment_data > rolling_mean, 1, -1)

        # Signal based on price prediction
        signals['Prediction_Signal'] = 0  # Initialize with neutral signal
        if self.prediction_data is not None and len(self.prediction_data) > 0:
            prediction_data = self.prediction_data.flatten() if self.prediction_data.ndim > 1 else self.prediction_data
            prediction_series = pd.Series(prediction_data, index=self.price_data.index[-len(prediction_data):])

            # Ensure we're only comparing data points that exist in both series
            common_index = self.price_data.index.intersection(prediction_series.index)

            if not common_index.empty:
                signals.loc[common_index, 'Prediction_Signal'] = np.where(
                    prediction_series[common_index] > self.price_data.loc[common_index, 'close'], 1, -1
                )

        # Combine signals
        signals['Combined_Signal'] = signals.mean(axis=1)

        # Generate final trading signal
        signals['Trading_Signal'] = np.where(signals['Combined_Signal'] > 0.5, 1, 0)  # Strong buy
        signals['Trading_Signal'] = np.where(signals['Combined_Signal'] < -0.5, -1, signals['Trading_Signal'])  # Strong sell

        return signals

    def calculate_signal_strength(self, signals):
        # Normalize the combined signal to a range of -100 to 100
        min_signal = signals['Combined_Signal'].min()
        max_signal = signals['Combined_Signal'].max()
        signals['Signal_Strength'] = 200 * (signals['Combined_Signal'] - min_signal) / (max_signal - min_signal) - 100
        return signals

    def get_latest_signal(self, signals):
        latest_signal = signals.iloc[-1]
        signal_strength = latest_signal['Signal_Strength']

        if latest_signal['Trading_Signal'] == 1:
            return f"Strong Buy (Strength: {signal_strength:.2f})"
        elif latest_signal['Trading_Signal'] == -1:
            return f"Strong Sell (Strength: {signal_strength:.2f})"
        elif signal_strength > 0:
            return f"Weak Buy (Strength: {signal_strength:.2f})"
        elif signal_strength < 0:
            return f"Weak Sell (Strength: {signal_strength:.2f})"
        else:
            return "Hold"
