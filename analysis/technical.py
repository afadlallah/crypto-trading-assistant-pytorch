class TechnicalAnalysis:
    def __init__(self, data):
        self.data = data

    def add_sma(self, windows=[20, 50, 100, 200]):
        """Add Simple Moving Averages to the dataset."""
        for window in windows:
            self.data[f'SMA_{window}'] = self.data['close'].rolling(window=window).mean()

    def add_rsi(self, window=14):
        """Add Relative Strength Index to the dataset."""
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

    def add_bollinger_bands(self, window=20, num_std=2):
        """Add Bollinger Bands to the dataset."""
        self.data['BB_middle'] = self.data['close'].rolling(window=window).mean()
        std = self.data['close'].rolling(window=window).std()
        self.data['BB_upper'] = self.data['BB_middle'] + (std * num_std)
        self.data['BB_lower'] = self.data['BB_middle'] - (std * num_std)

    def add_macd(self, short_window=12, long_window=26, signal_window=9):
        """Add Moving Average Convergence Divergence (MACD) to the dataset."""
        short_ema = self.data['close'].ewm(span=short_window, adjust=False).mean()
        long_ema = self.data['close'].ewm(span=long_window, adjust=False).mean()
        self.data['MACD'] = short_ema - long_ema
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=signal_window, adjust=False).mean()
        self.data['MACD_histogram'] = self.data['MACD'] - self.data['MACD_signal']

    def add_ema(self, window=20):
        """Add Exponential Moving Average to the dataset."""
        self.data[f'EMA_{window}'] = self.data['close'].ewm(span=window, adjust=False).mean()

    def add_stochastic_oscillator(self, window=14, smooth_window=3):
        """Add Stochastic Oscillator to the dataset."""
        low_min = self.data['low'].rolling(window=window).min()
        high_max = self.data['high'].rolling(window=window).max()
        self.data['%K'] = 100 * (self.data['close'] - low_min) / (high_max - low_min)
        self.data['%D'] = self.data['%K'].rolling(window=smooth_window).mean()

    def calculate_all_indicators(self):
        """Calculate all technical indicators."""
        self.add_sma()  # Calculate SMAs for 20, 50, 100, and 200 periods
        self.add_ema(20)  # Calculate EMA with window of 20
        self.add_rsi()    # Calculate RSI
        self.add_bollinger_bands()  # Calculate Bollinger Bands
        self.add_macd()   # Calculate MACD
        self.add_stochastic_oscillator()  # Calculate Stochastic Oscillator
        return self.data