# Crypto Trading Assistant (PyTorch Version)

Simple dashboard to assist in crypto trading. I built this to show how one can use common Python libraries like `ccxt` and `Streamlit` to implement an interactive ML pipeline and UI.

The app works by fetching real-time pricing data and news headlines relevant to the coin being analyzed, performing technical and sentiment analysis, and using this information to train an LSTM price prediction model. The model, technical indicators, and sentiment scores are then used to generate trading recommendations (`Strong Buy`, `Weak Buy`, `Strong Sell`, `Weak Sell`, or `Hold`). The dashboard also includes interactive charts and visualizations to aid in understanding the data and model predictions.

Currently, the app supports Bitcoin analysis and the following exchanges:

- Bitstamp
- Coinbase
- Gemini
- Kraken

Support for additional coins and exchanges may be added in the future.

This version uses PyTorch to train the LSTM model. A TensorFlow version is available [here](https://github.com/afadlallah/crypto-trading-assistant-tensorflow).

## Features

- Real-time crypto data fetching from various exchanges (`ccxt`)
- Technical analysis with multiple indicators
- Sentiment analysis of crypto-related news (`newsapi-python`, `nltk`)
- Price prediction using an LSTM model (`pytorch`, `scikit-learn`)
- Trading signal generation
- Interactive charts and visualizations (`plotly`, `matplotlib`, `streamlit`)

## Getting Started

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/afadlallah/crypto-trading-assistant-pytorch.git
   cd crypto-trading-assistant-pytorch
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Get a free News API key:

   Create a News API account and you'll immediately be issued a key: [https://newsapi.org/register](https://newsapi.org/register).

   The Developer plan is free and allows 100 API requests per day as long as you're using it for testing and development.

5. Set up environment variables:

   Rename the `.env.example` file to `.env` and add your News API key:

   ```
   NEWSAPI_KEY=your_newsapi_key_here
   ```

### Usage

Run the Streamlit app:

`streamlit run assistant.py`

Navigate to the provided local URL in your web browser to use the assistant.

## Project Structure

- `assistant.py`: Entry point of the application
- `analysis/sentiment.py`: Sentiment analysis of news data
- `analysis/technical.py`: Technical analysis calculations
- `data/collector.py`: Data collection from various sources
- `ml/predictor.py`: Machine learning model for price prediction
- `trading/signal_generator.py`: Trading signal generation
- `ui/dashboard.py`: Main Streamlit dashboard

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).