from datetime import datetime, timedelta
from dotenv import load_dotenv
from newsapi import NewsApiClient
import ccxt
import logging
import os
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, exchange_name='coinbase', newsapi_key=None):
        self.exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        self.blockchain_info_url = "https://api.blockchain.info/charts/"

        # Load environment variables if NEWSAPI_KEY is not provided
        if newsapi_key is None:
            load_dotenv()
            newsapi_key = os.getenv('NEWSAPI_KEY')

        self.newsapi = NewsApiClient(api_key=newsapi_key)

    def fetch_ohlcv(self, symbol='BTC/USD', timeframe='1h', limit=1000):
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a given symbol and timeframe.
        """
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def fetch_multiple_timeframes(self, symbol='BTC/USD', timeframes=['1h', '4h', '1d'], limit=100):
        """
        Fetch OHLCV data for multiple timeframes.
        """
        data = {}
        for timeframe in timeframes:
            data[timeframe] = self.fetch_ohlcv(symbol, timeframe, limit)
        return data

    def fetch_order_book(self, symbol='BTC/USD', limit=20):
        """
        Fetch current order book data.
        """
        order_book = self.exchange.fetch_order_book(symbol, limit)
        timestamp = pd.Timestamp.now()
        bids = pd.DataFrame(order_book['bids'], columns=['Price', 'Amount'])
        asks = pd.DataFrame(order_book['asks'], columns=['Price', 'Amount'])
        bids['Timestamp'] = timestamp
        asks['Timestamp'] = timestamp
        bids.set_index('Timestamp', inplace=True)
        asks.set_index('Timestamp', inplace=True)
        return {
            'bids': bids,
            'asks': asks
        }

    def fetch_recent_trades(self, symbol='BTC/USD', limit=100):
        """
        Fetch recent trades data.
        """
        trades = self.exchange.fetch_trades(symbol, limit=limit)
        df = pd.DataFrame(trades, columns=['timestamp', 'side', 'price', 'amount'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.index.name = 'Timestamp'
        df.columns = df.columns.str.capitalize()
        return df

    def fetch_on_chain_data(self, chart_name, timespan='1year', rolling_average='8hours'):
        """
        Fetch on-chain data from blockchain.info API.
        """
        url = f"{self.blockchain_info_url}{chart_name}"
        params = {
            'timespan': timespan,
            'rolling_average': rolling_average,
            'format': 'json'
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data['values'])
            df.columns = ['timestamp', 'value']
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for {chart_name}: {str(e)}")
            return pd.DataFrame()

    def fetch_multiple_on_chain_metrics(self, metrics=['transactions-per-second', 'mempool-size', 'hash-rate'], timespan='1year'):
        """
        Fetch multiple on-chain metrics.
        """
        data = {}
        for metric in metrics:
            df = self.fetch_on_chain_data(metric, timespan)
            if not df.empty:
                data[metric] = df
            else:
                logger.warning(f"No data available for {metric}")
        return data

    def fetch_news(self, query='bitcoin', days_back=7):
        end_date = datetime.now()
        # start_date = end_date - timedelta(days=days_back)

        all_articles = []

        for i in range(days_back):
            date = end_date - timedelta(days=i)
            articles = self.newsapi.get_everything(q=query,
                                                    from_param=date.strftime('%Y-%m-%d'),
                                                    to=(date + timedelta(days=1)).strftime('%Y-%m-%d'),
                                                    language='en',
                                                    sort_by='publishedAt',
                                                    page_size=15)
            all_articles.extend(articles['articles'])

        df = pd.DataFrame(all_articles)
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], format='%Y-%m-%dT%H:%M:%SZ')
        # print("df:", df.head(2000))
        return df.sort_values('publishedAt')