from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pandas as pd


class SentimentAnalysis:
    def __init__(self):
        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()

    def analyze_text(self, text):
        return self.sia.polarity_scores(text)

    def analyze_dataframe(self, df, text_column):
        df['sentiment'] = df[text_column].apply(lambda x: self.analyze_text(x)['compound'])
        return df

    def get_daily_sentiment(self, df, date_column, sentiment_column):
        df[date_column] = pd.to_datetime(df[date_column])
        daily_sentiment = df.groupby(df[date_column].dt.date)[sentiment_column].mean()

        scaled_sentiment = (daily_sentiment - daily_sentiment.mean()) / daily_sentiment.std()
        return scaled_sentiment