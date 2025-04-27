import warnings
from transformers import pipeline

from alpaca.data.live.news import NewsDataStream

from decouple import config

warnings.filterwarnings("ignore")
api_key = config("ALPACA_KEY")
secret_key = config("ALPACA_SECRET")
# live_news_url = config("ALPACA_WSS_URL")

wss_client = NewsDataStream(api_key=api_key, secret_key=secret_key) # url_override=URL(live_news_url))

classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')


async def news_data_handler(news):
    """Will fire each time news data is published
    """

    summary = news.summary
    headline = news.headline

    relevant_text = summary + headline
    sentiment = classifier(relevant_text)[0]

    if sentiment['label'] == 'POSITIVE' and sentiment['score'] > 0.95:
        #rest_client.submit_order("AAPL", 100)
        print(f'Positive news: {sentiment}')
        print(f"{headline}\n {summary}")
    elif sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.95:
        #rest_client.submit_order(“AAPL”, -100)
        print(f'Negative news: {sentiment}')
        print(f"{headline}\n {summary}")

wss_client.subscribe_news(news_data_handler, '*')

wss_client.run()
