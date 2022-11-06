import requests
import json
import os
from dotenv import load_dotenv
import pandas as pd
import datetime as dt
from tqdm import trange
load_dotenv()

TOKEN = os.getenv('TOKEN')
try:
    news = pd.read_csv("../data/News/news.csv")
except FileNotFoundError:
    news = pd.DataFrame(columns=["symbol","date","time","uuid","match","sentiment","content"])
API_site = "https://api.marketaux.com/v1/news/all"
others_setting = "filter_entities=true&language=en"
symbols = ["META", "AMZN", "AAPL", "NFLX", "GOOGL"]
stocknews = dict()

if news.shape[0] == 0:
    last_day = 0
else :
    last_day = news["date"].sort_values(ascending=True).iloc[0]
    last_day = dt.date.today() - dt.datetime.strptime(last_day,  "%Y-%m-%d").date()
    last_day = last_day.days

def update_news(news, stocknews):
    news_dict = dict()
    for symbol, stocknew in stocknews.items():
        if "error" in stocknew:
            return news
        for new_info in stocknew["data"]:
            news_dict["symbol"] = symbol
            news_dict["uuid"] = uuid = new_info["uuid"]
            news_dict["content"] = new_info["title"]
            time = dt.datetime.strptime(new_info["published_at"], '%Y-%m-%dT%H:%M:%S.000000Z')
            news_dict["date"] = dt.datetime.strftime(time, "%Y-%m-%d")
            news_dict["time"] = dt.datetime.strftime(time, "%H:%M:%S")
            score = new_info["entities"][0]
            news_dict["match"] = score["match_score"]
            news_dict["sentiment"] = score["sentiment_score"]
            df_news_dict = pd.DataFrame([news_dict])
            if not news["uuid"].str.contains(uuid).any():
                news = pd.concat([news, df_news_dict], ignore_index=True)
    return news

for day in trange(last_day, last_day+18, desc="updateing news"):
    yesterday = dt.datetime.now() - dt.timedelta(day)
    yesterday = dt.datetime.strftime(yesterday, '%Y-%m-%d')
    for symbol in symbols:
        reqstr = f'{API_site}?symbols={symbol}&{others_setting}&published_before={yesterday}&api_token={TOKEN}'
        data = requests.get(reqstr, timeout=5)
        stocknews[symbol] = json.loads(data.text)
    news = update_news(news, stocknews)

print("Finish update news")
news.to_csv("../data/News/news.csv", index=0)