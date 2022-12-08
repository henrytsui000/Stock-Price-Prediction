import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt

news = pd.read_csv("./data/news.csv")
stock = pd.read_csv("./data/stock.csv", index_col=0)
merge = pd.DataFrame(columns=["symbol", "date", "content", "preprice", "latprice", "match", "sentiment", "score"])

for title in ["symbol", "date", "content", "match", "sentiment"]:
    merge[title] = news[title]
merge["score"] = news["match"] * news["sentiment"]

for idx, new in tqdm(news.iterrows()):
    symbol = new["symbol"].lower()
    symbol = symbol if not symbol == "googl" else "goog"
    date = new["date"]
    if date in stock.index:
        merge.iloc[idx, merge.columns.get_loc('preprice')] = stock.loc[date][symbol]
    next_day = str((dt.datetime.strptime(date, "%Y-%m-%d") + dt.timedelta(days=1)).date())
    if next_day in stock.index:
        merge.iloc[idx, merge.columns.get_loc('latprice')] = stock.loc[next_day][symbol]

merge.to_csv("./data/train_bert.csv")