import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt
import time

news = pd.read_csv("./data/news.csv")
price = pd.read_csv("./data/stock.csv")

merge = pd.DataFrame(columns=["symbol", "date", "time","content", "preprice", "latprice", "match", "sentiment", "score"])

for title in ["symbol", "date", "time", "content", "match", "sentiment"]:
    merge[title] = news[title]
merge["score"] = news["match"] * news["sentiment"]

pzmerge = pd.DataFrame()
for idx, new in tqdm(news.iterrows()):
    date = new["date"]
    time = new["time"]
    match =new["match"]
    senti = new["sentiment"]
    cb = date +" "+ time
    usdate = dt.datetime.strptime(cb, '%Y-%m-%d %H:%M:%S')
    usdate -= dt.timedelta(hours=5)
    symbol = new["symbol"].lower()
    symbol = symbol if not symbol == "googl" else "goog"
    dtformat = dt.datetime.strptime(date, '%Y-%m-%d').date()
    while True:
        if price['Date'].eq(date).any():
            break
        else:
            dtformat+=dt.timedelta(days=1)
            date = dt.datetime.strftime(dtformat,'%Y-%m-%d')
    row = price[price["Date"]==date]
    if usdate.hour<16:
        nowprice = price.iloc[row.index][[symbol]]
        preprice = price.iloc[row.index-1][[symbol]]
        nextprice = price.iloc[row.index+1][[symbol]]
    else:
        preprice = price.iloc[row.index][[symbol]]
        nowprice = price.iloc[row.index+1][[symbol]]
        nextprice = price.iloc[row.index+2][[symbol]]
    pz = pd.concat([preprice,nowprice,nextprice]).T
    pz.columns = ["preprice","nowprice","nextprice"]
    pzmerge = pd.concat([pzmerge, pz])

df = news.reset_index(drop=True)
df1 = pzmerge.reset_index(drop=True)
merge = pd.concat([df,df1], axis = 1)
merge.to_csv("./data/merge2.csv")