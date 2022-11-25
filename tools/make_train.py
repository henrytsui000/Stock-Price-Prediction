import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt
import time

news = pd.read_csv("./data/news.csv")
price = pd.read_csv("./data/stock.csv")

revised_news = pd.DataFrame(columns=[["symbol", "datentime", "content", "match", "sentiment", "score"]])
for title in ["symbol","content", "match", "sentiment"]:
    revised_news[title] = news[title]
revised_news["score"] = news["match"] * news["sentiment"]
revised_news["datentime"] = pd.to_datetime(news['date']) + pd.to_timedelta(news['time'])


pzmerge = pd.DataFrame()
usdatecol = pd.DataFrame()
for idx, new in tqdm(news.iterrows()):
    date = new["date"]
    times = new["time"]
    match =new["match"]
    senti = new["sentiment"]
    cb = date +" "+ times
    usdate = dt.datetime.strptime(cb, '%Y-%m-%d %H:%M:%S')
    usdate += dt.timedelta(hours=3)
    usdateindex = dt.datetime.strftime(usdate,'%Y-%m-%d %H:%M:%S')
    usdateindex = pd.Series(usdateindex)
    usdatecol = pd.concat([usdatecol, usdateindex])
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
    nowprice = price.iloc[row.index][[symbol]]
    preprice = price.iloc[row.index-1][[symbol]]
    nextprice = price.iloc[row.index+1][[symbol]]
    pz = pd.concat([preprice,nowprice,nextprice]).T
    pz.columns = ["preprice","nowprice","nextprice"]
    pzmerge = pd.concat([pzmerge, pz])

usdatecol.reset_index(level=0, inplace=True)
revised_news["datentime"] = usdatecol[0]
revised_news.to_csv("./data/revised_news.csv")
revised_news = pd.read_csv("./data/revised_news.csv",index_col=0)
df = revised_news.reset_index(drop=True)
df1 = pzmerge.reset_index(drop=True)
merge = pd.concat([df,df1], axis = 1)
merge.to_csv("./data/bertdataset6.csv")

