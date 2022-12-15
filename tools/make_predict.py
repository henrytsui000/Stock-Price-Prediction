import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt
import time

news = pd.read_csv("./data/news.csv")
price = pd.read_csv("./data/stock.csv")

revised_news = pd.DataFrame(columns=[["symbol", "date", "content", "match", "sentiment", "score"]])
for title in ["symbol","content", "match", "sentiment"]:
    revised_news[title] = news[title]
revised_news["score"] = news["match"] * news["sentiment"]
revised_news["date"] = pd.to_datetime(news['date']) + pd.to_timedelta(news['time'])

close_time_USTime = 16
close_time_UTC0 = close_time_USTime+5
shift = 24-close_time_UTC0
price_merge = pd.DataFrame()
usdatecol = pd.DataFrame()
for idx, new in tqdm(news.iterrows()):
    date = new["date"]      # UTC+0
    times = new["time"]
    match =new["match"]
    senti = new["sentiment"]
    cb = date +" "+ times
    usdate = dt.datetime.strptime(cb, '%Y-%m-%d %H:%M:%S')
    usdate += dt.timedelta(hours=shift)
    usdateindex = dt.datetime.strftime(usdate,'%Y-%m-%d')   # shift UTC+0 time to match 00:01 open and 24:00 close 
    usdateindex_series = pd.Series(usdateindex)
    usdatecol = pd.concat([usdatecol, usdateindex_series])
    symbol = new["symbol"].lower()
    symbol = symbol if not symbol == "googl" else "goog"
    dtformat = dt.datetime.strptime(usdateindex, '%Y-%m-%d').date()    
    while True:
        if price['Date'].eq(usdateindex).any():
            break
        else:
            dtformat+=dt.timedelta(days=1)      #choose news between 00:00 to 23:59 in usdateindex date
            usdateindex = dt.datetime.strftime(dtformat,'%Y-%m-%d')     #update usdateindex
    row = price[price["Date"]==usdateindex]
    nowprice = price.iloc[row.index][[symbol]]
    preprice = price.iloc[row.index-1][[symbol]]
    pre2dprice = price.iloc[row.index-2][[symbol]]
    pre3dprice = price.iloc[row.index-3][[symbol]]
    nextprice = price.iloc[row.index+1][[symbol]]
    pz = pd.concat([pre3dprice,pre2dprice,preprice,nowprice,nextprice]).T
    pz.columns = ["pre3dprice","pre2dprice","preprice","nowprice","nextprice"]
    price_merge = pd.concat([price_merge, pz])

usdatecol.reset_index(level=0, inplace=True)
revised_news["date"] = usdatecol[0]
revised_news = revised_news.reset_index(drop=True)
revised_news.columns = [x[0] for x in revised_news.columns]
price_merge = price_merge.reset_index(drop=True)
merge = pd.concat([revised_news,price_merge], axis=1)
merge.to_csv("./data/predict_dataset.csv",index=False)

