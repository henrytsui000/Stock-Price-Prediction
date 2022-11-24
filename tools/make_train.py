import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt
import time

news = pd.read_csv("../data/News/news.csv")
price = pd.read_csv("../data/Stock/stock.csv")

merge = pd.DataFrame(columns=["symbol", "date", "time","content", "preprice", "latprice", "match", "sentiment", "score"])

for title in ["symbol", "date", "time", "content", "match", "sentiment"]:
    merge[title] = news[title]
merge["score"] = news["match"] * news["sentiment"]

idx = 0
new = news.iloc[idx]
print(new)
symbol = new["symbol"].lower()
symbol = symbol if not symbol == "googl" else "goog"
date = new["date"]
time = new["time"]
cb = date +" "+ time
datetime_object = datetime.datetime.strptime(cb, '%Y-%m-%d %H:%M:%S')
print(datetime_object)
mv = datetime.timedelta(hours=5)
datetime_object -= mv
print(datetime_object)
if date in price.index:
    merge.iloc[idx, merge.columns.get_loc('preprice')] = price.loc[date][symbol]
    print(merge.iloc[idx]["preprice"], price.loc[date][symbol])

pzmerge = pd.DataFrame()
for idx, new in tqdm(news.iterrows()):
    date = new["date"]
    time = new["time"]
    match =new["match"]
    senti = new["sentiment"]
    cb = date +" "+ time
    usdate = datetime.datetime.strptime(cb, '%Y-%m-%d %H:%M:%S')
    usdate -= datetime.timedelta(hours=5)
    symbol = new["symbol"].lower()
    symbol = symbol if not symbol == "googl" else "goog"
    dtformat = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    while True:
        if price['Date'].eq(date).any():
            # print(date, price['Date'].eq(date).any())
            break
        else:
            dtformat+=datetime.timedelta(days=1)
            date = datetime.datetime.strftime(dtformat,'%Y-%m-%d')
    row = price[price["Date"]==date]
    if usdate.hour<16:
        pz1 = price.iloc[row.index][[symbol]]
        pz0 = price.iloc[row.index-1][[symbol]]
        pz2 = price.iloc[row.index+1][[symbol]]
    else:
        pz0 = price.iloc[row.index][[symbol]]
        pz1 = price.iloc[row.index+1][[symbol]]
        pz2 = price.iloc[row.index+2][[symbol]]
    pz = pd.concat([pz0,pz1,pz2]).T
    pz.columns = ["pz0","pz1","pz2"]
    pzmerge = pd.concat([pzmerge, pz])

df = news.reset_index(drop=True)
df1 = pzmerge.reset_index(drop=True)
merge = pd.concat([df,df1], axis = 1)
print(merge)
merge.to_csv("../data/merge2.csv")