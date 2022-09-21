import pandas as pd
import ffn
import os
import datetime as dt
import parser

# table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# print(table)
PATH = "./src/stock_data/stock.csv"


def check_dir():
    if not os.path.isdir("./src"): os.mkdir("src")
    if not os.path.isdir("./src/stock_data"): os.mkdir("src/stock_data")

def read_data(start_day = dt.datetime(2022, 9, 10), stock_name = ""):
    stock_name = "SPY:Open,SPY:High,SPY:Low,SPY:Close,SPY"
    if os.path.isfile(PATH):
        stock = pd.read_csv(PATH)
    else :
        stock = ffn.get(stock_name, start = start_day)

    print(type(stock))
    # nowday = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    # endday = stock.index[-1]
    # if (nowday > endday) :
    #     # 補日子
    #     stock = ffn.get("SPY:Open,SPY:High,SPY:Low,SPY:Close,SPY", start=endday,)

    stock.to_csv(PATH)

    
def main():
    check_dir()
    read_data()

if __name__ == "__main__":
    main()
# df = table[0]
# stockdata = df['Symbol'].to_list()
# full_stock_data = ffn.get(stockdata, '2010-01-01', '2021-03-03')
# print(full_stock_data['Volume'])