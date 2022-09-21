from ast import parse
from operator import index, indexOf
from xml.sax import make_parser
import pandas as pd
import ffn
import os
import datetime as dt
from argparse import ArgumentParser

# table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# print(table)

def make_parser():
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", type = str, default="./src/stock_data/stock.csv")
    parser.add_argument("-s", "--stock",type = str, default="SPY:Open,SPY:High,SPY:Low,SPY:Close,SPY" )

    parser.add_argument('-d', '--date', type = lambda s: dt.datetime.strptime(s, '%Y-%m-%d'), 
                                            default = dt.datetime(2022, 9, 10))

    return parser

def check_dir(args):
    if not os.path.isdir("./src"): os.mkdir("src")
    if not os.path.isdir("./src/stock_data"): os.mkdir("src/stock_data")

def read_data(args):
    if os.path.isfile(args.path):
        stock = pd.read_csv(args.path, index_col = 0).round(4)
        stock.index = pd.to_datetime(stock.index)
        new_stock = ffn.get(args.stock, start = stock.index[-1],).round(4)
        stock = pd.concat([stock, new_stock]).drop_duplicates()
    else :
        stock = ffn.get(args.stock, start = args.date).round(4)

    stock.to_csv(args.path, index=True)

    
def main(args):
    check_dir(args)
    read_data(args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)