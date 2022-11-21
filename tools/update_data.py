from ast import parse
import ffn
import os
import pandas as pd
import datetime as dt
from argparse import ArgumentParser

def make_parser():
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", type = str, default="./data/stock.csv")
    parser.add_argument("-s", "--stock",type = str, default="META, GOOG, AMZN, NFLX, AAPL" )
    parser.add_argument("-f", "--force", action="store_false")
    parser.add_argument('-d', '--date', type = lambda s: dt.datetime.strptime(s, '%Y-%m-%d'), 
                                            default = dt.datetime(2012, 1, 1))
    return parser

def check_dir():
    if not os.path.isdir("./data"): os.mkdir("./data")
    if not os.path.isdir("./data/Stock"): os.mkdir("./data/Stock")

def read_data(args):
    if os.path.isfile(args.path) and args.force :
        stock = pd.read_csv(args.path, index_col = 0).round(4)
        stock.index = pd.to_datetime(stock.index)
        new_stock = ffn.get(args.stock, start = stock.index[-1],).round(4)
        stock = pd.concat([stock, new_stock]).drop_duplicates()
    else :
        stock = ffn.get(args.stock, start = args.date).round(4)

    stock.to_csv(args.path, index=True)

def main(args):
    check_dir()
    read_data(args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)