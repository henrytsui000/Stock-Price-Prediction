import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline, Rbf, make_interp_spline
from sklearn.linear_model import LinearRegression
from argparse import ArgumentParser

find_mask = lambda array : np.nonzero(array)[0]
DIS = lambda ground, inter : np.around(np.sum(np.abs(ground-inter)), decimals = 2)
inter_func = {"interp1d": interp1d,
              "UnivariateSpline" : UnivariateSpline, 
              "Rbf" : Rbf, 
              "make_interp_spline" : make_interp_spline,
              "LinearRegression": None}

def make_parser():
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", type = str, default="./data/Stock/stock.csv")
    parser.add_argument("-m", "--mask", type = float, default=0.1)
    parser.add_argument("-d", "--date", type = int, default=1000)
    parser.add_argument("-s", "--stock",type = str, default="META, GOOG, AMZN, NFLX, AAPL")
    parser.add_argument("-f", "--force", action="store_false")
    return parser

def read_data(args):
    try:
        SD = pd.read_csv(args.path, index_col=0)[-args.date:]
        return SD, SD.columns
    except :
        print('Error: stock.csv was empty. Please Run tools/updata_stock.py')

def mask_data(args, SD, title):
    # SDM stock_data_masked
    mask_percent = args.mask
    SD = SD.to_numpy()
    SDM = SD.copy()

    MASK = np.random.choice([True, False], size=SDM.shape, p=[mask_percent, 1-mask_percent])
    MASK[0, :] = MASK[-1, :] = False
    SDM = np.ma.masked_array(SDM, mask=MASK)
    return SD, SDM, MASK, title

def INTER(sdm, mask, func = interp1d):
    y_axis = np.delete(sdm, mask)
    interp_func = func(find_mask(~mask), y_axis)
    mask_val = interp_func(find_mask(mask))
    sdm[find_mask(mask)] = mask_val
    return np.around(sdm, decimals=2)

def LR(sdm, mask):
    y_axis = np.delete(sdm, mask)
    model = LinearRegression()
    model.fit(find_mask(~mask).reshape(-1, 1), y_axis)
    mask_val = model.predict(find_mask(mask).reshape(-1, 1))
    sdm[find_mask(mask)] = mask_val
    return np.around(sdm, decimals=2)

def main(args, SD, SDM, MASK, title):
    stock_avg = np.mean(SD, axis=0)
    loss_table = pd.DataFrame(columns=title, index=[])
    for inter_name, func in inter_func.items():
        loss_queue = dict()
        for col, col_name in enumerate(title):
            if inter_name == "LinearRegression":
                inter_res = LR(SDM[:, col].copy(), MASK[:, col].copy())
            else:
                inter_res = INTER(SDM[:, col].copy(), MASK[:, col].copy(), func)
            inter_dis = DIS(SD[:, col], inter_res)
            loss_queue[col_name] = inter_dis
        loss_queue = pd.DataFrame(loss_queue, columns=title, index=[inter_name])
        loss_table = pd.concat([loss_table, loss_queue])
    loss_table = loss_table.div(stock_avg, axis=1)
    print(loss_table)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args, *mask_data(args, *read_data(args)))
