import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from itertools import tee
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

class Stock(Dataset):
    def __init__(self, df, news_max = 20) -> None:
        features, prices, values = [], [], []
        
        date = df["date"]
        symbols = df["symbol"]
        date, symbols = list(set(date)), list(set(symbols))
        for days in tqdm(date):
            for symbol in symbols:
                day = df["date"] == days
                sym = df["symbol"] == symbol
                day_data = df[day & sym]
                news_num = len(day_data)
                if news_num == 0: continue
                feature = pd.concat([day_data[f"f{idx+1:02d}"] 
                                        for idx in range(16)], axis=1).to_numpy()
                price = [day_data[f"pre{idx}dreturn"].to_numpy() for idx in range(2, -1, -1)]
                price = np.concatenate([price], axis=0)[:,0]
                value = day_data["nextreturn"].to_numpy()[0]
                if news_num > news_max:
                    choice = np.random.choice(news_num, news_max, replace=False)
                    feature, news_num = feature[choice], news_max
                feature = np.pad(feature, [(0, news_max - news_num), (0, 0)])
                features.append(feature)
                prices.append(price)
                values.append(value)
                
        self.len = len(features)
        self.features, self.prices, self.values = features, prices, values
        
    def __getitem__(self, idx):
        return self.features[idx], self.prices[idx], self.values[idx]

    def __len__(self):
        return self.len
    
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="bert-base-uncased", help="model name")
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-b", "--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("--data", type=str, default="./data/predict_dataset.csv")
    parser.add_argument("--bert-weight", type=str, default="./pretrained/bert_weight.pt")
    parser.add_argument("--predict-weight", type=str, default="./pretrained/predict.pt")
    parser.add_argument("--news-max", type=int, default=20)

    return parser
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
def cal_ret(pre_day, cur_day):
    return 100 * (cur_day - pre_day) / pre_day    

class Score2Predict(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_down = nn.Sequential(
                                nn.Linear(320, 32),
                                nn.ReLU(),
                                nn.Linear(32, 3),
                                nn.ReLU()
                            )
        self.predict = nn.Linear(6, 1)
    def forward(self, news, price):
        news = torch.flatten(news, start_dim=1).to(torch.float32)
        price = torch.flatten(price, start_dim=1).to(torch.float32)
        news = self.feature_down(news)
        x = torch.cat((news, price), 1)
        x = self.predict(x)
        out = torch.flatten(x)
        return out
    
class Bert2Score(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bert = BertModel.from_pretrained(model)
        self.ft_fc = nn.Sequential(
                        nn.Linear(768, 128),
                        nn.ReLU(),
                        nn.Linear(128, 16),
                    )
        self.vl_fc = nn.Linear(16, 1)
    def forward(self, text, mask):
        _, output = self.bert(input_ids=text, attention_mask=mask,return_dict=False)
        feature = self.ft_fc(output)
        output = self.vl_fc(F.relu(feature))
        return torch.squeeze(feature, 1).double(), torch.squeeze(output, 1).double()

def GoPredict(prereturn, nowprice, news, tokenizer, bert, predict):
    contents = [tokenizer(text,padding='max_length', 
                    max_length = args.max_len, 
                    truncation=True,
                    return_tensors="pt") for text in news]
    with torch.no_grad():
        embs = []
        for content in contents:
            content = content.to(args.device)
            text, mask = content["input_ids"].squeeze(1), content["attention_mask"]
            text, mask = text.to(args.device), mask.to(args.device)
            emb, _ = bert(text, mask)
            embs.append(emb)
        embs = torch.cat(embs)
        embs = nn.ZeroPad2d((0, 0, args.news_max - embs.shape[0], 0))(embs)
        prereturn = torch.tensor(prereturn)
        prereturn = torch.unsqueeze(prereturn, 0).to(args.device)
        embs = torch.unsqueeze(embs, 0).to(args.device)

        ret = predict(embs, prereturn)
        ret = ret.detach().cpu().numpy()
        return (ret[0]/100+1)*nowprice
 

args = make_parser().parse_args()
pd.set_option('mode.chained_assignment', None)

price = pd.read_csv("./data/stock.csv", index_col = 0)
news = pd.read_csv("./data/predict_dataset.csv")
symbols = price.columns
p2n = {
    "meta" : "META",
    "goog" : "GOOGL",
    "amzn" : "AMZN",
    "nflx" : "NFLX",
    "aapl" : "AAPL",
}

tokenizer = BertTokenizer.from_pretrained(args.name)
bert = Bert2Score(args.name)
predict = Score2Predict()
bert.load_state_dict(torch.load(args.bert_weight))
predict.load_state_dict(torch.load(args.predict_weight))
bert = bert.to(args.device).eval()
predict = predict.to(args.device).eval()
   
for symbol in symbols: price[symbol+"_pred"] = np.nan

for didx, date in enumerate(tqdm(price.index)):
    for p_symbol in symbols:
        symbol = p2n[p_symbol]
        day = news["date"] == date
        sym = news["symbol"] == symbol
        day_data = news[day & sym]
        if not len(day_data): continue
        # print("ZERO", date, symbol)
        preprice = [price[p_symbol].iloc[didx + pidx] for pidx in range(-3, 1, 1)]
        prereturn = [cal_ret(pred, d) for pred, d in pairwise(preprice)]
        nowprice = price[p_symbol].iloc[didx]
        new = list(day_data["content"])
        random.shuffle(new)
        nprice = GoPredict(prereturn, nowprice, new[:20], tokenizer, bert, predict)
        # print(nprice)
        price[p_symbol+"_pred"].iloc[didx+1] = nprice.round(3)
        
price.to_csv("./data/predict_value.csv")