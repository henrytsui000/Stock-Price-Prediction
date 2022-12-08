import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter
from IPython.display import clear_output
import logging
import os
import argparse

class Stock(Dataset):
    def __init__(self, df, model, max_len) -> None:
        self.df = df
        self.score = [float(score) for score in df["score"]]
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.content = [self.tokenizer(text,padding='max_length', 
                       max_length = max_len, 
                       truncation=True,
                       return_tensors="pt") for text in df["content"]]

    def __getitem__(self, idx):
        return self.score[idx], self.content[idx]

    def __len__(self):
        return self.df.shape[0]

class Bert4price(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bert = BertModel.from_pretrained(model)
        self.fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, text, mask):
        _, output = self.bert(input_ids=text, attention_mask=mask,return_dict=False)
        output = self.fc(output)
        return torch.squeeze(output, 1).double()

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="bert-base-uncased", help="model name")
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-b", "--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("--data", type=str, default="./data/train_bert.csv")

    return parser

def get_loader(args):
    merge = pd.read_csv(args.data, index_col=0)
    merge = shuffle(merge)

    t0, t1, t2 = np.split(merge.sample(frac=1, random_state=42), [int(.8*merge.shape[0]), int(.9*merge.shape[0])])
    dataset = {x: Stock(s, args.name, args.max_len) for x, s in [("train", t0), ("valid", t1), ("test", t2)]}
    loader = {x: DataLoader(dataset[x], batch_size=args.batch_size, num_workers=24, shuffle=True) 
                                                for x in ["train", "valid", "test"]}
    logging.info("Finish Loading data")
    return loader

def helper(args):
    COMMENT = f"lr{args.lr}-B{args.batch_size}-EPS{args.epochs}"
    bert = Bert4price(args.name)
    bert = bert.to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(bert.parameters(), lr = args.lr, weight_decay=1e-9)
    lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    writer = SummaryWriter(comment=COMMENT)
    logging.info("Finish Build model")
    return bert, criterion, optimizer, lr_sch, writer

def train(model, criterion, optimizer, lr_sch, writer, loader, args):
    min_loss = 1e10
    Dist = nn.L1Loss()
    for epoch in range(args.epochs):
        for state in ["train", "valid"]:
            tqdm_bar = tqdm(loader[state], leave=False)
            tqdm_bar.set_description(f"[{epoch+1}/{args.epochs}]")
            loss_list, dist_list = [], []
            for value, content in tqdm_bar:
                text, mask = content["input_ids"].squeeze(1), content["attention_mask"]
                text, mask = text.to(args.device), mask.to(args.device)
            
                value = value.to(args.device)
                output = model(text, mask)
                loss = criterion(output, value)
                loss_list.append(loss.item())
                dist = Dist(output, value)
                dist_list.append(dist.item())
                if state == "train":
                    optimizer.zero_grad() 
                    loss.backward()
                    optimizer.step()
                    lr_sch.step()
            avg_loss = np.average(np.array(loss_list))
            avg_dist = np.average(np.array(dist_list))
            if avg_loss < min_loss:
                min_loss = avg_loss
                if not os.path.exists("./pretrained"):
                    os.mkdir("./pretrained")
                torch.save(model.state_dict(), f"./pretrained/bert_weight.pt")
            writer.add_scalar(f"{state}/loss", avg_loss, epoch)
            writer.add_scalar(f"{state}/dist", avg_dist, epoch)
    else:
        logging.info(f"Finish Training {args.epochs} epochs with loss:{min_loss}")

def main(args):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    logging.info("start training news to price process")
    loader = get_loader(args)
    function = helper(args)
    train(*function, loader, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)