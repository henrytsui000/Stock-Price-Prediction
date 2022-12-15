import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import pandas as pd
import argparse

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="bert-base-uncased", help="model name")
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-b", "--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("--data", type=str, default="../data/predict_dataset.csv")
    parser.add_argument("--weight", type=str, default="../pretrained/bert_weight.pt")

    return parser
args = make_parser().parse_args()

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

args = make_parser().parse_args(args=[])
model = Bert4price(args.name)
model = model.to(args.device)
model.load_state_dict(torch.load(args.weight))
model.eval()

merge = pd.read_csv(args.data, index_col=0)
dataset = Stock(merge, args.name, args.max_len)
loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1) 
for idx in range(1, 17): merge[f"f{idx:02d}"] = 0

tokenizer = BertTokenizer.from_pretrained(args.name)
model.eval()
cnt, sz = 0, merge.shape[0]
with torch.no_grad():
    for value, content in tqdm(loader):
        text, mask = content["input_ids"].squeeze(1), content["attention_mask"]
        text, mask = text.to(args.device), mask.to(args.device)
        value = value.to(args.device)

        emb, opt = model(text, mask)
        merge.iloc[cnt:min(cnt+args.batch_size, sz), -16: -1] = emb.detach().cpu().numpy()
        cnt += args.batch_size
merge.to_csv("../data/new_pred.csv")