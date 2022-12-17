import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class Bert2Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_down = nn.Sequential(
                                nn.Linear(320, 32),
                                nn.ReLU(),
                                nn.Linear(32, 3),
                                nn.ReLU()
                            )
        self.predict = nn.Linear(6, 1)
        self.relu = nn.ReLU()
    def forward(self, news, price):
        news = torch.flatten(news, start_dim=1).to(torch.float32)
        price = torch.flatten(price, start_dim=1).to(torch.float32)
        news = self.feature_down(news)
        x = torch.cat((news, price), 1)
        x = self.relu(self.predict(x))
        out = torch.flatten(x)
        return out
    
class Bert2price(nn.Module):
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
