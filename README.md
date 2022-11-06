# DataScienceProject
# About

This is a project about my data science course, the main idea is a AI model through fuse the stock news title and past stock price to predict the stock price in the future.

# Report

You can find the report in this table.

|      | HW1 | HW2| HW3 | HW4 | HW5 |
| ---- | -------------------------------- | -------------------- | --- | - | - |
| Link | [PDF](./HWreport/HW1_109511068.pdf) | [MD](./HWreport/hw2.md) |     |

# Set up environment
```bash
$git clone git@github.com:henrytsui000/DataScienceProject.git
$cd $THIS_PROJECT
$conda create -n IDS python=3.8
$conda activate IDS
$pip install -r requirements.txt
$conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
# install pytorch, please choose your cuda version
```

# Prepare Data
There are two ways to get the news data and the stock price data:

1. Run the python code to get the data. Your may apply an account at marketaux, and add a .env file which indicate the TOKEN
```bash
$python tools/update_data.py
$python tools/update_news.py
```

2. Download the files which I upload to the Google Drive.
```bash
$gdown ${news.csv}
$gdown ${price.csv}
```

And make the file like this:
```
./data
├── News
│   ├── FAANG_STOCK_NEWS.csv
│   └── news.csv
└── Stock
    └── stock.csv

2 directories, 3 files
```

# Visualize Data
### [More Detail](visulize/README.md)
## Correlation GIF
![](./src/correlation/corr2.jpg)

# Fine Tune BERT

# Go Predict

# Experiment

# Acknowledgement

This visiulize source code is based on FFN, Numpy, Pandas. Thanks for their wonderful works.

The code for predict stock news is based on BERT, Thanks for GOOGLE.
