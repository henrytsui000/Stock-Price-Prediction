# DataScienceProject

This is a project about my data science course, the main idea is a AI model through fuse the stock news title and past stock price to predict the stock price in the future.

# Report

You can find the report in this table.

|      | HW1 | HW2| HW3 | HW4 | HW5 |
| ---- | -------------------------------- | -------------------- | --- | - | - |
| Link | [PDF](./HWreport/HW1_109511068.pdf) | [MD](./HWreport/hw2.md) |     |

# Set up
```bash
$git clone git@github.com:henrytsui000/DataScienceProject.git
$cd $THIS_PROJECT
$conda create -n IDS python=3.8
$conda activate IDS
$pip install -r requirements.txt
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


# Experiment

# Acknowledgement

This source code is based on FFN, Numpy, Pandas. Thanks for their wonderful works.
