# DataScienceProject

# About

This is a project about my data science course, the main idea is a AI model through fuse the stock news title and past stock price to predict the stock price in the future.
<!-- 
# Report

You can find the report in this table.

|      | HW1                              | HW2                  | HW3 | HW4 | HW5 |
| ---- | -------------------------------- | -------------------- | --- | --- | --- |
| Link | [PDF](./HWreport/HW1_109511068.pdf) | [MD](./HWreport/hw2.md) |     |     |     | -->

# Set up environment

```bash
$git clone git@github.com:henrytsui000/DataScienceProject.git
$cd $(THIS_PROJECT)
$conda create -n IDS python=3.8
$conda activate IDS
$pip install -r requirements.txt
$conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
# install pytorch, please choose your cuda version
```

# Prepare Data

## Raw Data
There are two ways to get the news data and the stock price data:

1. Run the python code to get the data. You may apply an account at marketaux, and add an .env file which indicate the TOKEN

```bash
$python tools/update_data.py
$python tools/update_news.py
```

2. Use the command below to download the files which I upload to the Google Drive.

```bash
$gdown --folder https://drive.google.com/drive/folders/1WR5bq9gzMtNFHMnTB6dLEm8AeJj8b9ZT?usp=share_link
```

The file structure you downloaded is as follows:

```
./data
├── News
│   └── news.csv
└── Stock
    └── stock.csv 

2 directories, 2 files
```

## Training Data
This step for transform data in news and stock to format data, easier for model reading.
```bash
$python make_bert.py
$python make_regression.py
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
