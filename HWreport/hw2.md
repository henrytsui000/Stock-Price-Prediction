Data Science HW2
===

## Introduction
First of all, I choose to use markdown syntax to write this assignment, because in this assignment, I use python to analyze the relevant content of the assignment. The word format will make the entire layout very confusing when you put the code, and you cannot put dynamic files such as gif or mp4. In this case, I think MarkDown is more suitable (previously, I sent a letter to ask the professor if the format is not restricted)

## Environment
I think I mentioned the structure of the whole project in the last assignment, but there was not much explanation and introduction, and this time I added new code to make the analysis more convenient.

### Setup

It's important to download the file and let your env have same pip package with me.
```bash
$git clone git@github.com:henrytsui000/DataScienceProject.git
$cd ./DataScienceProject
$conda create -n IDS python=3.8
$conda activate IDS
$pip install -r requirement.txt
```

### Update Data
This is the code I mentioned in my last assignment, you can use the following command to update the stock market data under the data folder
```bash
$python update_data.py
````
By default, the python code will append the new data in the end of the data.csv, but you can use the command like this to specific start date or force to reload all the data(but it would cost more time than default mode)

```bash
$python update_date.py --date 20xx-xx-xx --force
$python update_date.py -d 20xx-xx-xx -f
```

## FAANG
FAANG, which is the abbreviation of Facebook, Apple, Amazon, Netflix and Google. The common point of these companies is all of them is about technology company. which is the most interested me combination of the stock market.

All the analysis below it could find the draw method in [Here](../visulize/)!

Because we should visualization the stock price and make it to a video, so this times we use seaborn, opencv-python, matplotlib.
```python
import ffn
import pandas as pd
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
```


### Risk
In this part, I wonder to calculate the stock's Risk. I define risk as the following formula: 

$Risk(stock) = \sqrt{\Sigma_{i=D_{start}}^{D_{end}} \frac{(D_i-D_{i-1})^2}{n}}$

$D_{start}$ := today, 

$D_{end}$ := last year today, 

$n$ := The day from $D_{end}$ to $D_{start}$

Then we can deduce:

$Risk(stock) = \sqrt{\frac{\Sigma_{i=D_{start}}^{D_{end}} (D_i-D_{i-1})^2}{n}} = \sqrt{2\frac{\Sigma_{i=D_{start}}^{D_{end}} D_i^2}{n}-2\Sigma_{i=D_{start}}^{D_{end}}D_iD_{i-1}}=\sqrt{2}\sqrt{(\frac{||D||_2^2}{n}+\Sigma_{i=D_{start}}^{D_{end}}D_iD_{i-1})}$

![](https://i.imgur.com/wjYecB1.gif)
