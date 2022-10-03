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

![](https://i.imgur.com/uEcNKSu.png)

(I use picture instead latex, because I don't know why the github can't show these progress)

There is the table of risk function for each stock

||meta  |goog   |    amzn   |    nflx   | aapl|
|-|-|-|-|-|-|
|100 |2.885279|2.318868| 3.271977|**4.041021**|2.143688|
200 |**2.607665**|1.899779| 2.387325| 2.506661|1.857939|
500| 2.832774|**3.045543**| 2.097889| 2.429130|2.394672|
1000|3.981930|3.367472| 3.376081| 3.636639|**4.435451**|
2000|5.943189|5.081090|13.596966|**16.554674**|7.147849|

Unfortunately, each stock doesn't have any apparent stock risk. But the accepted fact is that FAANG has really grown up in the past 6 years, especially the two new companies AMAZ and NFLX are the most unstable and are also bull markets.

    Code explain
    In the code block below, I will calculate the difference of stock first, and implement the Risk Function with pandas's Dataframe. Finally append it in to a Dataframe for make a table!
```python3
prices = pd.read_csv("../data/stock.csv", index_col=0)
length = [100, 200, 500, 1000, 2000]
Risk_Table = pd.DataFrame(columns=prices.columns)

for l in length:
    test = prices.iloc[-l:].rebase()
    Dnext = test[ 1:].reset_index(drop=True)
    Dthis = test[:-1].reset_index(drop=True)
    Risk_each_day = (Dnext - Dthis)**2
    Risk_square = Risk_each_day.sum(axis=0)/test.shape[0]
    Risk = Risk_square**(1/2)
    Risk.name = f"{l}"
    Risk_Table = pd.concat([Risk_Table, Risk.to_frame().T])
print(Risk_Table)
```

### Recent Years

After I draw
||2 months| 1 year |

![](../src/Recent2MonthPrice.png)
![](../src/Recent1YearPrice.png)


![](https://i.imgur.com/wjYecB1.gif)
