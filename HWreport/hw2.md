Data Science HW2
===

## [HOMEPAGE](../README.md)
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

#### Code explain
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

After I draw the price in 2 months and 1.5 years, I found that every stock has a big correlation with others.
For example, META(FB) and GOOG(google) in recent 2 months totally have the same rate of ups and downs. It's intuitive, because there are same type of the company. So they face the same marketing problems.Another example is in 1.5 years figure, the would ups and downs in same time.

The important thing is rebase function, we use this for reset the stock value with same start price. Thus, we can easier to observed the fluctuation.

#### Code explain
Thanks for FFN extension, I con't need to write the rebase function, and FFN also have a function named plot to print the figure on screen without matplotlib.

```python
def slide_windows(stock):
    return stock.rolling(10).mean(std = 5)
prices = pd.read_csv("../data/stock.csv", index_col=0)
slide_windows(prices).iloc[-400:, :].rebase().plot()
```

![](../src/Recent2MonthPrice.png)
2 months stock

![](../src/Recent1YearPrice.png)
1.5 years stock

### Correlation
The next thing is I think there must be a correlation in each stocks, so I made a gif indicate coreelation. The output is suprise me. All of the company have high correlation in period.

each frame is $Day_i$ ~ $Day_{i+100}$ 

for each 2022/1/1 to 2022/9/23

![](https://i.imgur.com/wjYecB1.gif)

#### Code explain

Actually, the hardest code implementation is this blocks. First of all, I make the heapmap which color-map is "YlGnBu". And save the figure in plot so on. Finally, transform the image to buffer that could resolve by cv2, so that can show on screen!

```python
offset = 100
matplotlib.use('agg') 
cnt = 0
for i in range(-300, 0, 5):
    heatmap = prices[i-offset:i].corr()
    swarm_plot = sns.heatmap(heatmap, cmap="YlGnBu" , vmin=-1, vmax=1)
    fig = swarm_plot.get_figure()

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    plt.clf()
    cv2.imshow("Heatmap", img)
    cv2.imwrite(f"./out/{cnt:03d}.jpg", img)
    cnt += 1
    cv2.waitKey(1)
cv2.destroyAllWindows()
```

## Sum up / Future work
In a nutshell, I write serveral code to analysis the stock data, special in the grow rate and the correlation with each others. I think the history price is not enough to predict the data in the future. Thus I would fuse the daily news, I think the method is give the title score represent positive/negative for the company.
![](../src/roberta.png)


## MOT dataset
This is the dataset I introduced last homework, and I think the course to analysis 2D dataset is in the next month, So I decide to postpone the analysis to next assignment. 
## MIMIC III
Go through serveral day to apply the access of physionet. It show that need some day for audit mine identity. Thus, I can't analysis the dataset now.
