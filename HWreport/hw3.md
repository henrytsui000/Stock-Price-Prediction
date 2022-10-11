Data Science HW3
================

## [HOMEPAGE](../README.md)

## Introduction

I chose to continue using markdown for this assignment because its convenient embedded image and formula system made it easier for me to analyze and introduce code. At the same time, I don't think it is necessary to use fancy reports like liberal arts, and the focus is on content.

## Environment

Base on the last job, this time no new environment is required, the only thing changed is to install scipy and sklearn. And pull the repo to local.

```sh
$conda activate IDS
$pip install scipy
$pip install scikit-learn
$git pull git@github.com:henrytsui000/DataScienceProject.git
```

## FAANG

FAANG, which is the abbreviation of Facebook, Apple, Amazon, Netflix and Google. The common point of these companies is all of them is about technology company. Which is the most interested me combination of the stock market.

### Missing data
The most fatal thing in the dataset is missing data, which will affect the training of the model. For example, the dataloader will input nan. Usually I choose to drop missing data directly in computer vision, but this problem becomes tricky when the data becomes linear, because linear data cannot have any discontinuities.

Luckily I didn't have missing values for the stock market price I chose, the only thing being that it was closed on weekends, but I guess that's not too important.
To simulate missing data, I chose to randomly mask out a fixed percentage of the data. And supplement the data with common methods in a variety of ways.

## Masking data

The program I wrote can be found [here](../tools/fill_up.py). The first is the masking data I mentioned earlier. I choose to set a $p$ to indicate how many percent of the data I want to mask, and use np.random.choice to make a NxM matrix, containing $p％$ True.
At the same time, I think it would be a wise decision to hollow out by myself, because it can be used as a validation through masking, just like training a neural network. By knowing the outcome of each interpolation method or each interval, the decision is made which method to use.

### Code explain
nd.random.choice is the function which I mention before. Through np.ma.mask, two matrices (numeric matrix and Boolean matrix) can be given as masks to remind me which positions are missing by my simulation.
It would return:
- SD : Stock Data
- SDM : Stock Data Masked
- MASK : bool martix, which position is masked
- title : Inherit read_data, the stock names  

```python3
def mask_data(args, SD, title):
    # SDM stock_data_masked
    mask_percent = args.mask
    SD = SD.to_numpy()
    SDM = SD.copy()

    MASK = np.random.choice([True, False], size=SDM.shape, p=[mask_percent, 1-mask_percent])
    MASK[0, :] = MASK[-1, :] = False
    SDM = np.ma.masked_array(SDM, mask=MASK)
    return SD, SDM, MASK, title
```

## Definition of Loss

It is important to define an evaluation for various imputation of missing values. We know that the amount of missing data will be $fac = D\cdot p$, which $D$ is the number of days, $p$ is the mask ratio. The error of one stock is $Loss(stock) = \sum^{D}_{i}{SD_i-SDM_i}$
Thus the Loss function could be:

$\widehat{Loss}=\frac{Loss}{fac} = \frac{\sum^{D}_{i}{SD_i-SDM_i}}{D\cdot p}$

This formula can be used to measure whether different interpolation methods are accurate for the same stock.
But the next problems come out, there is a lot of different in each stock:
|method|meta|goog| amzn| nflx| aapl|
|-|-|-|-|-|-|
|interp1d|1400.98|449.24| 676.23|3183.65| 531.46|

The reason why is obvious, because different stocks have different bases, so there will be differences in multiples when making up the difference, and the way to solve this problem is to divide their Loss by the average of the stock price:

$\widetilde{Loss} = \frac{\widehat{Loss}}{stock_{avg}} =  \frac{\sum^{D}_{i}{SD_i-SDM_i}}{D\cdot p \cdot stock_{avg}}$

Finally, the loss of these stock is fair:
|method|meta|goog| amzn| nflx| aapl|
|-|-|-|-|-|-|
|interp1d|1.462%|1.261%| 1.636%|1.817%| 1.351%|


### Code explain
Via lambda function, we can define the loss function, and the Normalization Factor(NF) could calculate by numpy function.

```python
LOSS = lambda ground, inter : np.around(np.sum(np.abs(ground-inter)), decimals = 2)
NF = np.mean(SD, axis=0)*args.date*args.mask/100
loss_table = loss_table.div(NF, axis=1)
```

## Method of Interpolation
Thanks for scipy, it extension provide me a easy way to call out interpolation function. The function call by these steps:
1. define the interpolation function by known pair(x, y).
2. Give the function which point(x) is missed.
3. Fill in the value which is predicted.


```python3
find_mask = lambda array : np.nonzero(array)[0]
def INTER(sdm, mask, func = interp1d):
    y_axis = np.delete(sdm, mask)
    interp_func = func(find_mask(~mask), y_axis)
    mask_val = interp_func(find_mask(mask))
    sdm[find_mask(mask)] = mask_val
    return np.around(sdm, decimals=2)
```



## Sum up / Future work