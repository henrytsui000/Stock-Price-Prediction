import pandas as pd
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('agg') 


def get_price(path = "./data/stock.csv"):
    prices = pd.read_csv(path, index_col=0)
    return prices

def vis(heatmap, cmap = "YlGnBu"):
    swarm_plot = sns.heatmap(heatmap, cmap=cmap , vmin=-1, vmax=1)
    fig = swarm_plot.get_figure()

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    plt.clf()
    
    return img

def draw(offset = 100):
    prices, cnt = get_price(), 0
    for i in range(-300, 0, 5):
        img = vis(prices[i-offset:i].corr())
        cv2.imshow("Heatmap", img)
        cnt += 1
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    draw()