##
## cf_plot.py: Confidence factorごとにprecision, recallをプロットする
##
##             Mar. 25. 2019, M. Yamada
##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":


    #
    # read data
    #
    filename = 'cf_SMOTE100.dat'
    df = pd.read_csv(filename)

    print(df.columns)

    #
    # plot precision 
    #
    width = 0.25
    x1 = [1-width, 2-width, 3-width, 4-width, 5-width, 6-width]
    y1 = df['precision(0.5)']

    plt.bar(x1, y1, color='b', width = width, label = '0.5', align="center")

    x2 = [1, 2, 3, 4, 5, 6]
    y2 = df['precision(0.7)']

    plt.bar(x2, y2, color='g', width = width, label = '0.7', align="center")

    x3 = [1+width, 2+width, 3+width, 4+width, 5+width, 6+width]
    y3 = df['precision(0.9)']

    plt.bar(x3, y3, color='r', width = width, label = '0.9', align="center")

    # 凡例
    plt.legend(loc=2)

    # 軸の目盛りを置換
    plt.ylim(0, 1)
    plt.title("Precision")
    plt.xlabel("Months")
    plt.xticks([1, 2, 3, 4, 5, 6], df['Months'])

    
    plt.show()

    #
    # plot recall
    #
    width = 0.25
    x1 = [1-width, 2-width, 3-width, 4-width, 5-width, 6-width]
    y1 = df['recall(0.5)']

    plt.bar(x1, y1, color='b', width = width, label = '0.5', align="center")

    x2 = [1, 2, 3, 4, 5, 6]
    y2 = df['recall(0.7)']

    plt.bar(x2, y2, color='g', width = width, label = '0.7', align="center")

    x3 = [1+width, 2+width, 3+width, 4+width, 5+width, 6+width]
    y3 = df['recall(0.9)']

    plt.bar(x3, y3, color='r', width = width, label = '0.9', align="center")

    # 凡例
    plt.legend(loc=2)

    # 軸の目盛りを置換
    plt.ylim(0, 1)
    plt.title("Recall")
    plt.xlabel("Months")
    plt.xticks([1, 2, 3, 4, 5, 6], df['Months'])

    
    plt.show()
    
