##
## metar_human_plot.py: MetarとBEFORE/AFTERの比較
##                     ：precision, recallをプロットする
##
##             Mar. 26. 2019, M. Yamada
##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":


    #
    # read data
    #
    filename = 'metar_human_v2.dat'
    df = pd.read_csv(filename)

    print(df.columns)

    #
    # plot precision 
    #
    width = 0.25
    x1 = [1-width, 2-width, 3-width, 4-width, 5-width, 6-width]
    y1 = df['precision(before)']

    plt.bar(x1, y1, color='b', width = width, label = 'BEFORE', align="center")

    x2 = [1, 2, 3, 4, 5, 6]
    y2 = df['precision(after)']

    plt.bar(x2, y2, color='g', width = width, label = 'AFTER', align="center")


    # 凡例
    plt.legend(loc=2)

    # 軸の目盛りを置換
    plt.ylim(0, 1)
    plt.title("Precision(v2)")
    plt.xlabel("Months")
    plt.xticks([1, 2, 3, 4, 5, 6], df['Months'])

    
    plt.show()

    #
    # plot recall
    #
    width = 0.25
    x1 = [1-width, 2-width, 3-width, 4-width, 5-width, 6-width]
    y1 = df['recall(before)']

    plt.bar(x1, y1, color='b', width = width, label = 'BEFORE', align="center")

    x2 = [1, 2, 3, 4, 5, 6]
    y2 = df['recall(after)']

    plt.bar(x2, y2, color='g', width = width, label = 'AFTER', align="center")


    # 凡例
    plt.legend(loc=2)

    # 軸の目盛りを置換
    plt.ylim(0, 1)
    plt.title("Recall(v2)")
    plt.xlabel("Months")
    plt.xticks([1, 2, 3, 4, 5, 6], df['Months'])

    
    plt.show()
    
