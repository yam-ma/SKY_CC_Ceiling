#
# predict_probaの出力取りまとめ
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":


    #
    # データ読み込み
    #
    filename = "conflevel_1112.csv"
    dat = pd.read_csv(filename)

    # header書き換え
    dat.columns = ["predict", "1", "2", "3", "4", "5", "6"]

    print(dat.head())

    #
    # predict_proba出力のトップ2を取り出して差を取る
    #
    diff2 = []
    for i in range(len(dat)):
        pp = dat.iloc[i, 1:7].tolist() 
        # print(sorted(pp, reverse=True))
        pp_sorted = sorted(pp, reverse=True)
        diff_2 = pp_sorted[0]-pp_sorted[1]
        # print(i, diff_2)
        diff2.append(diff_2)

    diff2_over50 = [i for i in diff2 if i >= 0.5]
    print(len(diff2_over50), len(diff2_over50)/len(diff2))
        
    #
    # ヒストグラム
    #
    plt.hist(diff2, bins=10)
    plt.show()
