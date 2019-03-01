import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import random

if __name__ == "__main__":

    flist = ["CIG_RJFK_0102_train.csv", "CIG_RJFK_0304_train.csv"
             ,"CIG_RJFK_0506_train.csv","CIG_RJFK_0708_train.csv"
             , "CIG_RJFK_0910_train.csv", "CIG_RJFK_1112_train.csv"]

    #
    # file loop
    #
    for file in flist:
        #
        # データ読み込み
        #
        filename = file
        dat = pd.read_csv(filename)

        #==========================
        # データ数を手で減らす
        #==========================
    
        #
        # カテゴリ別に出現回数を調べる
        #
        category = dat['CIG_category'].tolist()
        # print(category)

        c = collections.Counter(category)
        print(c)

        # 出現回数の多い順2番目から格納する
        c2 = c.most_common()[1:]
        # 最大値を除いて、残りの平均をとる
        hist_mean = int(np.mean([c2[i][1] for i in range(len(c2))]))
        print("hist_mean=", hist_mean)
        max_hist_value = c.most_common()[0][0]
        print("most common=", max_hist_value)
        
        #
        # 最大値をとるカテゴリーを平均値まで削る
        #
        print(dat[dat['CIG_category'] == max_hist_value].sample(hist_mean))
        dat6 = dat[dat['CIG_category'] == max_hist_value].sample(hist_mean)
        dato = dat[dat['CIG_category'] != max_hist_value]
        dat0 = pd.concat([dat6, dato], axis=0)

        print(len(dat), len(dat0), len(dat6))

        #
        # データ書き出し
        #
        oname = filename[0:14]+"mean"+filename[13:]
        print(oname)
        dat0.to_csv(oname, index=False)
