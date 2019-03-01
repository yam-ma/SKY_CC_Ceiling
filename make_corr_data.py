##
## 正解ラベルと相関の高い変数のみ残す
## 

import numpy as np
import pandas as pd


if __name__ == "__main__":

    #
    # データ読み込み
    #
    filename = "CIG_RJFK_train.csv"
    dat_train = pd.read_csv(filename)

    print(dat_train.head())

    filename = "CIG_RJFK_test.csv"
    dat_test = pd.read_csv(filename)
    
    
    #
    # 相関をとる
    #
    corr = dat_train.corr()['CIG_category']

    print(corr)
    print(type(corr))

    # Nanを落とす
    corr = corr.dropna()

    # 絶対値をとる
    corr = corr.map(lambda x: abs(x))

    # 並べ替え：
    corr = corr.sort_values(ascending=False)

    # 相関係数の絶対値top 30を取る:最初の値は自己相関なので、トータル31列になる
    top30 = corr.index.values[0:31]

    
    df30_train = dat_train[top30]
    df30_test = dat_test[top30]
    print(df30_train.head())

    #
    # データ書き出し
    #
    df30_train.to_csv("CIG_RJFK_top30_train.csv", index=False)
    df30_test.to_csv("CIG_RJFK_top30_test.csv", index=False)    
