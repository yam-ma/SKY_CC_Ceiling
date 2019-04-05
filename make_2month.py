##
## 二ヶ月ごとのデータセットを作成する
##
##              v02: 訓練データだけでなく、テスト用データも作成する
##              Feb. 28. 2019, M. Yamada

import numpy as np
import pandas as pd

if __name__ == "__main__":

    #===============
    # 訓練データ
    #===============
    
    #
    # データ読み込み
    #
    filename = "Input/CIG_RJFK_train_add9999.csv"
    dat_train = pd.read_csv(filename)

    print(dat_train.head())

    #
    # date行分割
    #

    # 文字列に直す
    dat_train['date'] = dat_train['date'].apply(lambda x: str(int(x)))
    # print(dat_train['date'])

    # test
    print("December")
    print(dat_train[dat_train['date'].str[4:6] == "12"].date)

    # 切り分け
    dat_0102 = dat_train[
        (dat_train['date'].str[4:6] == "01") |
        (dat_train['date'].str[4:6] == "02")
        ]

    dat_0304 = dat_train[
        (dat_train['date'].str[4:6] == "03") |
        (dat_train['date'].str[4:6] == "04")
        ]

    dat_0506 = dat_train[
        (dat_train['date'].str[4:6] == "05") |
        (dat_train['date'].str[4:6] == "06")
        ]

    dat_0708 = dat_train[
        (dat_train['date'].str[4:6] == "07") | 
        (dat_train['date'].str[4:6] == "08")
        ]
 
    dat_0910 = dat_train[
        (dat_train['date'].str[4:6] == "09") | 
        (dat_train['date'].str[4:6] == "10")
        ]

    dat_1112 = dat_train[
        (dat_train['date'].str[4:6] == "11") | 
        (dat_train['date'].str[4:6] == "12")
        ]

    #
    # データ書き出し
    #
    dat_0102.to_csv("CIG_RJFK_0102_add9999_train.csv", index=False)
    dat_0304.to_csv("CIG_RJFK_0304_add9999_train.csv", index=False)    
    dat_0506.to_csv("CIG_RJFK_0506_add9999_train.csv", index=False)
    dat_0708.to_csv("CIG_RJFK_0708_add9999_train.csv", index=False)    
    dat_0910.to_csv("CIG_RJFK_0910_add9999_train.csv", index=False)
    dat_1112.to_csv("CIG_RJFK_1112_add9999_train.csv", index=False)    

    #===============
    # テストデータ
    #===============
    
    #
    # データ読み込み
    #
    filename = "Input/CIG_RJFK_test_add9999.csv"
    dat_test = pd.read_csv(filename)

    print(dat_test.head())

    #
    # date行分割
    #

    # 文字列に直す
    dat_test['date'] = dat_test['date'].apply(lambda x: str(int(x)))

    # test
    print("December")
    print(dat_test[dat_train['date'].str[4:6] == "12"].date)

    # 切り分け
    dat_0102 = dat_test[
        (dat_test['date'].str[4:6] == "01") |
        (dat_test['date'].str[4:6] == "02")
        ]

    dat_0304 = dat_test[
        (dat_test['date'].str[4:6] == "03") |
        (dat_test['date'].str[4:6] == "04")
        ]

    dat_0506 = dat_test[
        (dat_test['date'].str[4:6] == "05") |
        (dat_test['date'].str[4:6] == "06")
        ]

    dat_0708 = dat_test[
        (dat_test['date'].str[4:6] == "07") | 
        (dat_test['date'].str[4:6] == "08")
        ]
 
    dat_0910 = dat_test[
        (dat_test['date'].str[4:6] == "09") | 
        (dat_test['date'].str[4:6] == "10")
        ]

    dat_1112 = dat_test[
        (dat_test['date'].str[4:6] == "11") | 
        (dat_test['date'].str[4:6] == "12")
        ]

    #
    # データ書き出し
    #
    dat_0102.to_csv("CIG_RJFK_0102_add9999_test.csv", index=False)
    dat_0304.to_csv("CIG_RJFK_0304_add9999_test.csv", index=False)    
    dat_0506.to_csv("CIG_RJFK_0506_add9999_test.csv", index=False)
    dat_0708.to_csv("CIG_RJFK_0708_add9999_test.csv", index=False)    
    dat_0910.to_csv("CIG_RJFK_0910_add9999_test.csv", index=False)
    dat_1112.to_csv("CIG_RJFK_1112_add9999_test.csv", index=False)    

