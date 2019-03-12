¥#########################################################################
# make_balance_test.py: カテゴリごとのデータ数を揃える実験。
#                      1) 手動で突出したカテゴリを削る、
#                      2) SMOTEを使ったdata augmentation
#                      を試す予定。
#
#                      v01: 手動で突出したカテゴリ(6)を1~5の平均に合わせる
#                      v02: v01を関数化
#                      v03: SMOTEで計算
#########################################################################

import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import random

## SMOTE
from imblearn.over_sampling import SMOTE

#========================================================
# 手動で突出カテゴリを切り詰めたデータフレームを返す
#========================================================
def balance_by_hand(df):

    #
    # カテゴリ別に出現回数を調べる
    #
    category = df['CIG_category'].tolist()
    c = collections.Counter(category)    

    # 出現回数の多い順2番目から格納する
    c2 = c.most_common()[1:]
    # 最大値を除いて、残りの平均をとる
    hist_mean = int(np.mean([c2[i][1] for i in range(len(c2))]))
    max_hist_value = c.most_common()[0][0]  # 最大値の出現数

    #
    # 最大値をとるカテゴリーを平均値まで削る
    #
    print(df[df['CIG_category'] == max_hist_value].sample(hist_mean))
    dat6 = df[df['CIG_category'] == max_hist_value].sample(hist_mean)
    dato = df[df['CIG_category'] != max_hist_value]
    df_balanced = pd.concat([dat6, dato], axis=0)

    
    return df_balanced

#==========
# Main 
#==========

if __name__ == "__main__":

    flist = ["CIG_RJFK_0102_train.csv", "CIG_RJFK_0304_train.csv"
             ,"CIG_RJFK_0506_train.csv","CIG_RJFK_0708_train.csv"
             , "CIG_RJFK_0910_train.csv", "CIG_RJFK_1112_train.csv"]

    #----------------
    # file loop
    #----------------
    for file in flist:
        #
        # データ読み込み
        #
        filename = file
        dat = pd.read_csv(filename)

        #==========================
        # データ数を手で減らす
        #==========================
        dat0 = balance_by_hand(dat)

        print(len(dat), len(dat0))

        #
        # データ書き出し
        #
        oname = filename[0:14]+"mean"+filename[13:]
        print(oname)
        dat0.to_csv(oname, index=False)

        #==========================
        # SMOTEでバランスする
        #==========================
        #ratio = {1:100, 2:100, 3:100, 4:100, 5:100, 6:100}
        # smote = SMOTE(kind='svm', random_state = 71)
        #smote = SMOTE(random_state = 71)

        #Y = dat["CIG_category"]        
        #X = dat.drop(["CIG_category","station_code"], axis=1)

        # NaN除去
        #X = X.fillna(0)
        #Y = Y.fillna(0)
        #c = collections.Counter(Y)
        #print(c)

        #X_resampled, Y_resampled = smote.fit_sample(X, Y)

        #c = collections.Counter(Y_resampled)
        #print(c)

